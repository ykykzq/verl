# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for BucketedWeightSender and BucketedWeightReceiver.

Sender and receiver run in separate processes to match real-world usage
and because CUDA IPC requires distinct processes.
"""

import asyncio
import multiprocessing as mp
import os
import uuid

import pytest
import torch

from verl.utils.device import get_device_name, get_torch_device, is_support_ipc

PROCESS_TIMEOUT = 60

# Use string checks to avoid initializing CUDA in the main pytest process,
# which would make subsequent fork-based multiprocessing in other tests unsafe.
HAS_ACCELERATOR = get_device_name() != "cpu"
HAS_CUDA = "cuda" in get_device_name()


def _unique_zmq_handle():
    return f"ipc:///tmp/test-bwt-{uuid.uuid4().hex}.sock"


def _generate_weights(weight_specs, seed):
    """Deterministically generate weights on the best available device from specs.

    Args:
        weight_specs: list of (name, shape, dtype) tuples
        seed: random seed for reproducibility
    Returns:
        list of (name, tensor_on_device) tuples
    """
    device_name = get_device_name()
    device = torch.device(f"{device_name}:0")
    get_torch_device().manual_seed(seed)
    weights = []
    for name, shape, dtype in weight_specs:
        # Generate in float32 then cast, since torch.randn doesn't support all dtypes
        t = torch.randn(shape, dtype=torch.float32, device=device).to(dtype)
        weights.append((name, t))
    return weights


class _FakeSocket:
    def __init__(self):
        self.messages = []

    def send_pyobj(self, message):
        self.messages.append(message)

    def recv(self):
        return b""


class _FakeTorchDevice:
    def synchronize(self):
        pass


def test_sender_accepts_strided_tensor(monkeypatch):
    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer

    base = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    weight = base[:, 0, :]
    buffer = torch.empty(weight.nbytes, dtype=torch.uint8)
    socket = _FakeSocket()
    sender = bucketed_weight_transfer.BucketedWeightSender(
        zmq_handle="ipc:///tmp/test-bwt-unused.sock",
        bucket_size_mb=1,
        use_shm=True,
    )

    assert not weight.is_contiguous()
    with pytest.raises(RuntimeError):
        weight.view(-1).view(torch.uint8)

    monkeypatch.setattr(sender, "_init_socket", lambda: setattr(sender, "socket", socket))
    monkeypatch.setattr(sender, "_init_buffer", lambda: setattr(sender, "buffer", buffer))
    monkeypatch.setattr(sender, "_cleanup", lambda: None)
    monkeypatch.setattr(bucketed_weight_transfer, "get_torch_device", lambda: _FakeTorchDevice())

    asyncio.run(sender.async_send_weights(iter([("strided", weight)])))

    recovered = buffer.view(dtype=weight.dtype).view(weight.shape)

    assert socket.messages == [
        {
            "bucket_meta": {
                "strided": {
                    "name": "strided",
                    "shape": weight.shape,
                    "dtype": weight.dtype,
                    "offset": 0,
                    "handle": None,
                }
            },
            "is_last": True,
        }
    ]
    assert buffer.dtype == torch.uint8
    assert buffer.numel() == weight.nbytes
    assert torch.equal(recovered, weight)


# ---------------------------------------------------------------------------
# Process entry points (must be module-level for pickling with spawn)
# ---------------------------------------------------------------------------
def _sender_fn(zmq_handle, weight_specs, seed, bucket_size_mb, use_shm):
    """Sender process: generate weights, move to device, send."""
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender

    weights = _generate_weights(weight_specs, seed)
    sender = BucketedWeightSender(
        zmq_handle=zmq_handle,
        bucket_size_mb=bucket_size_mb,
        use_shm=use_shm,
    )
    asyncio.run(sender.async_send_weights(iter(weights)))


def _receiver_fn(zmq_handle, use_shm, result_queue):
    """Receiver process: receive weights, send back (name, dtype, shape, checksum)."""
    from verl.utils.device import get_device_name
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

    device = torch.device(f"{get_device_name()}:0")
    receiver = BucketedWeightReceiver(
        zmq_handle=zmq_handle,
        device=device,
        use_shm=use_shm,
    )
    received = []
    receiver.receive_weights(
        on_bucket_received=lambda w, is_last: received.extend([(name, t.clone()) for name, t in w])
    )
    # Only send lightweight metadata + checksum back through the queue
    summaries = [(name, t.dtype, tuple(t.shape), t.float().sum().item()) for name, t in received]
    result_queue.put(summaries)


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------
def _transfer_and_validate(weight_specs, bucket_size_mb, use_shm):
    """Spawn sender + receiver processes, then validate received tensors."""
    zmq_handle = _unique_zmq_handle()
    seed = 42
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    sender_p = ctx.Process(
        target=_sender_fn,
        args=(zmq_handle, weight_specs, seed, bucket_size_mb, use_shm),
    )
    receiver_p = ctx.Process(
        target=_receiver_fn,
        args=(zmq_handle, use_shm, result_queue),
    )

    # Start sender first (it binds), then receiver (it connects)
    sender_p.start()
    receiver_p.start()

    sender_p.join(timeout=PROCESS_TIMEOUT)
    receiver_p.join(timeout=PROCESS_TIMEOUT)

    assert sender_p.exitcode == 0, f"Sender process failed with exit code {sender_p.exitcode}"
    assert receiver_p.exitcode == 0, f"Receiver process failed with exit code {receiver_p.exitcode}"

    summaries = result_queue.get(timeout=5)

    # Regenerate expected weights on device with the same seed
    expected = _generate_weights(weight_specs, seed)

    assert len(summaries) == len(expected), f"Expected {len(expected)} weights, got {len(summaries)}"

    for (exp_name, exp_tensor), (recv_name, recv_dtype, recv_shape, recv_cksum) in zip(
        expected, summaries, strict=False
    ):
        assert exp_name == recv_name, f"Name mismatch: expected {exp_name}, got {recv_name}"
        assert tuple(exp_tensor.shape) == recv_shape, (
            f"Shape mismatch for {exp_name}: expected {tuple(exp_tensor.shape)}, got {recv_shape}"
        )
        assert exp_tensor.dtype == recv_dtype, (
            f"Dtype mismatch for {exp_name}: expected {exp_tensor.dtype}, got {recv_dtype}"
        )
        exp_sum = exp_tensor.float().sum().item()
        assert exp_sum == recv_cksum, f"Data mismatch for {exp_name}"


# ---------------------------------------------------------------------------
# Shared memory tests
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not (HAS_ACCELERATOR and not HAS_CUDA), reason="Requires (shm only tested)")
class TestBucketedWeightTransferSHM:
    """Test BucketedWeightSender/Receiver via shared memory path."""

    def test_single_small_weight(self):
        specs = [("layer.weight", (32, 16), torch.float32)]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=True)

    def test_multiple_weights_single_bucket(self):
        specs = [
            ("layer0.weight", (16, 16), torch.float32),
            ("layer0.bias", (16,), torch.float32),
            ("layer1.weight", (16, 8), torch.bfloat16),
        ]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=True)

    def test_multiple_buckets(self):
        # ~64 KB each x 20 = ~1.25 MB, bucket = 1 MB => spans 2 buckets
        specs = [(f"layer{i}.weight", (128, 128), torch.float32) for i in range(20)]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=True)

    def test_mixed_dtypes(self):
        specs = [
            ("fp32_param", (64, 64), torch.float32),
            ("bf16_param", (64, 64), torch.bfloat16),
            ("fp16_param", (32, 32), torch.float16),
        ]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=True)

    def test_empty_weights(self):
        _transfer_and_validate([], bucket_size_mb=1, use_shm=True)


# ---------------------------------------------------------------------------
# CUDA IPC tests (CUDA only — IPC is not supported on NPU)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not is_support_ipc(), reason="Requires IPC support")
class TestBucketedWeightTransferIPC:
    """Test BucketedWeightSender/Receiver via CUDA IPC path."""

    def test_single_small_weight(self):
        specs = [("layer.weight", (32, 16), torch.float32)]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=False)

    def test_multiple_weights_single_bucket(self):
        specs = [
            ("layer0.weight", (16, 16), torch.float32),
            ("layer0.bias", (16,), torch.float32),
            ("layer1.weight", (16, 8), torch.bfloat16),
        ]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=False)

    def test_multiple_buckets(self):
        specs = [(f"layer{i}.weight", (128, 128), torch.float32) for i in range(20)]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=False)

    def test_mixed_dtypes(self):
        specs = [
            ("__delta_spec__", (5,), torch.uint8),
            ("__positions__", (8,), torch.uint8),
            ("__values__", (2,), torch.bfloat16),
            ("fp32_param", (64, 64), torch.float32),
            ("bf16_param", (64, 64), torch.bfloat16),
            ("fp16_param", (32, 32), torch.float16),
        ]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=False)

    def test_empty_weights(self):
        _transfer_and_validate([], bucket_size_mb=1, use_shm=False)

    def test_exact_bucket_boundary(self):
        # 1 MB bucket = 1048576 bytes; float32 = 4 bytes => 262144 elements
        numel = (1 << 20) // 4
        specs = [("exact_fit", (numel,), torch.float32)]
        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=False)

    def test_large_weight(self):
        specs = [("embedding", (1024, 1024), torch.float32)]  # 4MB
        specs.extend([(f"layer{i}.weight", (128,), torch.bfloat16) for i in range(5)])
        specs.append(("gate_up_proj", (1024, 1024), torch.float32))  # 4MB
        specs.extend([(f"layer{i}.weight", (128,), torch.bfloat16) for i in range(20)])
        specs.append(("lm_head", (1024, 1024), torch.float32))  # 4MB

        _transfer_and_validate(specs, bucket_size_mb=1, use_shm=False)


# ---------------------------------------------------------------------------
# Buffer-reuse contract
# ---------------------------------------------------------------------------
# Half a 1 MB bucket in bfloat16, so exactly two tensors fill one bucket.
_PAIRED_NUMEL = (1 << 19) // 2
# Interleaved so every pair straddles a bucket boundary except the last, which
# serves as the in-bucket control.
_PAIRED_NAMES = ["a0", "b0", "a1", "b1", "c0", "c1"]
_PAIRS = [("a0", "a1"), ("b0", "b1"), ("c0", "c1")]


def _paired_sender_fn(zmq_handle, use_shm):
    """Sender process: each tensor is filled with its own index so it is self-identifying."""
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender

    device = torch.device(f"{get_device_name()}:0")
    weights = [
        (name, torch.full((_PAIRED_NUMEL,), float(i), dtype=torch.bfloat16, device=device))
        for i, name in enumerate(_PAIRED_NAMES)
    ]
    sender = BucketedWeightSender(zmq_handle=zmq_handle, bucket_size_mb=1, use_shm=use_shm)
    asyncio.run(sender.async_send_weights(iter(weights)))


def _paired_receiver_fn(zmq_handle, use_shm, copy_retained, result_queue):
    """Receiver process: hold each pair's first member until its partner arrives."""
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

    device = torch.device(f"{get_device_name()}:0")
    receiver = BucketedWeightReceiver(zmq_handle=zmq_handle, device=device, use_shm=use_shm)
    pending, observed = {}, {}

    def on_bucket_received(weights, is_last):
        pending.update(weights)
        for left, right in _PAIRS:
            if left in pending and right in pending:
                for name in (left, right):
                    observed[name] = pending.pop(name).min().item()
        if copy_retained:
            for name in list(pending):
                pending[name] = pending[name].clone()

    receiver.receive_weights(on_bucket_received)
    result_queue.put(observed)


def _run_paired_transfer(use_shm, copy_retained):
    zmq_handle = _unique_zmq_handle()
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    sender_p = ctx.Process(target=_paired_sender_fn, args=(zmq_handle, use_shm))
    receiver_p = ctx.Process(target=_paired_receiver_fn, args=(zmq_handle, use_shm, copy_retained, result_queue))
    sender_p.start()
    receiver_p.start()
    sender_p.join(timeout=PROCESS_TIMEOUT)
    receiver_p.join(timeout=PROCESS_TIMEOUT)
    assert sender_p.exitcode == 0, f"Sender failed with exit code {sender_p.exitcode}"
    assert receiver_p.exitcode == 0, f"Receiver failed with exit code {receiver_p.exitcode}"
    return result_queue.get(timeout=5)


@pytest.mark.skipif(not HAS_ACCELERATOR, reason="Requires an accelerator")
@pytest.mark.parametrize("use_shm", [True, False])
def test_consumer_must_copy_tensors_retained_across_buckets(use_shm):
    """A consumer that fuses several source tensors has to hold the ones that arrived first.

    On the IPC path those are views into a buffer the sender refills for the next bucket, so
    holding one without copying silently reads the next bucket's contents. The shared-memory
    path stages each tensor onto the device first, so its tensors are already private.
    """
    if not use_shm and not is_support_ipc():
        pytest.skip("Requires IPC support")

    expected = {name: float(i) for i, name in enumerate(_PAIRED_NAMES)}

    assert _run_paired_transfer(use_shm, copy_retained=True) == expected

    without_copy = _run_paired_transfer(use_shm, copy_retained=False)
    corrupted = {name for name, value in without_copy.items() if value != expected[name]}
    # a0 and b0 are the members whose partner only arrives in the next bucket.
    assert corrupted == (set() if use_shm else {"a0", "b0"}), (
        f"unexpected corruption pattern for use_shm={use_shm}: {without_copy}"
    )


def _multi_round_sender_fn(zmq_handle, use_shm, rounds):
    """Sender process that stays alive across rounds, as the real trainer does."""
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender

    device = torch.device(f"{get_device_name()}:0")
    for r in range(rounds):
        weights = [
            (name, torch.full((_PAIRED_NUMEL,), float(i + 100 * r), dtype=torch.bfloat16, device=device))
            for i, name in enumerate(_PAIRED_NAMES)
        ]
        sender = BucketedWeightSender(zmq_handle=zmq_handle, bucket_size_mb=1, use_shm=use_shm)
        asyncio.run(sender.async_send_weights(iter(weights)))


def _multi_round_receiver_fn(zmq_handle, use_shm, rounds, result_queue):
    """Receiver process that stays alive across rounds, as the real rollout server does."""
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

    device = torch.device(f"{get_device_name()}:0")
    per_round = []
    for _ in range(rounds):
        receiver = BucketedWeightReceiver(zmq_handle=zmq_handle, device=device, use_shm=use_shm)
        observed = {}
        receiver.receive_weights(lambda w, is_last, into=observed: into.update({n: t.min().item() for n, t in w}))
        per_round.append(observed)
    result_queue.put(per_round)


@pytest.mark.skipif(not HAS_ACCELERATOR, reason="Requires an accelerator")
@pytest.mark.parametrize("use_shm", [True, False])
def test_consecutive_rounds_reuse_one_socket_path(use_shm):
    """Weight sync runs once per training step, reusing the same endpoint every time.

    The sender unlinks and rebinds the socket each round while the receiver reconnects, so a
    round can only be trusted if the previous one left no state behind.
    """
    if not use_shm and not is_support_ipc():
        pytest.skip("Requires IPC support")

    rounds = 3
    zmq_handle = _unique_zmq_handle()
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    sender_p = ctx.Process(target=_multi_round_sender_fn, args=(zmq_handle, use_shm, rounds))
    receiver_p = ctx.Process(target=_multi_round_receiver_fn, args=(zmq_handle, use_shm, rounds, result_queue))
    sender_p.start()
    receiver_p.start()
    sender_p.join(timeout=PROCESS_TIMEOUT)
    receiver_p.join(timeout=PROCESS_TIMEOUT)

    assert sender_p.exitcode == 0, f"Sender failed with exit code {sender_p.exitcode}"
    assert receiver_p.exitcode == 0, f"Receiver failed with exit code {receiver_p.exitcode}"

    per_round = result_queue.get(timeout=5)
    assert per_round == [{name: float(i + 100 * r) for i, name in enumerate(_PAIRED_NAMES)} for r in range(rounds)]


def _failing_receiver_fn(zmq_handle, use_shm):
    """Receiver that completes the handshake and then raises, never acknowledging a bucket."""
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

    device = torch.device(f"{get_device_name()}:0")
    receiver = BucketedWeightReceiver(zmq_handle=zmq_handle, device=device, use_shm=use_shm)

    def explode(weights, is_last):
        raise RuntimeError("simulated receiver failure")

    receiver.receive_weights(explode)


def _timeout_sender_fn(zmq_handle, use_shm, timeout_s, result_queue):
    # Set before the module is imported, since the deadline is a module-level constant.
    os.environ["VERL_WEIGHT_TRANSFER_ACK_TIMEOUT_S"] = str(timeout_s)
    try:
        _multi_round_sender_fn(zmq_handle, use_shm, rounds=1)
    except Exception as e:
        result_queue.put((type(e).__name__, str(e)))
        return
    result_queue.put(None)


@pytest.mark.skipif(not HAS_ACCELERATOR, reason="Requires an accelerator")
def test_sender_fails_instead_of_hanging_when_receiver_dies():
    """A receiver's exception lives in an unawaited task, so the sender must not wait forever."""
    if not is_support_ipc():
        pytest.skip("Requires IPC support")

    zmq_handle = _unique_zmq_handle()
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    receiver_p = ctx.Process(target=_failing_receiver_fn, args=(zmq_handle, False))
    sender_p = ctx.Process(target=_timeout_sender_fn, args=(zmq_handle, False, 5, result_queue))
    sender_p.start()
    receiver_p.start()

    sender_p.join(timeout=PROCESS_TIMEOUT)
    receiver_p.join(timeout=PROCESS_TIMEOUT)

    assert sender_p.exitcode is not None, "Sender hung instead of timing out on the missing ack"
    raised = result_queue.get(timeout=5)
    assert raised is not None, "Sender reported success even though no ack ever arrived"
    exc_name, message = raised
    assert exc_name == "RuntimeError", f"unexpected failure from the sender: {exc_name}: {message}"
    assert "did not acknowledge" in message, message
