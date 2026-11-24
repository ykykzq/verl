"""Opt-in actor wall/current-stream timing; no tensor contents or microbatch events.

CUDA events measure elapsed time on the calling stream, including its waits and
host launch gaps. They are not kernel busy time. Nested phases must not be added
to one another or across ranks to reconstruct the driver's actor wall time.
"""

import functools
import json
import logging
import os
import socket
import time
from pathlib import Path


ENV = "VERL_ACTOR_TIMING_DIR"
_LOG = logging.getLogger(__name__)
_MISSING = object()


class ActorUpdateTiming:
    """One synchronous actor RPC, with a single final CUDA-event drain.

    Only instance methods are wrapped, and every original attribute is restored
    in finally. The module is stdlib-only; the caller supplies torch.cuda. This
    is intentionally a measurement aid, not a replacement for the driver timer.
    """

    def __init__(
        self, worker, directory, *, cuda, rank, world_size, metadata=None, minibatch_metadata=None, max_spans=256
    ):
        self.worker, self.engine, self.cuda = worker, worker.engine, cuda
        self.directory = Path(directory)
        if not self.directory.is_absolute():
            raise ValueError(f"{ENV} must be an absolute directory")
        if max_spans < 1:
            raise ValueError("actor timing max_spans must be positive")
        self.max_spans = max_spans
        self.saved, self.spans, self.errors = [], [], []
        self.mini_index = None
        self.minis, self.micro_calls, self.optimizer_calls = 0, 0, 0
        self.metadata = dict(metadata or {})
        self.minibatch_metadata = minibatch_metadata
        self.rank, self.world_size = int(rank), int(world_size)
        self.host, self.pid = socket.gethostname(), os.getpid()
        self.path = self.directory / f"actor-timing-{self.host}-{self.pid}-rank{self.rank}.jsonl"

    def _event(self):
        event = self.cuda.Event(enable_timing=True)
        event.record(self.stream)
        return event

    def _save_method(self, obj, name, replacement):
        previous = vars(obj).get(name, _MISSING)
        setattr(obj, name, replacement)
        self.saved.append((obj, name, previous))

    def _restore(self):
        for obj, name, previous in reversed(self.saved):
            if previous is _MISSING:
                delattr(obj, name)
            else:
                setattr(obj, name, previous)
        self.saved.clear()

    def _wrap(self, obj, name, stage, *, cuda=True, mini=False, extra=None):
        original = getattr(obj, name)

        @functools.wraps(original)
        def measured(*args, **kwargs):
            if len(self.spans) >= self.max_spans:
                raise RuntimeError("actor timing span capacity exceeded; no records dropped silently")
            previous_mini = self.mini_index
            if mini:
                self.mini_index = self.minis
                self.minis += 1
            row = {"stage": stage, "mini": self.mini_index}
            if extra is not None:
                row.update(extra(args, kwargs))
            if stage == "optimizer_inner_step":
                self.optimizer_calls += 1
            start_ns = time.perf_counter_ns()
            start_event = self._event() if cuda else None
            item = [row, start_event, None]
            self.spans.append(item)
            row["start_offset_s"] = (start_ns - self.start_ns) / 1e9
            row["micro_calls_before"] = self.micro_calls if stage == "forward_backward" else None
            error = None
            try:
                return original(*args, **kwargs)
            except BaseException as exc:
                error = exc
                row["error_type"] = type(exc).__name__
                raise
            finally:
                row["host_s"] = (time.perf_counter_ns() - start_ns) / 1e9
                if stage == "forward_backward":
                    row["microbatch_calls"] = self.micro_calls - row.pop("micro_calls_before")
                else:
                    row.pop("micro_calls_before")
                if cuda:
                    try:
                        if self.cuda.current_stream() != self.stream:
                            raise RuntimeError("actor timing calling stream changed inside a phase")
                        item[2] = self._event()
                    except BaseException as exc:
                        self.errors.append(f"{stage}:{type(exc).__name__}")
                        if error is None:
                            raise
                if mini:
                    self.mini_index = previous_mini

        self._save_method(obj, name, measured)

    def __enter__(self):
        if getattr(self.engine, "_actor_timing_active", False):
            raise RuntimeError("concurrent/nested actor timing on one engine is unsupported")
        if not self.cuda.is_available():
            raise RuntimeError("actor timing requires CUDA when enabled")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.start_ns, self.start_epoch_ns = time.perf_counter_ns(), time.time_ns()
        self.stream = self.cuda.current_stream()
        self.call_index = getattr(self.worker, "_actor_timing_call_index", 0) + 1
        self.worker._actor_timing_call_index = self.call_index
        try:
            self._save_method(self.engine, "_actor_timing_active", True)
            self.first_event = self._event()
            self._wrap(self.worker, "train_mini_batch", "train_mini_batch", cuda=False)
            self._wrap(
                self.worker, "train_batch", "minibatch_worker", cuda=False, mini=True, extra=self.minibatch_metadata
            )
            self._wrap(self.engine, "train_batch", "minibatch_engine")
            self._wrap(self.engine, "forward_backward_batch", "forward_backward")
            self._wrap(self.engine, "optimizer_step", "optimizer_total")
            self._wrap(self.engine, "optimizer_zero_grad", "zero_grad", cuda=False)
            self._wrap(self.engine.optimizer, "step", "optimizer_inner_step")
            self._wrap(
                self.engine,
                "to",
                "model_optimizer_transfer",
                extra=lambda args, kw: {
                    "device": str(kw.get("device", args[0] if args else "unspecified")),
                    "model": bool(kw.get("model", True)),
                    "optimizer": bool(kw.get("optimizer", True)),
                    "grad": bool(kw.get("grad", True)),
                },
            )
            # Count actual microbatch forwards, without a CUDA event or a clock
            # call in the hot path. Checkpoint recomputations occur inside the
            # model and are not additional engine.forward_step invocations.
            original_forward = self.engine.forward_step

            @functools.wraps(original_forward)
            def count_forward(*args, **kwargs):
                self.micro_calls += 1
                return original_forward(*args, **kwargs)

            self._save_method(self.engine, "forward_step", count_forward)
        except BaseException:
            self._restore()
            raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        end_ns, end_epoch_ns = time.perf_counter_ns(), time.time_ns()
        self._restore()
        drain_start = time.perf_counter_ns()
        final_event = None
        try:
            if self.cuda.current_stream() != self.stream:
                raise RuntimeError("actor timing calling stream changed at return")
            final_event = self._event()
            # No global synchronize and no per-microbatch/minibatch synchronize.
            # Wait only for this call's final current-stream event, once.
            final_event.synchronize()
            for row, begin, end in self.spans:
                row["cuda_current_stream_s"] = None
                if begin is not None:
                    if end is None or not end.query():
                        raise RuntimeError("actor timing event incomplete after final stream drain")
                    row["cuda_current_stream_s"] = begin.elapsed_time(end) / 1000.0
            call_cuda_s = self.first_event.elapsed_time(final_event) / 1000.0
        except BaseException as profile_exc:
            self.errors.append(f"finalize:{type(profile_exc).__name__}")
            call_cuda_s = None
        drain_s = (time.perf_counter_ns() - drain_start) / 1e9
        record = {
            "schema": "verl-actor-timing-v1",
            "host": self.host,
            "pid": self.pid,
            "rank": self.rank,
            "world_size": self.world_size,
            "call_index": self.call_index,
            "start_epoch_ns": self.start_epoch_ns,
            "end_epoch_ns": end_epoch_ns,
            "worker_actor_host_s": (end_ns - self.start_ns) / 1e9,
            "actor_cuda_current_stream_s": call_cuda_s,
            "profiling_drain_host_s": drain_s,
            "minibatch_calls": self.minis,
            "microbatch_forward_calls": self.micro_calls,
            "optimizer_inner_step_calls": self.optimizer_calls,
            "status": "complete" if exc is None and not self.errors else "failed",
            "training_error_type": None if exc is None else type(exc).__name__,
            "profile_errors": self.errors,
            "metadata": self.metadata,
            "stages": [item[0] for item in self.spans],
            "cuda_scope": "calling_stream_elapsed_including_waits_and_host_launch_gaps_not_kernel_busy_time",
            "forward_backward_scope": "includes_microbatch_H2D_model_loss_backward_collectives_and_postprocess",
            "optimizer_total_scope": "includes_clip_collectives_parameter_probe_CPU_copies_inner_step_and_validation",
            "transfer_scope": "engine_model_optimizer_to_only_microbatch_H2D_and_probe_D2H_are_not_separated",
            "aggregation": "nested_intervals_overlap_do_not_sum_stages_or_ranks_as_actor_wall",
        }
        write_start = time.perf_counter_ns()
        try:
            # One line per actor call. I/O is outside worker_actor_host_s but is
            # included in the unchanged driver's blocking RPC wall timer.
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
                handle.flush()
        except BaseException as profile_exc:
            self.errors.append(f"write:{type(profile_exc).__name__}")
        self.write_host_s = (time.perf_counter_ns() - write_start) / 1e9
        if self.errors:
            message = "actor timing failed: " + ", ".join(self.errors)
            if exc is None:
                raise RuntimeError(message)
            _LOG.error("%s; preserving original %s", message, type(exc).__name__)
        return False
