# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os
import sys
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from verl.workers.config import RolloutConfig
from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.rtp_llm_rollout.rtp_llm_async_server import RTPLLMHttpServer, RTPLLMReplica
from verl.workers.rollout.rtp_llm_rollout.rtp_llm_rollout import ServerAdapter

MODULE = "verl.workers.rollout.rtp_llm_rollout.rtp_llm_async_server"


def make_server(tp=4, rank=0, replica=0):
    config = RolloutConfig(name="rtp_llm", tensor_model_parallel_size=tp, max_model_len=1024)
    model = SimpleNamespace(
        local_path="/model", hf_config=SimpleNamespace(max_position_embeddings=1024, architectures=["Qwen3ForCausalLM"])
    )
    with (
        patch(f"{MODULE}.omega_conf_to_dataclass", side_effect=lambda obj, **kw: obj),
        patch(f"{MODULE}.ray.util.get_node_ip_address", return_value="10.0.0.1"),
        patch.dict(os.environ),
    ):
        return RTPLLMHttpServer(
            config,
            model,
            RolloutMode.STANDALONE,
            [],
            replica,
            0,
            tp,
            1,
            ",".join(str(i + 4) for i in range(tp)),
            tp_rank=rank,
        )


def remote(method):
    return SimpleNamespace(remote=method)


class TestTPConfiguration(unittest.TestCase):
    def test_tp4_constructs_four_distinct_ranks_with_shared_ports(self):
        servers = [make_server(rank=rank) for rank in range(4)]
        self.assertEqual([server.tp_rank for server in servers], [0, 1, 2, 3])
        self.assertEqual(len({server._start_port() for server in servers}), 1)
        next_base = make_server(replica=1)._start_port()
        self.assertGreater(next_base - 11, servers[0]._start_port() + 4 * 9 - 1)

    def test_boot_arguments_include_rank_world_and_pure_tp(self):
        server = make_server(rank=3)
        args = []

        def setup(argv):
            args.extend(argv)
            return SimpleNamespace(server_config=SimpleNamespace(), vit_config=SimpleNamespace())

        modules = {
            "rtp_llm.server.server_args.server_args": SimpleNamespace(setup_args=setup),
            "rtp_llm.ops": SimpleNamespace(VitSeparation=SimpleNamespace(VIT_SEPARATION_REMOTE="remote")),
        }
        with (
            patch.dict(sys.modules, modules),
            patch.object(server, "_max_concurrency", return_value=8),
            patch.object(server, "_kv_cache_mem_mib", return_value=4096),
        ):
            cfg = server._build_config()
        values = dict(zip(args[::2], args[1::2], strict=True))
        self.assertEqual(values["--tp_size"], "4")
        self.assertEqual(values["--world_size"], "4")
        self.assertEqual(values["--local_world_size"], "4")
        self.assertEqual(values["--world_rank"], "3")
        self.assertEqual(values["--ep_size"], "1")
        self.assertEqual(cfg.server_config.ip, "10.0.0.1")

    def test_rejects_unsupported_parallel_topologies_before_launch(self):
        for field in ["data_parallel_size", "pipeline_model_parallel_size", "expert_parallel_size"]:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                RolloutConfig(name="rtp_llm", tensor_model_parallel_size=4, **{field: 4})
        config = RolloutConfig(name="rtp_llm", tensor_model_parallel_size=4)
        with self.assertRaisesRegex(ValueError, "single node"):
            RTPLLMReplica(0, config, SimpleNamespace(), gpus_per_node=2)

    def test_each_checkpoint_rank_has_its_own_weight_receiver(self):
        config = RolloutConfig(name="rtp_llm", tensor_model_parallel_size=4)
        for rank in range(4):
            with (
                self.subTest(rank=rank),
                patch.dict(os.environ, {"RANK": str(rank), "RAY_LOCAL_WORLD_SIZE": "4"}),
                patch("verl.workers.rollout.rtp_llm_rollout.rtp_llm_rollout.BaseRollout.__init__"),
                patch("verl.workers.rollout.rtp_llm_rollout.rtp_llm_rollout.set_expandable_segments"),
                patch("verl.workers.rollout.rtp_llm_rollout.rtp_llm_rollout.is_support_ipc", return_value=True),
                patch("verl.workers.rollout.rtp_llm_rollout.rtp_llm_rollout.ray.get_actor") as get_actor,
            ):
                # BaseRollout normally installs this config before adapter initialization.
                adapter = ServerAdapter.__new__(ServerAdapter)
                adapter.config = config
                adapter.__init__(config, SimpleNamespace(), None, replica_rank=2)
                self.assertTrue(adapter._ensure_server_handle())
                get_actor.assert_called_once_with(f"rtp_llm_server_2_{rank}")
                self.assertEqual(adapter.is_leader_rank, rank == 0)

    def test_cache_namespace_separates_tp_layouts(self):
        self.assertNotEqual(make_server(tp=1)._cache_key_salt(7), make_server(tp=4)._cache_key_salt(7))
        self.assertEqual(make_server(rank=0)._cache_key_salt(7), make_server(rank=3)._cache_key_salt(7))


class TestTPControl(unittest.IsolatedAsyncioTestCase):
    async def test_dead_rank_wakes_requests_parked_during_weight_update(self):
        server = make_server()
        await server.abort_all_requests()
        peer = SimpleNamespace(get_tp_health=remote(AsyncMock(side_effect=RuntimeError("rank died"))))
        server.configure_tp_peers([peer])
        with (
            patch.dict(
                sys.modules, {"rtp_llm.utils.base_model_datatypes": SimpleNamespace(GenerateInput=SimpleNamespace)}
            ),
            patch.object(server, "_build_generate_config", return_value=SimpleNamespace()),
        ):
            task = asyncio.create_task(server.generate([1, 2], {}, "parked"))
            await asyncio.sleep(0)
            with self.assertRaises(RuntimeError):
                await server._check_tp_health()
            with self.assertRaisesRegex(RuntimeError, "rank died"):
                await asyncio.wait_for(task, 0.2)

    async def test_failure_during_resume_status_probe_does_not_reopen_admission(self):
        server = make_server()
        await server.abort_all_requests()

        async def probe():
            server._tp_failure = "rank died"
            return {"weight_version": None, "updating": False, "error": None}

        server.configure_tp_peers([SimpleNamespace(get_tp_weight_status=remote(probe))])
        with self.assertRaisesRegex(RuntimeError, "rank died"):
            await server.resume_generation()
        self.assertFalse(server._generation_allowed.is_set())

    async def test_migration_bounds_unresponsive_peer_rpc(self):
        server = make_server()
        server.engine = SimpleNamespace(wait_remote_cache_idle=Mock(return_value=True))

        async def never_returns(timeout_s):
            await asyncio.Event().wait()

        server.configure_tp_peers([SimpleNamespace(wait_local_remote_cache_idle=remote(never_returns))])
        task = asyncio.create_task(server.wait_remote_cache_idle(0.01))
        done, _ = await asyncio.wait([task], timeout=0.2)
        if not done:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self.fail("migration ignored timeout_s while waiting for a peer")
        with self.assertRaises(asyncio.TimeoutError):
            await task

    async def test_dead_rank_closes_replica_and_rejects_new_generation(self):
        server = make_server()
        peer = SimpleNamespace(get_tp_health=remote(AsyncMock(side_effect=RuntimeError("rank 2 died"))))
        server.configure_tp_peers([peer])
        with self.assertRaisesRegex(RuntimeError, "rank 2 died"):
            await server._check_tp_health()
        self.assertFalse(server._generation_allowed.is_set())
        self.assertTrue(server._rejecting)
        with self.assertRaisesRegex(RuntimeError, "rank 2 died"):
            await server.generate([1, 2], {}, "new-request")

    async def test_weight_apply_selects_rank_gpu_in_background_thread(self):
        server = make_server(rank=3)
        selected = []

        @contextmanager
        def device(index):
            selected.append(index)
            yield

        with patch(f"{MODULE}.get_torch_device", return_value=SimpleNamespace(device=device)):
            result = await asyncio.to_thread(server._run_on_device, lambda: "applied")
        self.assertEqual(result, "applied")
        self.assertEqual(selected, [3])

    async def test_resume_rejects_mixed_versions_or_failed_rank(self):
        for status in [
            {"weight_version": 6, "updating": False, "error": None},
            {"weight_version": 7, "updating": True, "error": None},
            {"weight_version": 7, "updating": False, "error": "failed"},
        ]:
            with self.subTest(status=status):
                server = make_server()
                server.global_steps = 7
                await server.abort_all_requests()
                server.configure_tp_peers(
                    [SimpleNamespace(get_tp_weight_status=remote(AsyncMock(return_value=status)))]
                )
                with self.assertRaises(RuntimeError):
                    await server.resume_generation()
                self.assertFalse(server._generation_allowed.is_set())

    async def test_resume_accepts_consistently_installed_weights(self):
        server = make_server()
        server.global_steps = 7
        await server.abort_all_requests()
        peer = SimpleNamespace(
            get_tp_weight_status=remote(
                AsyncMock(
                    return_value={
                        "weight_version": 7,
                        "updating": False,
                        "error": None,
                    }
                )
            )
        )
        server.configure_tp_peers([peer] * 3)
        await server.resume_generation()
        self.assertTrue(server._generation_allowed.is_set())
        self.assertEqual(peer.get_tp_weight_status.remote.await_count, 3)

    async def test_migration_waits_for_all_rank_cache_writes(self):
        server = make_server()
        server.engine = SimpleNamespace(wait_remote_cache_idle=Mock(return_value=True))
        entered, release = asyncio.Event(), asyncio.Event()

        async def flush(timeout_s):
            entered.set()
            await release.wait()
            return True

        server.configure_tp_peers([SimpleNamespace(wait_local_remote_cache_idle=remote(flush))])
        task = asyncio.create_task(server.wait_remote_cache_idle(1))
        await entered.wait()
        self.assertFalse(task.done())
        release.set()
        await task
        server.engine.wait_remote_cache_idle.assert_called_once_with(1000)

    async def test_migration_rejects_failed_rank_flush(self):
        server = make_server()
        server.engine = SimpleNamespace(wait_remote_cache_idle=Mock(return_value=True))
        server.configure_tp_peers([SimpleNamespace(wait_local_remote_cache_idle=remote(AsyncMock(return_value=False)))])
        with self.assertRaises(asyncio.TimeoutError):
            await server.wait_remote_cache_idle(1)

    async def test_nonzero_rank_cannot_serve_generation(self):
        server = make_server(rank=1)
        with self.assertRaisesRegex(RuntimeError, "rank 0"):
            await server.generate([1, 2], {}, "request")


class TestTPLaunch(unittest.IsolatedAsyncioTestCase):
    async def test_startup_timeout_covers_actor_initialization_and_configuration(self):
        replica, _, handles = self.make_replica()
        configure = replica.server_class.options

        def options(**kwargs):
            factory = configure(**kwargs)
            create = factory.remote

            def remote_create(**kwargs):
                handle = create(**kwargs)

                async def pending(peers):
                    await asyncio.Event().wait()

                handle.configure_tp_peers = remote(pending)
                return handle

            return SimpleNamespace(remote=remote_create)

        replica.server_class.options = options
        with patch.dict(os.environ, {"VERL_RTP_LLM_STARTUP_TIMEOUT_S": "0.01"}), patch(f"{MODULE}.ray.kill") as kill:
            task = asyncio.create_task(replica.launch_servers())
            done, _ = await asyncio.wait([task], timeout=0.2)
            if not done:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                self.fail("startup timeout did not cover actor initialization")
            with self.assertRaises(asyncio.TimeoutError):
                await task
        self.assertEqual(kill.call_count, len(handles))
        for handle in handles:
            kill.assert_any_call(handle, no_restart=True)
        self.assertEqual(replica.servers, [])
        self.assertIsNone(replica.server_handle)
        self.assertIsNone(replica._server_address)

    def make_replica(self, *, failure=False, node_ids=None, devices=None):
        config = RolloutConfig(name="rtp_llm", tensor_model_parallel_size=4)
        replica = RTPLLMReplica(2, config, SimpleNamespace(), gpus_per_node=8)
        replica.rollout_mode = RolloutMode.STANDALONE
        node_ids = node_ids or ["a" * 56] * 4
        devices = devices or ["4", "5", "6", "7"]
        replica.workers = [
            SimpleNamespace(__ray_call__=remote(AsyncMock(return_value=(node_id, device))))
            for node_id, device in zip(node_ids, devices, strict=True)
        ]
        created, started, handles = [], [], []
        all_started = asyncio.Event()

        def options(**options):
            def create(**kwargs):
                rank = kwargs["tp_rank"]
                created.append((options, kwargs))

                async def launch():
                    started.append(rank)
                    if len(started) == 4:
                        all_started.set()
                    await all_started.wait()
                    if failure and rank == 2:
                        raise RuntimeError("rank boot failed")

                handle = SimpleNamespace(
                    configure_tp_peers=remote(AsyncMock()),
                    launch_server=remote(launch),
                    start_tp_health_monitor=remote(AsyncMock()),
                    get_server_address=remote(AsyncMock(return_value=("10.0.0.1", 1234))),
                )
                handles.append(handle)
                return handle

            return SimpleNamespace(remote=create)

        replica.server_class = SimpleNamespace(options=options)
        return replica, created, handles

    async def test_launches_concurrently_and_exposes_only_rank_zero(self):
        replica, created, handles = self.make_replica()
        await asyncio.wait_for(replica.launch_servers(), 2)
        self.assertEqual(replica.server_handle, handles[0])
        self.assertEqual(len(handles), 4)
        for rank, (options, kwargs) in enumerate(created):
            self.assertEqual(options["name"], f"rtp_llm_server_2_{rank}")
            self.assertEqual(kwargs["cuda_visible_devices"], "4,5,6,7")
            self.assertEqual(kwargs["workers"], [replica.workers[rank]])
            handles[rank].configure_tp_peers.remote.assert_awaited_once_with(handles[:rank] + handles[rank + 1 :])

    async def test_partial_boot_failure_terminates_whole_replica(self):
        replica, _, handles = self.make_replica(failure=True)
        with patch(f"{MODULE}.ray.kill") as kill, self.assertRaisesRegex(RuntimeError, "rank boot failed"):
            await replica.launch_servers()
        self.assertEqual(kill.call_count, 4)
        for handle in handles:
            kill.assert_any_call(handle, no_restart=True)
        self.assertEqual(replica.servers, [])
        self.assertIsNone(replica.server_handle)

    async def test_rejects_split_nodes_or_shared_gpu_before_creating_engines(self):
        for kwargs in [
            {"node_ids": ["a", "a", "b", "b"]},
            {"devices": ["0", "1", "0", "1"]},
        ]:
            with self.subTest(kwargs=kwargs):
                replica, created, _ = self.make_replica(**kwargs)
                with self.assertRaisesRegex(ValueError, "distinct GPUs on a single node"):
                    await replica.launch_servers()
                self.assertEqual(created, [])


if __name__ == "__main__":
    unittest.main()
