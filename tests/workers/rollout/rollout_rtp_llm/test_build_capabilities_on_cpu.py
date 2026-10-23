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

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from verl.workers.rollout.rtp_llm_rollout.rtp_llm_async_server import RTPLLMHttpServer


class _ReachedConfigBuild(Exception):
    pass


class TestRemoteCacheBuildPreflight(unittest.TestCase):
    def setUp(self):
        self.server = RTPLLMHttpServer.__new__(RTPLLMHttpServer)
        self.server.config = SimpleNamespace(
            enable_prefix_caching=True,
            trajectory_migration=SimpleNamespace(enabled=True, kv_transfer_backend="remote_prefix"),
        )
        self.server._build_config = Mock(side_effect=_ReachedConfigBuild)
        # These imports normally load the CUDA backend. Config construction is
        # the first operation after preflight, so reaching it ends this CPU test.
        self.modules = {
            "rtp_llm.config.server_config_setup": SimpleNamespace(
                set_parallelism_config=None,
                setup_and_configure_server=None,
                setup_cuda_device_and_accl_env=None,
            ),
            "rtp_llm.model_factory": SimpleNamespace(ModelFactory=None),
            "rtp_llm.server.backend_manager": SimpleNamespace(BackendManager=None),
            "rtp_llm.server.backend_rpc_server_visitor": SimpleNamespace(create_backend_rpc_server_visitor=None),
            "rtp_llm.utils.concurrency_controller": SimpleNamespace(
                ConcurrencyController=None, set_global_controller=None
            ),
        }

    def boot_with_ops(self, ops):
        with patch.dict(sys.modules, {**self.modules, "rtp_llm": SimpleNamespace(ops=ops), "rtp_llm.ops": ops}):
            self.server._boot_engine_impl()

    def test_build_without_kvcm_fails_before_model_configuration(self):
        ops = SimpleNamespace(get_build_capabilities=lambda: {"remote_kv_cache": False})
        with self.assertRaisesRegex(RuntimeError, "KVCM.*compiled"):
            self.boot_with_ops(ops)
        self.server._build_config.assert_not_called()

    def test_legacy_wheel_without_capability_api_fails_with_rebuild_action(self):
        with self.assertRaisesRegex(RuntimeError, "get_build_capabilities.*[Rr]ebuild"):
            self.boot_with_ops(SimpleNamespace())
        self.server._build_config.assert_not_called()

    def test_unknown_or_malformed_capability_does_not_advertise_support(self):
        for result in ({}, {"remote_kv_cache": "true"}, {"remote_kv_cache": 1}, None):
            with self.subTest(result=result), self.assertRaisesRegex(RuntimeError, "remote_kv_cache"):
                self.boot_with_ops(SimpleNamespace(get_build_capabilities=lambda result=result: result))
        self.server._build_config.assert_not_called()

    def test_native_load_failure_preserves_the_cause(self):
        failure = ImportError("kv_cache_manager_client.so: cannot open shared object file")
        ops = SimpleNamespace(get_build_capabilities=Mock(side_effect=failure))
        with self.assertRaisesRegex(RuntimeError, "KVCM") as caught:
            self.boot_with_ops(ops)
        self.assertIs(caught.exception.__cause__, failure)
        self.server._build_config.assert_not_called()

    def test_supported_build_proceeds_to_model_configuration(self):
        ops = SimpleNamespace(get_build_capabilities=lambda: {"remote_kv_cache": True, "remote_prefix_write_status": 1})
        with self.assertRaises(_ReachedConfigBuild):
            self.boot_with_ops(ops)

    def test_kvcm_build_without_prefix_status_is_rejected_before_loading_weights(self):
        for version in (None, False, True, 0, "1"):
            with self.subTest(version=version), self.assertRaisesRegex(RuntimeError, "prefix.*[Rr]ebuild"):
                self.boot_with_ops(
                    SimpleNamespace(
                        get_build_capabilities=lambda version=version: {
                            "remote_kv_cache": True,
                            "remote_prefix_write_status": version,
                        }
                    )
                )
        self.server._build_config.assert_not_called()

    def test_disabled_migration_does_not_require_the_new_api(self):
        self.server.config.trajectory_migration.enabled = False
        with self.assertRaises(_ReachedConfigBuild):
            self.boot_with_ops(SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
