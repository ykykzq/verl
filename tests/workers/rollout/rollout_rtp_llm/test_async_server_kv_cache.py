# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from verl.workers.rollout.rtp_llm_rollout.rtp_llm_async_server import RTPLLMHttpServer


class TestRTPLLMKVCacheTransition(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _server(*, generation_allowed: bool, abort_requested: bool, inflight: dict | None = None):
        server = RTPLLMHttpServer.__new__(RTPLLMHttpServer)
        server._generation_allowed = asyncio.Event()
        if generation_allowed:
            server._generation_allowed.set()
        server._abort_requested = abort_requested
        server._request_admission_lock = asyncio.Lock()
        server._inflight = {} if inflight is None else inflight
        clear_kv_cache = Mock()
        server.engine = SimpleNamespace(clear_kv_cache=clear_kv_cache)
        return server, clear_kv_cache

    async def test_clear_requires_generation_to_be_paused(self):
        server, clear_kv_cache = self._server(generation_allowed=True, abort_requested=False)

        with self.assertRaisesRegex(RuntimeError, "paused"):
            await server.clear_kv_cache()

        clear_kv_cache.assert_not_called()

    async def test_clear_rejects_registered_requests(self):
        server, clear_kv_cache = self._server(
            generation_allowed=False,
            abort_requested=True,
            inflight={"request-1": object()},
        )

        with self.assertRaisesRegex(RuntimeError, "in flight"):
            await server.clear_kv_cache()

        clear_kv_cache.assert_not_called()

    async def test_clear_calls_binding_after_drain(self):
        server, clear_kv_cache = self._server(generation_allowed=False, abort_requested=True)

        await server.clear_kv_cache()

        clear_kv_cache.assert_called_once_with()

    async def test_release_hook_uses_the_same_transition(self):
        server, clear_kv_cache = self._server(generation_allowed=False, abort_requested=True)

        await server.release_kv_cache()

        clear_kv_cache.assert_called_once_with()
