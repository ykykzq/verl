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
"""Validate native RTP-LLM features before loading model weights."""

from collections.abc import Mapping


def require_remote_cache_build() -> None:
    """Fail closed when the loaded engine cannot prove KVCM build support."""
    rebuild = (
        "Rebuild and install the RTP-LLM native wheel with KVCM enabled "
        "(--define=use_remote_kv_cache=true after the CUDA config); "
        "see rtp-llm/docs/backend/remote_cache_build.md."
    )
    try:
        from rtp_llm import ops
    except Exception as exc:
        raise RuntimeError(f"Cannot load RTP-LLM to verify KVCM support. {rebuild}") from exc

    get_capabilities = getattr(ops, "get_build_capabilities", None)
    if not callable(get_capabilities):
        raise RuntimeError(
            "remote_prefix migration requires a verifiable KVCM build, but this RTP-LLM wheel does not expose "
            f"rtp_llm.ops.get_build_capabilities(). {rebuild}"
        )
    try:
        capabilities = get_capabilities()
    except Exception as exc:
        raise RuntimeError(f"Cannot query the loaded RTP-LLM engine's KVCM build capability. {rebuild}") from exc

    supported = capabilities.get("remote_kv_cache") if isinstance(capabilities, Mapping) else None
    if supported is False:
        raise RuntimeError(
            f"remote_prefix migration requires KVCM, but the loaded RTP-LLM engine was compiled without it. {rebuild}"
        )
    if supported is not True:
        raise RuntimeError(
            "RTP-LLM get_build_capabilities() did not return a boolean remote_kv_cache capability; "
            f"KVCM support cannot be verified. {rebuild}"
        )
