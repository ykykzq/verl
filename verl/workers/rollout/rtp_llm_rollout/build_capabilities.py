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


def require_runtime_build_features(*features: str) -> None:
    """Prove opt-in native behavior before model allocation or profiling."""
    if not features:
        return
    try:
        from rtp_llm import ops

        capabilities = ops.get_build_capabilities()
    except Exception as exc:
        raise RuntimeError("Cannot verify RTP runtime features; install the matching native wheel.") from exc
    missing = [
        name
        for name in features
        if not isinstance(capabilities, Mapping) or type(capabilities.get(name)) is not int or capabilities[name] != 1
    ]
    if missing:
        raise RuntimeError(f"RTP native wheel lacks verified runtime features: {missing}; rebuild and install it.")


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
    prefix_status = capabilities.get("remote_prefix_write_status")
    if type(prefix_status) is not int or prefix_status != 1:
        raise RuntimeError(
            "remote_prefix migration requires native prefix write status API version 1; "
            f"the loaded wheel reports {prefix_status!r}. {rebuild}"
        )


def require_selected_checkpoint_build() -> None:
    """Verify the native selected-stream API before loading model weights."""
    try:
        from rtp_llm import ops

        ops.ensure_engine_ops_loaded()
        supported = callable(getattr(ops.RtpLLMOp, "request_kv_checkpoint", None))
    except Exception as exc:
        raise RuntimeError(
            "Cannot verify native request_kv_checkpoint support; rebuild and install the RTP wheel."
        ) from exc
    if not supported:
        raise RuntimeError(
            "on_demand remote-prefix migration requires native request_kv_checkpoint; "
            "rebuild and install the RTP wheel."
        )
