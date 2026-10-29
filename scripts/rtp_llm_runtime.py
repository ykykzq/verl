#!/usr/bin/env python3
"""Opt-in SDK curl compatibility, applied before the training interpreter starts.

Usage: python scripts/rtp_llm_runtime.py -m <training.module> [Hydra arguments]
Only the standard library may be imported before the exec boundary.
"""

import ctypes
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

PRELOAD_OPTION = "VERL_RTP_LLM_SDK_CURL_PRELOAD"


def compatibility_environment(inherited):
    environment = dict(inherited)
    selected = environment.get(PRELOAD_OPTION)
    if not selected:
        return environment
    if sys.platform != "linux":
        raise ValueError(f"{PRELOAD_OPTION} requires Linux and ABI-compatible shared libraries")
    libraries = selected.split(":")
    for library in libraries:
        if (
            not library
            or any(character.isspace() for character in library)
            or not Path(library).is_absolute()
            or not Path(library).is_file()
            or not os.access(library, os.R_OK)
        ):
            raise ValueError(f"{PRELOAD_OPTION} requires readable absolute library paths: {library!r}")
    # Put the selected curl/SSL dependencies first, retaining unrelated preloads.
    previous = re.split(r"[:\s]+", environment.get("LD_PRELOAD", ""))
    environment["LD_PRELOAD"] = ":".join(dict.fromkeys(item for item in libraries + previous if item))
    arrow_options = [
        item.strip()
        for item in environment.get("JE_ARROW_MALLOC_CONF", "").split(",")
        if item.strip() and item.split(":", 1)[0].strip() != "background_thread"
    ]
    environment["JE_ARROW_MALLOC_CONF"] = ",".join([*arrow_options, "background_thread:false"])
    return environment


class CurlVersionInfo(ctypes.Structure):
    # Stable prefix of curl_version_info_data at CURLVERSION_FIRST.
    _fields_ = [
        ("age", ctypes.c_int),
        ("version", ctypes.c_char_p),
        ("version_num", ctypes.c_uint),
        ("host", ctypes.c_char_p),
        ("features", ctypes.c_int),
    ]


def check_runtime(module):
    try:
        version_info = ctypes.CDLL(None).curl_version_info
    except AttributeError as error:
        raise RuntimeError("LD_PRELOAD did not provide a global curl_version_info symbol") from error
    version_info.argtypes = [ctypes.c_int]
    version_info.restype = ctypes.POINTER(CurlVersionInfo)
    info = version_info(0)
    if not info or not (info.contents.features & (1 << 7)):
        raise RuntimeError("Preloaded curl must provide AsynchDNS; synchronous DNS is unsafe for this SDK")
    version = info.contents.version.decode("utf-8", errors="replace")
    # Match `python -m`: the full entry import, not a PyArrow-only import, reproduced
    # the allocator crash. The parent also checks this process's exit/teardown.
    sys.path[0] = os.getcwd()
    importlib.import_module(module)
    print(
        json.dumps(
            {
                "rtp_runtime_preflight": "passed",
                "curl_version": version,
                "asynchronous_dns": True,
                "entry_module": module,
                "LD_PRELOAD": os.environ.get("LD_PRELOAD"),
                "JE_ARROW_MALLOC_CONF": os.environ.get("JE_ARROW_MALLOC_CONF"),
            }
        ),
        flush=True,
    )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == "--check-runtime":
        check_runtime(argv[1])
        return 0
    if len(argv) < 2 or argv[0] != "-m":
        print(f"Usage: {sys.argv[0]} -m <training.module> [arguments]", file=sys.stderr)
        return 2
    try:
        environment = compatibility_environment(os.environ)
        if environment.get(PRELOAD_OPTION):
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--check-runtime", argv[1]],
                env=environment,
                check=True,
                timeout=180,
            )
        os.execve(sys.executable, [sys.executable, *argv], environment)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"RTP runtime preparation failed; training was not started: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
