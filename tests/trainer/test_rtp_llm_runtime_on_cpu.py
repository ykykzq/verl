"""Check the exec boundary that must precede Arrow, Torch and SDK imports."""

import ctypes
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/rtp_llm_runtime.py"
spec = importlib.util.spec_from_file_location("rtp_runtime_launcher", SCRIPT)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
MODULE = "verl.experimental.fully_async_policy.fully_async_main"


@pytest.fixture
def preload_environment(tmp_path, monkeypatch):
    libraries = [tmp_path / name for name in ("libcrypto.so", "libssl.so", "libcurl.so")]
    for library in libraries:
        library.touch()
    monkeypatch.setattr(runtime.sys, "platform", "linux")
    return {"VERL_RTP_LLM_SDK_CURL_PRELOAD": ":".join(map(str, libraries))}


def test_opt_in_preserves_other_preloads_and_allocator_settings(preload_environment):
    selected = preload_environment["VERL_RTP_LLM_SDK_CURL_PRELOAD"]
    inherited = {
        **preload_environment,
        "LD_PRELOAD": selected + " /opt/observer.so:/opt/observer.so",
        "JE_ARROW_MALLOC_CONF": "dirty_decay_ms:1000,background_thread:true,abort_conf:true",
    }
    result = runtime.compatibility_environment(inherited)
    assert result["LD_PRELOAD"] == selected + ":/opt/observer.so"
    assert result["JE_ARROW_MALLOC_CONF"] == "dirty_decay_ms:1000,abort_conf:true,background_thread:false"
    assert inherited["JE_ARROW_MALLOC_CONF"].endswith("abort_conf:true")


@pytest.mark.parametrize("selected", ["libcurl.so", "/missing/libcurl.so", ":/libcurl.so"])
def test_invalid_explicit_libraries_fail_before_python_exec(selected, monkeypatch):
    monkeypatch.setattr(runtime.sys, "platform", "linux")
    with pytest.raises(ValueError):
        runtime.compatibility_environment({"VERL_RTP_LLM_SDK_CURL_PRELOAD": selected})


def test_opt_in_rejects_non_linux(preload_environment, monkeypatch):
    monkeypatch.setattr(runtime.sys, "platform", "darwin")
    with pytest.raises(ValueError, match="Linux"):
        runtime.compatibility_environment(preload_environment)


def test_unconfigured_launcher_executes_without_changing_environment(monkeypatch):
    monkeypatch.delenv("VERL_RTP_LLM_SDK_CURL_PRELOAD", raising=False)
    original = dict(os.environ)
    preflight, execute = Mock(), Mock()
    monkeypatch.setattr(runtime.subprocess, "run", preflight)
    monkeypatch.setattr(runtime.os, "execve", execute)
    assert runtime.main(["-m", MODULE, "trainer.total_training_steps=10"]) == 0
    preflight.assert_not_called()
    execute.assert_called_once_with(
        sys.executable, [sys.executable, "-m", MODULE, "trainer.total_training_steps=10"], original
    )


@pytest.mark.parametrize("failure", [None, "crash", "timeout"])
def test_preflight_and_training_share_environment_and_failure_stops_launch(preload_environment, monkeypatch, failure):
    for key, value in preload_environment.items():
        monkeypatch.setenv(key, value)
    expected_env = runtime.compatibility_environment(os.environ)
    error = None
    if failure == "crash":
        error = subprocess.CalledProcessError(-11, "full-entry-import")
    elif failure == "timeout":
        error = subprocess.TimeoutExpired("full-entry-import", 180)
    preflight, execute = Mock(side_effect=error), Mock()
    monkeypatch.setattr(runtime.subprocess, "run", preflight)
    monkeypatch.setattr(runtime.os, "execve", execute)
    assert runtime.main(["-m", MODULE]) == (1 if failure else 0)
    call = preflight.call_args
    assert call.args[0] == [sys.executable, str(SCRIPT), "--check-runtime", MODULE]
    assert call.kwargs["env"] == expected_env
    assert call.kwargs["check"] is True
    assert call.kwargs["timeout"] > 0
    if failure:
        execute.assert_not_called()
    else:
        assert execute.call_args.args[2] == expected_env


@pytest.mark.parametrize("asynchronous", [False, True])
def test_runtime_probe_requires_global_asynchronous_curl_and_imports_full_entry(monkeypatch, asynchronous):
    info = runtime.CurlVersionInfo()
    info.version = b"test-curl"
    info.features = (1 << 7) if asynchronous else 0
    curl = Mock()
    curl.curl_version_info.return_value = ctypes.pointer(info)
    monkeypatch.setattr(runtime.ctypes, "CDLL", Mock(return_value=curl))
    imported = Mock()
    monkeypatch.setattr(runtime.importlib, "import_module", imported)
    if asynchronous:
        runtime.check_runtime(MODULE)
        imported.assert_called_once_with(MODULE)
    else:
        with pytest.raises(RuntimeError, match="AsynchDNS"):
            runtime.check_runtime(MODULE)
        imported.assert_not_called()


def test_disabled_wrapper_really_executes_a_fresh_module(tmp_path):
    (tmp_path / "entry_probe.py").write_text("import sys\nprint('ENTRY', sys.argv[1:])\n")
    environment = dict(os.environ, PYTHONPATH=str(tmp_path))
    environment.pop("VERL_RTP_LLM_SDK_CURL_PRELOAD", None)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "-m", "entry_probe", "--cfg", "job"],
        env=environment,
        text=True,
        capture_output=True,
        timeout=15,
        check=True,
    )
    assert "ENTRY ['--cfg', 'job']" in result.stdout
