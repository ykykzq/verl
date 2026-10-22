"""Execute the launcher with a command recorder instead of starting training."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("configured,expected", [(None, "1800"), ("2400", "2400")])
def test_27b_launcher_sets_configurable_publication_wait_timeout(tmp_path, configured, expected):
    repo = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.update(LOG_DIR=str(tmp_path / "logs"), CKPTS_DIR=str(tmp_path / "checkpoints"))
    environment.pop("WEIGHT_PUBLICATION_WAIT_TIMEOUT_S", None)
    if configured is not None:
        environment["WEIGHT_PUBLICATION_WAIT_TIMEOUT_S"] = configured
    result = subprocess.run(
        [
            "bash",
            "-c",
            'python3() { printf "%s\\n" "$@"; }; export -f python3; exec bash "$1"',
            "launcher-test",
            str(repo / "examples/grpo_trainer/run_qwen3_5_27b_fsdp2.sh"),
        ],
        env=environment,
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
    )
    assert f"actor_rollout_ref.rollout.weight_publication.wait_timeout_s={expected}" in result.stdout.splitlines()
