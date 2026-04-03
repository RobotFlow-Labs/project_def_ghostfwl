import subprocess
import sys


def test_cli_infer_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "anima_def_ghostfwl.cli.infer", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--threshold" in result.stdout
