import shutil
import subprocess
import sys
from pathlib import Path

from hamcrest import *

from src.utils.config import ROOT_DIR

TY = shutil.which("ty", path=str(Path(sys.executable).parent))  # the venv's pinned ty


def test_src_type_checks():
    assert_that(TY, not_none(), "ty is not installed in this environment: uv add --dev ty")
    output = subprocess.run([TY, "check", "--python", sys.executable, "--output-format", "concise"],
                            cwd=ROOT_DIR, capture_output=True, text=True).stdout

    issues = sorted(line for line in output.splitlines() if ".py:" in line)

    assert_that(issues, empty(), f"{len(issues)} issues:\n" + "\n".join(issues))
