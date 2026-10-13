import subprocess
import sys
from os import path, makedirs
from pathlib import Path

ROOT_DIR = path.realpath(path.join(path.dirname(__file__), '../..'))
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
LOG_DIR = path.join(ROOT_DIR, "logs")
EXP_LOG_DIR = path.join(LOG_DIR, "experiments")
RESULTS_DIR = path.join(ROOT_DIR, "results")


def experiment_dir(name: str) -> str:
    """Log folder for a named experiment."""
    return path.join(EXP_LOG_DIR, name)


def git_commit() -> str:
    """Short hash of the current commit; warns if there are uncommitted changes."""
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT_DIR, capture_output=True, text=True, check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT_DIR, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if dirty:
        print(f"Warning: uncommitted changes; results are saved under commit {commit} "
              f"but the code that produced them differs from it")
    return commit


def run_script() -> str:
    """Name of the script this process was started with."""
    main_file = getattr(sys.modules["__main__"], "__file__", None)
    if main_file is None:
        print("Warning: no script file (interactive session?); using 'interactive'")
        return "interactive"
    return Path(main_file).stem


def result_path(name: str) -> str:
    """CSV path for result `name` from the running script at the current commit."""
    folder = path.join(RESULTS_DIR, run_script(), git_commit())
    makedirs(folder, exist_ok=True)
    return path.join(folder, f"{name}.csv")