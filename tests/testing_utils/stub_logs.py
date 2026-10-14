from pathlib import Path

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset
from inspect_ai import eval as inspect_eval

from src.maintainance.recompute import Changes
from src.solvers.replay import replay
from src.tasks.replay import replay_sample, rescore_task
from src.scorers.completeness_and_additions import additions
from tests.testing_utils.stubs import stub_addition_completeness_scorers

def stub_replay_samples(n):
    return [replay_sample(id=i, input="q", target="t", answer="a") for i in range(n)]


def stub_broken_inspect_log(log_dir: Path, additions_grades, completeness_grades):
    """A log as rescoring produced it before the fix: Inspect's default reducer."""
    task = Task(name="rescore_test", dataset=MemoryDataset(stub_replay_samples(len(additions_grades))),
                solver=replay(), scorer=stub_addition_completeness_scorers(additions_grades, completeness_grades))
    return inspect_eval(task, model="mockllm/model", log_dir=str(log_dir), display="none")[0]


def stub_correct_inspect_log(log_dir: Path, additions_grades, completeness_grades):
    """A log produced through the fixed production path."""
    task = rescore_task(stub_replay_samples(len(additions_grades)),
                        stub_addition_completeness_scorers(additions_grades, completeness_grades), name="rescore_test")
    return inspect_eval(task, model="mockllm/model", log_dir=str(log_dir), display="none")[0]


def stub_no_changes():
    return Changes(scorer=additions.__name__)


def file_bytes(location: str) -> bytes:
    return Path(location.removeprefix("file://")).read_bytes()
