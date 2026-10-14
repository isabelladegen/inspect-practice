from inspect_ai.log import list_eval_logs, read_eval_log, write_eval_log

from src.maintainance.recompute import Changes, compute_changes, apply_changes
from src.metrics.metric_ignoring_na import na_metrics_names
from src.scorers.completeness_and_additions import additions
from src.scorers.score_reducers import single
from src.utils.config import experiment_dir


def recompute_log_file(location: str, scorer, reducer, tag: str) -> Changes:
    """Fix `scorer`'s metrics and reductions in the log at `location`, writing only if needed.
    Returns the changes made; empty if the log was left alone."""
    log = read_eval_log(location)
    if log.status != "success":
        print(f"Skipping {location} (status {log.status})")
        return Changes(scorer=scorer.__name__)

    changes = compute_changes(log, scorer, reducer)
    if changes:
        apply_changes(log, changes, tag)
        write_eval_log(log, location)
    return changes


if __name__ == "__main__":
    experiment_name = "security_guide_V2_rescore_completeness_and_additions_V2"
    tag = "metrics_recomputed"

    changed = 0
    for info in list_eval_logs(experiment_dir(experiment_name)):
        if recompute_log_file(info.name, scorer=additions,
                              metrics= na_metrics_names(),
                              reducer=single, tag=tag):
            changed += 1
            print(f"Fixed {info.name}")
    print(f"{changed} log(s) changed in experiment {experiment_name}")
