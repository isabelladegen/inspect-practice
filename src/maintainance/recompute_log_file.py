from inspect_ai.log import list_eval_logs, read_eval_log, write_eval_log

from src.analysis.rescoring_comparision import compare_models, sample_grades
from src.maintainance.recompute import Changes, compute_changes, apply_changes
from src.metrics.metric_ignoring_na import na_metrics_names
from src.scorers.completeness_and_additions import additions
from src.scorers.score_reducers import single
from src.utils.config import experiment_dir, result_path


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
        if recompute_log_file(info.name, scorer=additions, reducer=single, tag=tag):
            changed += 1
            print(f"Fixed {info.name}")
    print(f"{changed} log(s) changed in experiment {experiment_name}")

    result_df = compare_models(source="security_guide_V2", experiment=experiment_name)
    print(result_df.round(3))
    result_df.to_csv(result_path(name=experiment_name))

    # sample level results
    samples_df = sample_grades(experiment=experiment_name)
    samples_df.to_csv(result_path(name=f"{experiment_name}_samples"), index=False)
