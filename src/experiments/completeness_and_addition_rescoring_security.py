from inspect_ai import eval as inspect_eval
from inspect_ai.log import list_eval_logs, read_eval_log

from src.analysis.rescoring_comparision import compare_models
from src.scorers.completeness_and_additions import SEC_COMPLETENESS_INSTRUCTIONS, completeness, additions, \
    SEC_ADDITIONS_INSTRUCTIONS
from src.tasks.replay import replay_task
from src.utils.config import experiment_dir, result_path
from src.utils.models import Model


def rescore_experiment(source: str, experiment: str, scorers: list) -> None:
    """Score the answers of experiment `source` as a new experiment `experiment`."""
    for info in list_eval_logs(experiment_dir(source)):
        log = read_eval_log(info)
        if log.status != "success":
            print(f"Skipping run {log.location} (model {log.eval.model}, status {log.status}) in experiment {source}")
            continue
        inspect_eval(replay_task(log, scorers), model=log.eval.model, log_dir=experiment_dir(experiment),
                     tags=["rescore"])


if __name__ == "__main__":
    grader = Model.PHI4_14B
    grader_max_tokens = 1024
    temp = 0.0

    source_experiment = "security_guide_V2"
    experiment_name = f"{source_experiment}_rescore_completeness_and_additions_V2"

    scorers = [
        completeness(instruction=SEC_COMPLETENESS_INSTRUCTIONS, grader=grader,
                     max_tokens=grader_max_tokens, temp=temp),
        additions(instruction=SEC_ADDITIONS_INSTRUCTIONS, grader=grader,
                  max_tokens=grader_max_tokens, temp=temp),
    ]

    rescore_experiment(source=source_experiment, experiment=experiment_name, scorers=scorers)

    result_df = compare_models(source=source_experiment, experiment=experiment_name)
    print(result_df.round(3))
    result_df.to_csv(result_path(name=experiment_name))
