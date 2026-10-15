from inspect_ai import eval as inspect_eval

from src.analysis.rescoring_comparision import compare_models, sample_grades
from src.scorers.completeness_and_additions import completeness, additions, completeness_scoring, additions_scoring
from src.tasks.replay import replay_task
from src.utils.config import experiment_dir, result_path
from src.utils.inspect_logs import successful_logs
from src.utils.models import Model


def rescore_experiment(source: str, experiment: str, scorers: list, max_samples: int, max_connections: int) -> None:
    """Score the answers of experiment `source` as a new experiment `experiment`."""
    for log in successful_logs(source):
        inspect_eval(replay_task(log, scorers), model=log.eval.model, log_dir=experiment_dir(experiment),
                     tags=["rescore"], max_samples=max_samples, max_connections=max_connections)


if __name__ == "__main__":
    grader = Model.PHI4_14B
    grader_max_tokens = 1024
    temp = 0.0

    source_experiment = "security_guide_V2"
    experiment_name = f"{source_experiment}_rescore_completeness_and_additions_V3"

    scorers = [
        completeness(instructions=completeness_scoring(), grader=grader, max_tokens=grader_max_tokens, temp=temp),
        additions(instructions=additions_scoring(), grader=grader, max_tokens=grader_max_tokens, temp=temp),
    ]

    rescore_experiment(source=source_experiment, experiment=experiment_name, scorers=scorers, max_samples=1,
                       max_connections=1)

    result_df = compare_models(source=source_experiment, experiment=experiment_name)
    print(result_df.round(3))
    result_df.to_csv(result_path(name=experiment_name))

    # sample level results
    samples_df = sample_grades(experiment=experiment_name)
    samples_df.to_csv(result_path(name=f"{experiment_name}_samples"), index=False)
