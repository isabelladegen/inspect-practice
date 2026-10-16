from src.analysis.rescoring_comparision import compare_models, sample_grades
from src.scorers.completeness_and_additions import completeness_of_several_answers, additions_of_several_answers
from src.tasks.replay import several_answers_task
from src.utils.config import experiment_dir, result_path
from src.utils.inspect_logs import successful_logs
from src.utils.models import Model
from inspect_ai import eval as inspect_eval


def several_answers_rescoring(source: str, experiment: str, scorers: list, grader: str,
                              order_seed: int, max_samples: int, max_connections: int) -> None:
    """Grade every item's answers from experiment `source` together, by `grader`, as `experiment`."""
    task = several_answers_task(successful_logs(source), scorers, order_seed)
    inspect_eval(task, model=grader, log_dir=experiment_dir(experiment), tags=["rescore_several_answers"],
                 max_samples=max_samples, max_connections=max_connections)


def fair_grading_message() -> str:
    return ("You are grading several people's answers to the same security question. The submissions are "
            "anonymous and their order is random and changes from question to question, so position says "
            "nothing about quality. Judge every submission fairly and consistently: apply the same standard "
            "to all, and give submissions that make the same points the same grade, whatever their wording, "
            "length or position. An expert will review your grades.")


if __name__ == "__main__":
    grader_max_tokens = 2048  # four answers to reason about
    temp = 0.0
    order_seed = 112

    source_experiment = "security_guide_V2"
    experiment_name = f"{source_experiment}_rescore_several_answers_V2"

    # the model under test in this eval, judging previous models answers all at once
    graders = [Model.PHI4_14B, Model.APERTUS_V11_4B]

    for grader in graders:
        grader_model = grader.as_inspect(temperature=temp, max_tokens=grader_max_tokens,
                                         system_message=fair_grading_message())
        scorers = [completeness_of_several_answers(grader=grader_model),
                   additions_of_several_answers(grader=grader_model)]
        several_answers_rescoring(source=source_experiment, experiment=experiment_name, scorers=scorers,
                                  grader=grader, order_seed=order_seed, max_samples=1, max_connections=1)

    # analysis
    result_df = compare_models(source=source_experiment, experiment=experiment_name)
    print(result_df.round(3))
    result_df.to_csv(result_path(name=experiment_name))

    samples_df = sample_grades(experiment=experiment_name)
    samples_df.to_csv(result_path(name=f"{experiment_name}_samples"), index=False)
