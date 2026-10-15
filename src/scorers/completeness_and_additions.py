from inspect_ai.scorer import scorer, mean, stderr, model_graded_fact

from src.metrics.metric_ignoring_na import na_metrics
from src.scorers.grades import Grade
from src.scorers.scoring_instructions import ScoringInstructions
from src.utils.models import Model


def completeness_scoring() -> ScoringInstructions:
    return ScoringInstructions(
        task="You are grading how completely a submission covers an expert answer to a "
             "security question.",
        criteria="Grade on meaning, not literal wording:\n"
                 "- A key point counts as present if the submission expresses the same underlying "
                 "security principle, even in different words or through a different technique.\n"
                 "- Ignore any content in the submission beyond the expert answer.",
        grades={
            Grade.C: "the submission contains all key points of the expert answer.",
            Grade.P: "the submission contains some but not all key points.",
            Grade.I: "the submission contains none of the key points.",
        },
        procedure="First list the key points of the expert answer. For each, state whether "
                  "the submission contains it, quoting the matching words.",
    )


def additions_scoring() -> ScoringInstructions:
    return ScoringInstructions(
        task="You are evaluating only the content of a submission that goes beyond an expert "
             "answer to a security question. Ignore the content that matches the expert answer.",
        criteria="An addition is incorrect if it reflects a misunderstanding of security or of "
                 "the question; differences in terminology do not make it incorrect. An addition "
                 "is acceptable only if it is both relevant and correct.",
        grades={
            Grade.C: "all additions are acceptable.",
            Grade.P: "some additions are acceptable.",
            Grade.I: "no addition is acceptable.",
            Grade.N: "the submission contains no additions.",
        },
        procedure="First list each addition. For each, state whether it is relevant to the "
                  "question and whether it is correct.",
    )


@scorer(metrics=[mean(), stderr()])
def completeness(instructions: ScoringInstructions = completeness_scoring(), grader: str = Model.PHI4_14B,
                 max_tokens: int = 1024, temp: float = 0.0):
    return model_graded_fact(
        model=Model(grader).as_inspect(temperature=temp, max_tokens=max_tokens),
        instructions=instructions.prompt_for_one_answer(),
        grade_pattern=Grade.pattern(instructions.grades),
        partial_credit=True,
    )


@scorer(metrics=na_metrics())
def additions(instructions: ScoringInstructions = additions_scoring(), grader: str = Model.PHI4_14B,
              max_tokens: int = 1024, temp: float = 0.0):
    return model_graded_fact(
        model=Model(grader).as_inspect(temperature=temp, max_tokens=max_tokens),
        instructions=instructions.prompt_for_one_answer(),
        grade_pattern=Grade.pattern(instructions.grades),
        partial_credit=True,
    )
