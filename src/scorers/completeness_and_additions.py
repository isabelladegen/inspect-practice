from inspect_ai.core import Target
from inspect_ai.scorer import scorer, mean, stderr, model_graded_fact, Score, Scorer
from inspect_ai.solver import TaskState
from inspect_ai.model import Model as InspectModel

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
        task="You are evaluating the content of a submission that goes beyond an expert answer to a "
             "security question. The expert answer is a reference and may be incomplete; the purpose is "
             "to check whether what the submission says beyond it is relevant and correct.",
        criteria="An addition is any point the submission makes that is not one of the expert answer's "
                 "points: another measure, attribute or fact, and also an explanation or reason given for "
                 "a point. Restating an expert point in other words is not an addition.\n"
                 "An addition is relevant only if it helps answer the question as asked; a statement about "
                 "something else is not relevant, even if it is true. Judge relevance against the question, "
                 "not against the expert answer. An addition is incorrect if it reflects a misunderstanding "
                 "of security or of the question.",
        grades={
            Grade.C: "every addition you listed is both relevant and correct.",
            Grade.P: "some, but not all, of the additions you listed are both relevant and correct.",
            Grade.I: "none of the additions you listed are both relevant and correct.",
            Grade.N: "the submission contains no additions.",
        },
        procedure="First go through the submission point by point, quoting its words, and mark each point "
                  "as either one of the expert answer's points or an addition; if every point is one of the "
                  "expert answer's points, write 'Additions: none'. For each addition, state whether it is "
                  "relevant to the question and whether it is correct. The grade must follow from this "
                  "list: N only if you marked no addition.",
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


@scorer(metrics={"*": [mean(), stderr()]})
def completeness_of_several_answers(grader: InspectModel, instructions: ScoringInstructions = completeness_scoring()):
    return _grades_per_model(instructions, _graded_line(instructions, grader))


@scorer(metrics={"*": na_metrics()})
def additions_of_several_answers(grader: InspectModel, instructions: ScoringInstructions = additions_scoring()):
    return _grades_per_model(instructions, _graded_line(instructions, grader))


def _graded_line(instructions: ScoringInstructions, grader: InspectModel) -> Scorer:
    """Inspect's fact grading of all submissions at once; the score's value is the grades line."""
    return model_graded_fact(template=several_facts_template(),
                             instructions=instructions.prompt_for_several_answers(),
                             grade_pattern=Grade.several_answers_pattern(), model=grader)


def _grades_per_model(instructions: ScoringInstructions, graded_line: Scorer) -> Scorer:
    """Splits the grades line into one grade per submission, keyed by the model that answered."""

    async def score(state: TaskState, target: Target) -> Score:
        line = await graded_line(state, target)
        if line is None:
            raise ValueError(f"Item {state.sample_id}: the grader returned no score")
        models = state.metadata["models"]  # label to model, as written by answers_by_item
        labels = list(models)
        pairs = Grade.parse_grades_line(line.value, labels, instructions.grades) if isinstance(line.value, str) else []
        problems = _grading_problems(labels, [label for label, _ in pairs])
        if problems:
            raise ValueError(f"Invalid grading of item {state.sample_id}: {'; '.join(problems)}. "
                             f"Reply: {line.explanation}")
        return Score(value={models[label]: grade for label, grade in pairs},
                     answer=line.answer, explanation=line.explanation, metadata=line.metadata)

    return score


def _grading_problems(labels: list[str], graded: list[str]) -> list[str]:
    """Why a grading is invalid: a label without a grade, or graded more than once. Empty if valid."""
    return ([f"no grade for {label}" for label in labels if label not in graded] +
            [f"{label} graded more than once" for label in sorted(set(graded)) if graded.count(label) > 1])


def several_facts_template() -> str:
    """Inspect's fact template, for several submissions."""
    return """
You are comparing several submitted answers to an expert answer on a given question. Here is the data:

[BEGIN DATA]
************
[Question]: {question}
************
[Expert]: {criterion}
************
{answer}
************
[END DATA]

Compare the factual content of each submitted answer with the expert answer. Ignore any differences in style, grammar, or punctuation.

Does each submission contain the content in the expert answer?

{instructions}
"""
