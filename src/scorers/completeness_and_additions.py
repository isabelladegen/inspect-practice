from inspect_ai.scorer import scorer, mean, stderr, model_graded_fact

from src.metrics.metric_ignoring_na import mean_excl_na, stderr_excl_na, share_na
from src.utils.models import Model

SEC_COMPLETENESS_INSTRUCTIONS = """
You are grading how completely a submission covers an expert answer to a security question.

Grade on meaning, not literal wording:
- A key point counts as present if the submission expresses the same underlying
  security principle, even in different words or through a different technique.
- Ignore any content in the submission beyond the expert answer.

Grades:
- C: the submission contains all key points of the expert answer.
- P: the submission contains some but not all key points.
- I: the submission contains none of the key points.

First list the key points of the expert answer. For each, state whether the
submission contains it, quoting the matching words. Then end with
'GRADE: $LETTER' where LETTER is one of C, P or I.
"""

SEC_ADDITIONS_INSTRUCTIONS = """
You are evaluating only the content of a submission that goes beyond an expert answer
to a security question. Ignore the content that matches the expert answer.

First list each addition. For each, state whether it is relevant to the question
and whether it is correct. An addition is incorrect if it reflects a misunderstanding
of security or of the question; differences in terminology do not make it incorrect.
An addition is acceptable only if it is both relevant and correct.

Grades:
- C: all additions are acceptable.
- P: some additions are acceptable.
- I: no addition is acceptable.
- N: the submission contains no additions.

End with 'GRADE: $LETTER' where LETTER is one of C, P, I or N.
"""


@scorer(metrics=[mean(), stderr()])
def completeness(instruction: str = SEC_COMPLETENESS_INSTRUCTIONS, grader: str = Model.PHI4_14B, max_tokens: int = 1024,
                 temp: float = 0.0):
    return model_graded_fact(
        model=Model(grader).as_inspect(temperature=temp, max_tokens=max_tokens),
        instructions=instruction,
        partial_credit=True,
    )


@scorer(metrics=[mean_excl_na(), stderr_excl_na(), share_na()])
def additions(instruction: str = SEC_ADDITIONS_INSTRUCTIONS, grader: str = Model.PHI4_14B,
              max_tokens: int = 1024, temp: float = 0.0):
    return model_graded_fact(
        model=Model(grader).as_inspect(temperature=temp, max_tokens=max_tokens),
        instructions=instruction,
        grade_pattern=r"(?i)GRADE\s*:\s*([CPIN])",
        partial_credit=True,
    )
