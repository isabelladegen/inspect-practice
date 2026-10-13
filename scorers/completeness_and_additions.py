from inspect_ai.scorer import scorer, mean, stderr, model_graded_fact

from utils.models import Model

SEC_COMPLETENESS_INSTRUCTIONS = """
You are grading answers to security questions against a target expert answer.

Grade on meaning, not literal wording:
    - The same concept in different words counts as present
      (e.g. "encode outputs" = "output encoding", "MFA" = "multi-factor authentication").
    - A different technique does not count as a key point, even if it achieves
      a similar effect. Treat it as additional content.
    - Synonyms and equivalent techniques count as the same content
      (e.g. "encode outputs" = "output encoding").
    - Extra content in the submission does not lower the grade unless it contradicts the expert answer.

Grades:
- C: the submission contains all key points of the expert answer and does not contradict the expert answer.
- P: the submission contains some but not all key points.
- I: the submission contains none of the key points, or contradicts the expert answer.

First list the key points of the expert answer. For each, state whether the
submission contains it, quoting the matching words. Then end with
'GRADE: $LETTER' where LETTER is one of C, P or I.
"""

SEC_ADDITIONS_INSTRUCTIONS = """
You are reviewing the content of a submission that goes beyond an expert answer
to a security question. Ignore the content that matches the expert answer, this has already been analysed separately.

First list each addition. For each, state whether it is:
- relevant to the question or not
- if it is relevant state whether it is correct or incorrect. A technique applied to the wrong problem is incorrect.

Grades:
- C: no additions, or all additions are correct and relevant.
- P: some additions are correct but not relevant; none are incorrect.
- I: at least one addition is incorrect.

End with 'GRADE: $LETTER' where LETTER is one of C, P or I.
"""


@scorer(metrics=[mean(), stderr()])
def completeness(instruction: str = SEC_COMPLETENESS_INSTRUCTIONS, grader: str = Model.PHI4_14B, max_tokens: int = 1024,
                 temp: float = 0.0):
    return model_graded_fact(
        model=Model(grader).as_inspect(temperature=temp, max_tokens=max_tokens),
        instructions=instruction,
        partial_credit=True,
    )


@scorer(metrics=[mean(), stderr()])
def additions(instruction: str = SEC_ADDITIONS_INSTRUCTIONS, grader: str = Model.PHI4_14B, max_tokens: int = 1024,
              temp: float = 0.0):
    return model_graded_fact(
        model=Model(grader).as_inspect(temperature=temp, max_tokens=max_tokens),
        instructions=instruction,
        partial_credit=True,
    )
