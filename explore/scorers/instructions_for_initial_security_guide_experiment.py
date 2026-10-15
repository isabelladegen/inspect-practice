from src.scorers.completeness_and_additions import completeness_scoring, additions_scoring

if __name__ == "__main__":
    ORIG_V2_SEC_COMPLETENESS_INSTRUCTIONS = """
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

    ORIG_V2_SEC_ADDITIONS_INSTRUCTIONS = """
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

    print("COMPLETENESS")
    print(ORIG_V2_SEC_COMPLETENESS_INSTRUCTIONS)
    print("Built via instruction class:\n")
    print(completeness_scoring().prompt_for_one_answer())
    print("\n\nADDITIONS")
    print(ORIG_V2_SEC_ADDITIONS_INSTRUCTIONS)
    print("Built via instruction class:\n")
    print(additions_scoring().prompt_for_one_answer())
