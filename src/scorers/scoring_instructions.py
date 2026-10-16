from dataclasses import dataclass

from src.scorers.grades import Grade


@dataclass(frozen=True)
class ScoringInstructions:
    """What is scored and how; assembles the prompt for one submission or several."""
    task: str
    criteria: str
    grades: dict[Grade, str]
    procedure: str

    def prompt_for_one_answer(self) -> str:
        return "\n\n".join([
            self.task,
            self.criteria,
            self._grades(),
            f"{self.procedure} {Grade.one_answer_format(self.grades)}",
        ])

    def prompt_for_several_answers(self) -> str:
        return "\n\n".join([
            self.task,
            self.criteria,
            self._grades(),
            "Judge each submission against the expert answer and the question only, never against the "
            "other submissions.",
            f"For each submission in turn: {self.procedure}",
            "Then review your grades together. Submissions that make the same points must get "
            "the same grade, whatever their wording, length or position. Revise any grade that "
            "fails this.",
            Grade.several_answers_format(self.grades),
        ])

    def _grades(self) -> str:
        return "Grades:\n" + "\n".join(f"- {g}: {meaning}" for g, meaning in self.grades.items())
