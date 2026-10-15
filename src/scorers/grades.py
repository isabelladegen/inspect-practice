import re
from enum import StrEnum
from typing import Iterable


class Grade(StrEnum):
    """The grading scale used by the scorers: C, P and I on a scale; N for nothing to grade."""
    C = "C"
    P = "P"
    I = "I"
    N = "N"

    @property
    def numeric(self) -> float | None:
        """Position on the scale; None for N, which is not on it."""
        match self:
            case Grade.C:
                return 1.0
            case Grade.P:
                return 0.5
            case Grade.I:
                return 0.0
            case Grade.N:
                return None

    @property
    def applicable(self) -> bool:
        """False for a grade that is not on the scale, e.g. N: there was nothing to grade."""
        return self.numeric is not None

    @classmethod
    def parse(cls, value) -> "Grade | None":
        """The grade for a stored score value, or None if it is not a known grade."""
        try:
            return cls(value)
        except ValueError:
            return None

    @classmethod
    def pattern(cls, grades: Iterable["Grade"]) -> str:
        """Regex matching 'GRADE: X' in a grader's reply, for the given grades."""
        return rf"(?i)GRADE\s*:\s*([{''.join(grades)}])"

    @classmethod
    def one_answer_format(cls, grades: Iterable["Grade"]) -> str:
        """How to ask for one grade; parsed by `pattern`."""
        return f"End with 'GRADE: $LETTER' where LETTER is one of {cls._named(grades)}."

    @classmethod
    def several_answers_format(cls, labels: list[str], grades: Iterable["Grade"]) -> str:
        """How to ask for one grade per answer; parsed by `parse_several`."""
        example = ", ".join(f"{label}=$LETTER" for label in labels)
        return (f"End with one line 'GRADES: {example}' where each LETTER is one of "
                f"{cls._named(grades)}.")

    @classmethod
    def parse_several(cls, reply: str, labels: list[str],
                      grades: Iterable["Grade"]) -> list[tuple[str, "Grade"]]:
        """Every (label, grade) pair in the last 'GRADES:' line, in order, repeats included."""
        lines = re.findall(r"(?im)^\s*GRADES\s*:(.*)$", reply)
        if not lines:
            return []
        alternatives = "|".join(re.escape(label) for label in labels)
        pairs = re.findall(rf"(?i)\b({alternatives})\s*=\s*([{''.join(grades)}])\b", lines[-1])
        return [(label.upper(), cls(grade.upper())) for label, grade in pairs]

    @staticmethod
    def _named(grades: Iterable["Grade"]) -> str:
        letters = list(grades)
        return f"{', '.join(letters[:-1])} or {letters[-1]}"