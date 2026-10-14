from enum import StrEnum


class Grade(StrEnum):
    """The grading scale used by the scorers: C, P and I on a scale; N for nothing to grade."""
    C = "C"
    P = "P"
    I = "I"
    N = "N"

    @property
    def applicable(self) -> bool:
        """False for N: there was nothing to grade."""
        return self is not Grade.N

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

    @classmethod
    def parse(cls, value) -> "Grade | None":
        """The grade for a stored score value, or None if it is not a known grade."""
        try:
            return cls(value)
        except ValueError:
            return None

    @classmethod
    def pattern(cls, include_na: bool) -> str:
        """Regex matching 'GRADE: X' in a grader's reply, for the letters in use."""
        letters = "".join(g for g in cls if include_na or g.applicable)
        return rf"(?i)GRADE\s*:\s*([{letters}])"