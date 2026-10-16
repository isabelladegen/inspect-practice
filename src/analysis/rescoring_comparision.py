from enum import StrEnum

import pandas as pd
from inspect_ai.log import EvalLog

from src.scorers.grades import Grade
from src.utils.inspect_logs import logged_samples, successful_logs


class SampleGradeColumn(StrEnum):
    """Columns of the per-sample grade table."""
    GRADER = "grader"
    MODEL = "model"
    SAMPLE_ID = "sample_id"
    SCORER = "scorer"
    GRADE = "grade"
    APPLICABLE = "applicable"
    VALUE = "value"


def sample_grades(experiment: str) -> pd.DataFrame:
    """One row per grader, answering model, sample and scorer, from the successful runs of `experiment`.

    A score holding one grade per answering model (several answers graded at once) gives one row per model.
    `applicable` is False for N (nothing to grade) and for unknown grades; `value` is then NaN.
    """
    col = SampleGradeColumn
    rows = []
    runs = {}
    for log in successful_logs(experiment):
        runs.setdefault(log.eval.model, []).append(log.location)
        graders = {s.name: (s.options or {}).get("grader") for s in (log.eval.scorers or [])}
        for s in logged_samples(log):
            for scorer_name, sc in (s.scores or {}).items():
                for grader, model, value in _graded_answers(log, graders.get(scorer_name), sc.value):
                    grade = Grade.parse(value)
                    rows.append({
                        col.GRADER: grader,
                        col.MODEL: model,
                        col.SAMPLE_ID: s.id,
                        col.SCORER: scorer_name,
                        col.GRADE: value,
                        col.APPLICABLE: grade is not None and grade.applicable,
                        col.VALUE: grade.position if grade is not None else None,
                    })

    # Check 1: one successful run per model
    for model, locations in runs.items():
        if len(locations) > 1:
            print(f"Warning: {len(locations)} successful runs for {model} in experiment "
                  f"{experiment}; their samples are pooled: {locations}")

    if not rows:
        print(f"Warning: no graded samples found in experiment {experiment}")
        return pd.DataFrame(columns=list(col))

    df = pd.DataFrame(rows, columns=list(col))
    df[col.VALUE] = df[col.VALUE].astype(float)

    # Check 2: only known grades
    for r in df[df[col.GRADE].map(Grade.parse).isna()].itertuples():
        print(f"Warning: unknown grade {r.grade!r} for {r.model}, sample {r.sample_id}, "
              f"scorer {r.scorer} in experiment {experiment}; excluded from means")

    return df


def _graded_answers(log: EvalLog, scorer_grader, value) -> list[tuple]:
    """(grader, answering model, grade) for each answer a score grades."""
    if isinstance(value, dict):  # several answers at once: the eval's model graded every key's answer
        return [(log.eval.model, model, grade) for model, grade in value.items()]
    return [(scorer_grader, log.eval.model, value)]  # one answer: the eval's model answered it


def compare_models(source: str, experiment: str) -> pd.DataFrame:
    """Per grader and model: original score; per scorer the share of applicable answers, the mean
    over them, and the grade distribution over them."""
    col = SampleGradeColumn
    before = sample_grades(source)
    after = sample_grades(experiment)

    # Check 3: same sample ids on both sides
    before_ids = before.groupby(col.MODEL)[col.SAMPLE_ID].apply(set)
    after_ids = after.groupby(col.MODEL)[col.SAMPLE_ID].apply(set)
    for model in before_ids.index.union(after_ids.index):
        b = before_ids.get(model, set())
        a = after_ids.get(model, set())
        if b != a:
            print(f"Warning: sample ids differ for {model}; "
                  f"only in {source}: {sorted(b - a)}, only in {experiment}: {sorted(a - b)}")

    applicable_before = before[before[col.APPLICABLE]]
    applicable_after = after[after[col.APPLICABLE]]
    index = [col.GRADER, col.MODEL]

    original = applicable_before.groupby(col.MODEL)[col.VALUE].mean().rename("original")
    applicable_share = (
        after.pivot_table(index=index, columns=col.SCORER, values=col.APPLICABLE, aggfunc="mean")
        .add_suffix("_applicable")
    )
    new = applicable_after.pivot_table(index=index, columns=col.SCORER, values=col.VALUE, aggfunc="mean")
    distribution = (
        applicable_after
        .groupby(index + [col.SCORER])[col.GRADE].value_counts(normalize=True)
        .unstack(fill_value=0)
        .reindex(columns=[g for g in Grade if g.applicable], fill_value=0)
        .unstack(col.SCORER)
    )
    distribution.columns = [f"{scorer}_{grade}" for grade, scorer in distribution.columns]
    return pd.concat([applicable_share, new, distribution], axis=1).join(original)
