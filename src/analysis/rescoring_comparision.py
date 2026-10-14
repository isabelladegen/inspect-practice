from enum import StrEnum

import pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log

from src.scorers.completeness_and_additions import additions
from src.scorers.grades import Grade
from src.utils.config import experiment_dir
from src.utils.inspect_registery import registered_name


class SampleGradeColumn(StrEnum):
    """Columns of the per-sample grade table."""
    MODEL = "model"
    SAMPLE_ID = "sample_id"
    SCORER = "scorer"
    GRADE = "grade"
    APPLICABLE = "applicable"
    VALUE = "value"


def sample_grades(experiment: str) -> pd.DataFrame:
    """One row per model, sample and scorer, from the successful runs of `experiment`.

    `applicable` is False for N (nothing to grade) and for unknown grades; `value` is then NaN.
    """
    col = SampleGradeColumn
    rows = []
    runs = {}
    for info in list_eval_logs(experiment_dir(experiment)):
        log = read_eval_log(info)
        if log.status != "success":
            continue
        runs.setdefault(log.eval.model, []).append(log.location)
        for s in log.samples:
            for scorer_name, sc in (s.scores or {}).items():
                grade = Grade.parse(sc.value)
                rows.append({
                    col.MODEL: log.eval.model,
                    col.SAMPLE_ID: s.id,
                    col.SCORER: scorer_name,
                    col.GRADE: sc.value,
                    col.APPLICABLE: grade is not None and grade.applicable,
                    col.VALUE: grade.numeric if grade is not None else None,
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


def compare_models(source: str, experiment: str) -> pd.DataFrame:
    """Per model: original score; per scorer the share of applicable answers and the mean
    over them; and the additions grade distribution over applicable answers."""
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

    original = applicable_before.groupby(col.MODEL)[col.VALUE].mean().rename("original")
    applicable_share = (
        after.pivot_table(index=col.MODEL, columns=col.SCORER, values=col.APPLICABLE,
                          aggfunc="mean")
        .add_suffix("_applicable")
    )
    new = applicable_after.pivot_table(index=col.MODEL, columns=col.SCORER,
                                       values=col.VALUE, aggfunc="mean")
    additions_name = registered_name(additions)
    distribution = (
        applicable_after[applicable_after[col.SCORER] == additions_name]
        .groupby(col.MODEL)[col.GRADE].value_counts(normalize=True)
        .unstack(fill_value=0)
        .reindex(columns=[g for g in Grade if g.applicable], fill_value=0)
        .add_prefix(f"{additions_name}_")
    )
    return pd.concat([original, applicable_share, new, distribution], axis=1)