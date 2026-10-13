import pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log

from src.utils.config import experiment_dir

GRADE_VALUE = {"C": 1.0, "P": 0.5, "I": 0.0}
COLUMNS = ["model", "sample_id", "scorer", "grade", "value"]


def sample_grades(experiment: str) -> pd.DataFrame:
    """One row per model, sample and scorer, from the successful runs of `experiment`."""
    rows = []
    runs = {}
    for info in list_eval_logs(experiment_dir(experiment)):
        log = read_eval_log(info)
        if log.status != "success":
            continue
        runs.setdefault(log.eval.model, []).append(log.location)
        for s in log.samples:
            for scorer_name, sc in (s.scores or {}).items():
                rows.append({"model": log.eval.model, "sample_id": s.id,
                             "scorer": scorer_name, "grade": sc.value})

    # Check 1: one successful run per model
    for model, locations in runs.items():
        if len(locations) > 1:
            print(f"Warning: {len(locations)} successful runs for {model} in experiment "
                  f"{experiment}; their samples are pooled: {locations}")

    if not rows:
        print(f"Warning: no graded samples found in experiment {experiment}")
        return pd.DataFrame(columns=COLUMNS)

    df = pd.DataFrame(rows)
    df["value"] = df["grade"].map(GRADE_VALUE)

    # Check 2: only known grades
    for r in df[df["value"].isna()].itertuples():
        print(f"Warning: unknown grade {r.grade!r} for {r.model}, sample {r.sample_id}, "
              f"scorer {r.scorer} in experiment {experiment}; excluded from means")

    return df


def compare_models(source: str, experiment: str) -> pd.DataFrame:
    """Per model: original score, new scores per axis, and the additions grade distribution."""
    before = sample_grades(source)
    after = sample_grades(experiment)

    # Check 3: same sample ids on both sides
    before_ids = before.groupby("model")["sample_id"].apply(set)
    after_ids = after.groupby("model")["sample_id"].apply(set)
    for model in before_ids.index.union(after_ids.index):
        b = before_ids.get(model, set())
        a = after_ids.get(model, set())
        if b != a:
            print(f"Warning: sample ids differ for {model}; "
                  f"only in {source}: {sorted(b - a)}, only in {experiment}: {sorted(a - b)}")

    original = before.groupby("model")["value"].mean().rename("original")
    new = after.pivot_table(index="model", columns="scorer", values="value", aggfunc="mean")
    additions = (
        after[after["scorer"] == "additions"]
        .groupby("model")["grade"].value_counts(normalize=True)
        .unstack(fill_value=0)
        .reindex(columns=["C", "P", "I"], fill_value=0)
        .add_prefix("additions_")
    )
    return pd.concat([original, new, additions], axis=1)