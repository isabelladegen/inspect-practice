from os import path

import pandas as pd

from src.analysis.rescoring_comparision import compare_models, sample_grades
from src.utils.config import RESULTS_DIR

if __name__ == "__main__":
    source = "security_guide_V2"
    experiment = f"{source}_rescore_completeness_and_additions_V3"
    # where the V3 run filed its results: run script, then the commit recorded in its log
    saved = path.join(RESULTS_DIR, "completeness_and_addition_rescoring_security", "fb0c28f")

    old_summary = pd.read_csv(path.join(saved, f"{experiment}.csv"), index_col="model")
    new_summary = compare_models(source=source, experiment=experiment).droplevel("grader")
    print("columns new in the summary:", sorted(set(new_summary.columns) - set(old_summary.columns)))
    pd.testing.assert_frame_equal(new_summary[old_summary.columns], old_summary,
                                  check_names=False, check_dtype=False)
    print("summary: same values")

    old_samples = pd.read_csv(path.join(saved, f"{experiment}_samples.csv"))
    new_grades = sample_grades(experiment)
    print("graders:", new_grades["grader"].unique())
    key = ["model", "sample_id", "scorer"]
    pd.testing.assert_frame_equal(
        new_grades.drop(columns="grader").sort_values(key).reset_index(drop=True),
        old_samples.sort_values(key).reset_index(drop=True),
        check_dtype=False)
    print("samples: same values")