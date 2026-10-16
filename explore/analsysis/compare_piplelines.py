from itertools import combinations

import pandas as pd

from src.analysis.rescoring_comparision import SampleGradeColumn as col, sample_grades
from src.scorers.completeness_and_additions import additions, additions_of_several_answers, completeness, \
    completeness_of_several_answers
from src.utils.inspect_registery import registered_name

if __name__ == "__main__":
    single = sample_grades("security_guide_V2_rescore_completeness_and_additions_V3")
    joint = sample_grades("security_guide_V2_rescore_several_answers_V1"    )

    axis = {registered_name(completeness): "completeness",
            registered_name(completeness_of_several_answers): "completeness",
            registered_name(additions): "additions",
            registered_name(additions_of_several_answers): "additions"}
    short = lambda model: model.split("/")[-1]
    single["pipeline"] = "single " + single[col.GRADER].map(short)
    joint["pipeline"] = "joint " + joint[col.GRADER].map(short)
    grades = pd.concat([single, joint])
    grades["axis"] = grades[col.SCORER].map(axis)

    wide = grades.pivot_table(index=[col.MODEL, col.SAMPLE_ID, "axis"], columns="pipeline",
                              values=col.GRADE, aggfunc="first")

    for axis_name in ["completeness", "additions"]:
        same_axis = wide.xs(axis_name, level="axis")
        print(f"===== {axis_name} =====")
        for a, b in combinations(wide.columns, 2):
            agree = (same_axis[a] == same_axis[b]).mean()
            print(f"----- {a} vs {b}: same grade on {agree:.0%} of {len(same_axis)}")
            print(pd.crosstab(same_axis[a], same_axis[b]))

    print("===== means per model (applicable grades only) =====")
    print(grades[grades[col.APPLICABLE]]
          .pivot_table(index=col.MODEL, columns=["axis", "pipeline"], values=col.VALUE, aggfunc="mean")
          .round(2).to_string())

    print("===== answers on which the pipelines disagree =====")
    print(wide[wide.nunique(axis=1) > 1].to_string())
