import pandas as pd


def segment_analysis(result):
    result = result.copy()

    result["age_group"] = pd.cut(
        result["age"],
        bins=[18, 25, 35, 45, 60],
        labels=["18-25", "26-35", "36-45", "46-60"]
    )

    segment_summary = result.groupby(
        "age_group",
        observed=True
    )["uplift_score"].mean()

    best_users = result.sort_values(
        "uplift_score",
        ascending=False
    ).head(10)

    return segment_summary, best_users


def print_segment_analysis(segment_summary, best_users):
    print("\n===== SEGMENT ANALYSIS =====")
    print(segment_summary)

    print("\n===== BEST USERS TO TARGET =====")
    print(
        best_users[
            ["age", "previous_purchases", "time_on_site", "uplift_score"]
        ]
    )