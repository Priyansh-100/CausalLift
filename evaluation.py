import numpy as np
import matplotlib.pyplot as plt


def summarize_uplift(result):
    top_20 = result.sort_values("uplift_score", ascending=False).head(
        int(len(result) * 0.2)
    )

    return {
        "average_uplift": result["uplift_score"].mean(),
        "top_20_average_uplift": top_20["uplift_score"].mean()
    }


def print_uplift_summary(summary):
    print("\n===== T-LEARNER RESULTS =====")
    print(f"Average predicted uplift: {summary['average_uplift']:.4f}")
    print(f"Top 20% average uplift:   {summary['top_20_average_uplift']:.4f}")


def plot_qini_curve(result):
    result = result.sort_values("uplift_score", ascending=False).reset_index(drop=True)

    result["treated_conversions"] = (
        (result["actual_treatment"] == 1) &
        (result["actual_conversion"] == 1)
    ).cumsum()

    result["control_conversions"] = (
        (result["actual_treatment"] == 0) &
        (result["actual_conversion"] == 1)
    ).cumsum()

    result["treated_count"] = (result["actual_treatment"] == 1).cumsum()
    result["control_count"] = (result["actual_treatment"] == 0).cumsum()

    result["uplift_curve"] = (
        result["treated_conversions"]
        - result["control_conversions"]
        * result["treated_count"]
        / result["control_count"].replace(0, np.nan)
    )

    plt.figure(figsize=(8, 5))
    plt.plot(result["uplift_curve"])
    plt.title("Qini / Uplift Curve")
    plt.xlabel("Users ranked by predicted uplift")
    plt.ylabel("Incremental conversions")
    plt.grid(True)
    plt.show()