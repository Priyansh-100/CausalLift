import pandas as pd
from scipy.stats import chi2_contingency


def run_ab_test(df):
    control = df[df["treatment"] == 0]
    treatment = df[df["treatment"] == 1]

    control_rate = control["converted"].mean()
    treatment_rate = treatment["converted"].mean()
    uplift = treatment_rate - control_rate

    table = pd.crosstab(df["treatment"], df["converted"])
    _, p_value, _, _ = chi2_contingency(table)

    return {
        "control_rate": control_rate,
        "treatment_rate": treatment_rate,
        "absolute_uplift": uplift,
        "p_value": p_value,
        "significant": p_value < 0.05
    }


def print_ab_results(results):
    print("\n===== BASIC A/B TEST =====")
    print(f"Control conversion rate:   {results['control_rate']:.4f}")
    print(f"Treatment conversion rate: {results['treatment_rate']:.4f}")
    print(f"Absolute uplift:           {results['absolute_uplift']:.4f}")
    print(f"P-value:                   {results['p_value']:.4f}")
    print("Result:", "Significant" if results["significant"] else "Not significant")