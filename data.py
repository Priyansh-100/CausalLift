import numpy as np
import pandas as pd


def create_data(n=5000, seed=42):
    np.random.seed(seed)

    age = np.random.randint(18, 60, n)
    previous_purchases = np.random.poisson(2, n)
    time_on_site = np.random.normal(5, 2, n).clip(0, 15)
    treatment = np.random.binomial(1, 0.5, n)

    base_prob = 0.05 + 0.002 * age + 0.03 * previous_purchases + 0.01 * time_on_site

    uplift = (
        0.03
        + 0.04 * (previous_purchases > 2)
        + 0.03 * (time_on_site > 6)
    )

    conversion_prob = np.clip(base_prob + treatment * uplift, 0, 1)
    converted = np.random.binomial(1, conversion_prob)

    return pd.DataFrame({
        "age": age,
        "previous_purchases": previous_purchases,
        "time_on_site": time_on_site,
        "treatment": treatment,
        "converted": converted
    })