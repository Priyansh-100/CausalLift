from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


FEATURES = ["age", "previous_purchases", "time_on_site"]


def train_t_learner(df):
    X = df[FEATURES]
    y = df["converted"]
    t = df["treatment"]

    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        X, y, t, test_size=0.3, random_state=42
    )

    control_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    treatment_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    control_model.fit(X_train[t_train == 0], y_train[t_train == 0])
    treatment_model.fit(X_train[t_train == 1], y_train[t_train == 1])

    pred_control = control_model.predict_proba(X_test)[:, 1]
    pred_treatment = treatment_model.predict_proba(X_test)[:, 1]

    result = X_test.copy()
    result["actual_treatment"] = t_test.values
    result["actual_conversion"] = y_test.values
    result["pred_control"] = pred_control
    result["pred_treatment"] = pred_treatment
    result["uplift_score"] = pred_treatment - pred_control

    return result, control_model, treatment_model