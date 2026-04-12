import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.features.build import build_preprocessor

TARGET_COLUMN = "Churn Value"


def build_xgb_pipeline(df):
    preprocessor = build_preprocessor(df, scale_numeric=False)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline


def evaluate_calibration(train_path, test_path):
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train = train.drop(columns=[TARGET_COLUMN])
    y_train = train[TARGET_COLUMN]

    X_test = test.drop(columns=[TARGET_COLUMN])
    y_test = test[TARGET_COLUMN]

    # -------------------
    # Base model
    # -------------------
    base_model = build_xgb_pipeline(train)
    base_model.fit(X_train, y_train)

    probs_before = base_model.predict_proba(X_test)[:, 1]

    roc_before = roc_auc_score(y_test, probs_before)
    pr_before = average_precision_score(y_test, probs_before)
    brier_before = brier_score_loss(y_test, probs_before)

    # -------------------
    # Calibrated model
    # -------------------
    calibrated = CalibratedClassifierCV(
        estimator=base_model,
        method="isotonic",
        cv=5
    )

    calibrated.fit(X_train, y_train)

    probs_after = calibrated.predict_proba(X_test)[:, 1]

    roc_after = roc_auc_score(y_test, probs_after)
    pr_after = average_precision_score(y_test, probs_after)
    brier_after = brier_score_loss(y_test, probs_after)

    print("\nBefore Calibration:")
    print(f"ROC: {roc_before:.4f}, PR: {pr_before:.4f}, Brier: {brier_before:.4f}")

    print("\nAfter Calibration:")
    print(f"ROC: {roc_after:.4f}, PR: {pr_after:.4f}, Brier: {brier_after:.4f}")

    # -------------------
    # MLflow tracking
    # -------------------
    # params
    mlflow.log_param("model", "xgboost")
    mlflow.log_param("calibration", "isotonic")
    mlflow.log_param("cv", 5)

    # BEFORE metrics
    mlflow.log_metric("roc_before", roc_before)
    mlflow.log_metric("pr_before", pr_before)
    mlflow.log_metric("brier_before", brier_before)

    # AFTER metrics
    mlflow.log_metric("roc_after", roc_after)
    mlflow.log_metric("pr_after", pr_after)
    mlflow.log_metric("brier_after", brier_after)

    # improvement
    mlflow.log_metric("brier_improvement", brier_before - brier_after)

    # log model
    mlflow.sklearn.log_model(calibrated, artifact_path="calibrated_model")

    # -------------------
    # Save locally (for API)
    # -------------------
    Path("models").mkdir(exist_ok=True)
    joblib.dump(calibrated, "models/calibrated_xgb.pkl")

    print("\nCalibrated model saved")

    # -------------------
    # Calibration curve (optional)
    # -------------------
    prob_true, prob_pred = calibration_curve(y_test, probs_after, n_bins=10)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(prob_pred, prob_true, marker="o", label="Calibrated XGBoost")
    ax.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("True probability")
    ax.set_title("Calibration Curve")
    ax.legend()

    Path("reports").mkdir(exist_ok=True)
    fig_path = Path("reports/calibration_curve.png")
    fig.savefig(fig_path)
    plt.close(fig)

    print(f"Calibration curve saved to {fig_path}")

    return calibrated

if __name__=='__main__':
    calibrated_model = evaluate_calibration(
        train_path="data/processed/train_pool.csv",
        test_path="data/processed/test.csv"
    )