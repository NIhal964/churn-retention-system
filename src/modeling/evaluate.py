import pandas as pd
import joblib
import mlflow

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)

TARGET_COLUMN = "Churn Value"


# -----------------------------
# Ranking Metrics
# -----------------------------
def recall_at_k(y_true, probs, k=0.1):
    df = pd.DataFrame({
        "y_true": y_true,
        "probs": probs
    })

    df = df.sort_values("probs", ascending=False)

    cutoff = max(1, int(len(df) * k))
    top_k = df.head(cutoff)

    return top_k["y_true"].sum() / df["y_true"].sum()


def precision_at_k(y_true, probs, k=0.1):
    df = pd.DataFrame({
        "y_true": y_true,
        "probs": probs
    })

    df = df.sort_values("probs", ascending=False)

    cutoff = max(1, int(len(df) * k))
    top_k = df.head(cutoff)

    return top_k["y_true"].mean()


def lift_at_k(y_true, probs, k=0.1):
    df = pd.DataFrame({
        "y_true": y_true,
        "probs": probs
    })

    df = df.sort_values("probs", ascending=False)

    cutoff = max(1, int(len(df) * k))
    top_k = df.head(cutoff)

    baseline_rate = df["y_true"].mean()

    return top_k["y_true"].mean() / baseline_rate


# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate_model(
    test_path,
    model_path,
    model_type="xgboost",
    calibration="none"
):

    test = pd.read_csv(test_path)

    X_test = test.drop(columns=[TARGET_COLUMN])
    y_test = test[TARGET_COLUMN]

    model = joblib.load(model_path)

    probs = model.predict_proba(X_test)[:, 1]

    # -------------------
    # Model Metrics
    # -------------------
    roc = roc_auc_score(y_test, probs)
    pr = average_precision_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)

    # -------------------
    # Ranking Metrics
    # -------------------
    r10 = recall_at_k(y_test, probs, 0.10)
    r5 = recall_at_k(y_test, probs, 0.05)

    p5 = precision_at_k(y_test, probs, 0.05)
    p10 = precision_at_k(y_test, probs, 0.10)

    lift10 = lift_at_k(y_test, probs, 0.10)

    # -------------------
    # MLflow Logging
    # -------------------

    # Parameters
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("calibration", calibration)
    mlflow.log_param("dataset", "test")

    # Model metrics
    mlflow.log_metric("roc_auc", roc)
    mlflow.log_metric("pr_auc", pr)
    mlflow.log_metric("brier", brier)

    # Ranking metrics
    mlflow.log_metric("precision_at_5", p5)
    mlflow.log_metric("precision_at_10", p10)

    mlflow.log_metric("recall_at_5", r5)
    mlflow.log_metric("recall_at_10", r10)

    mlflow.log_metric("lift_at_10", lift10)

    # Optional: log predictions for analysis
    results_df = pd.DataFrame({
        "y_true": y_test,
        "prob": probs
    })

    results_df.to_csv("evaluation_results.csv", index=False)
    mlflow.log_artifact("evaluation_results.csv")

    # -------------------
    # Console Output (for debugging)
    # -------------------
    print(f"\nModel: {model_type} | Calibration: {calibration}")
    print("ROC-AUC:", roc)
    print("PR-AUC:", pr)
    print("Brier Score:", brier)

    print("\nRanking Metrics")
    print("Precision@5%:", p5)
    print("Precision@10%:", p10)
    print("Recall@5%:", r5)
    print("Recall@10%:", r10)
    print("Lift@10%:", lift10)

def compute_ranking_metrics(probs, y_true):
    """
    Compute ranking metrics from in-memory probabilities and true labels.
    Used inside the end-to-end pipeline.
    """

    r10 = recall_at_k(y_true, probs, 0.10)
    r5 = recall_at_k(y_true, probs, 0.05)

    p5 = precision_at_k(y_true, probs, 0.05)
    p10 = precision_at_k(y_true, probs, 0.10)

    lift10 = lift_at_k(y_true, probs, 0.10)

    return {
        "precision_at_5": p5,
        "precision_at_10": p10,
        "recall_at_5": r5,
        "recall_at_10": r10,
        "lift_at_10": lift10,
    }
# -----------------------------
# Run All Models
# -----------------------------
if __name__ == "__main__":

    # Logistic (baseline)
    evaluate_model(
        "data/processed/test.csv",
        "models/logistic_pipeline.pkl",
        model_type="logistic",
        calibration="none"
    )

    # XGBoost (raw)
    evaluate_model(
        "data/processed/test.csv",
        "models/xgboost_pipeline.pkl",
        model_type="xgboost",
        calibration="none"
    )

    # XGBoost (calibrated)
    evaluate_model(
        "data/processed/test.csv",
        "models/calibrated_xgb.pkl",
        model_type="xgboost",
        calibration="isotonic"
    )