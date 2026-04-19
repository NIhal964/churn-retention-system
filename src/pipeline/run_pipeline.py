import json
import os

import mlflow
import pandas as pd

from src.modeling.train import train_model
from src.modeling.calibrate import evaluate_calibration
from src.decisioning.policy import run_policy_comparison
from src.modeling.evaluate import compute_ranking_metrics
from src.decisioning.threshold import find_optimal_threshold


def run_pipeline(model_type="xgboost", calibration="isotonic"):

    mlflow.set_experiment("churn_retention_decision_system")

    run_name = f"{model_type}_{calibration}_full_pipeline"

    with mlflow.start_run(run_name=run_name):

        # -----------------------------
        # 1. Train model
        # -----------------------------
        print("Training model...")
        model = train_model("data/processed/train_pool.csv", model_type)

        mlflow.log_param("model_type", model_type)

        # -----------------------------
        # 2. Calibration
        # -----------------------------
        print("Calibrating model...")
        calibrated_model = evaluate_calibration(
            train_path="data/processed/train_pool.csv",
            test_path="data/processed/test.csv"
        )

        mlflow.log_param("calibration", calibration)

        # -----------------------------
        # 3. Predictions
        # -----------------------------
        print("Generating predictions...")

        test_df = pd.read_csv("data/processed/test.csv")

        X_test = test_df.drop(columns=["Churn Value"])
        y_test = test_df["Churn Value"]

        probs = calibrated_model.predict_proba(X_test)[:, 1]

        pred_df = pd.DataFrame({
            "churn_probability": probs
        })

        os.makedirs("data/predictions", exist_ok=True)
        pred_df.to_csv("data/predictions/test_predictions.csv", index=False)

        # -----------------------------
        # 4. Monitoring baseline
        # -----------------------------
        baseline = {
            "avg_churn_prob": float(probs.mean()),
            "std_churn_prob": float(probs.std()),
            "avg_monthly_charges": float(X_test["Monthly Charges"].mean()),
            "std_monthly_charges": float(X_test["Monthly Charges"].std())
        }

        os.makedirs("configs", exist_ok=True)
        with open("configs/monitoring_baseline.json", "w") as f:
            json.dump(baseline, f, indent=4)

        print("\nSaved monitoring baseline:")
        print(baseline)

        # -----------------------------
        # 5. Ranking metrics
        # -----------------------------
        print("Computing ranking metrics...")
        ranking_metrics = compute_ranking_metrics(probs, y_test)

        for k, v in ranking_metrics.items():
            mlflow.log_metric(k, v)

        # -----------------------------
        # 6. Policy Simulation (CORE)
        # -----------------------------
        print("Running policy comparison...")

        results_df = run_policy_comparison()

        # -----------------------------
        # 7. Select BEST policy
        # -----------------------------
        best = results_df.sort_values(
            "expected_profit",
            ascending=False
        ).iloc[0]

        print("\nBest Policy:")
        print(best)

        # Log params (important for API)
        mlflow.log_param("best_policy", best["policy"])
        mlflow.log_param("policy", best["policy"])
        mlflow.log_param("budget_pct", best["budget_pct"])
        mlflow.log_param("rescue_rate", best["rescue_rate"])

        # Log metrics
        mlflow.log_metric("best_expected_profit", best["expected_profit"])
        mlflow.log_metric("best_roi", best["roi"])
        mlflow.log_metric("best_customers_targeted", best["customers_targeted"])

        # -----------------------------
        # 8. Threshold (EXECUTION LAYER)
        # -----------------------------
        print("Running threshold optimization...")

        df = pred_df.copy()
        df["ValueProxy"] = test_df["Monthly Charges"] * 12
        df["Churn Value"] = y_test.values

        optimal, targeted_df = find_optimal_threshold(
            df,
            budget_pct=best["budget_pct"],
            rescue_rate=best["rescue_rate"],
            policy=best["policy"]   # 🔥 KEY FIX
        )

        print("\nOptimal Threshold:")
        print(optimal)

        # -----------------------------
        # 9. Save decision config (for API)
        # -----------------------------
        decision_config = {
            "score_threshold": float(optimal["score_threshold"]),
            "budget_pct": float(best["budget_pct"]),
            "rescue_rate": float(best["rescue_rate"]),
            "policy": best["policy"]
        }

        with open("configs/decision_config.json", "w") as f:
            json.dump(decision_config, f, indent=4)

        print("\nSaved decision_config.json:")
        print(decision_config)

        # -----------------------------
        # 10. Log artifacts
        # -----------------------------
        mlflow.log_param("score_threshold", optimal["score_threshold"])

        mlflow.log_metric("optimal_threshold_profit", optimal["expected_profit"])
        mlflow.log_metric("optimal_threshold_roi", optimal["roi"])
        mlflow.log_metric("optimal_threshold_customers", optimal["customers_targeted"])

        targeted_df.to_csv("targeted_customers.csv", index=False)
        mlflow.log_artifact("targeted_customers.csv")

        mlflow.log_artifact("policy_comparison.csv")

        print("\nPipeline completed successfully 🚀")


if __name__ == "__main__":
    run_pipeline()