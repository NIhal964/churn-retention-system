import pandas as pd
from src.config import load_config
import mlflow

config = load_config()


# -------------------------------
# Loaders
# -------------------------------
def load_predictions(pred_path):
    return pd.read_csv(pred_path)


def load_test_data(test_path):
    return pd.read_csv(test_path)


# -------------------------------
# Build base table
# -------------------------------
def build_policy_table(predictions, test_data):
    df = predictions.copy()

    df["MonthlyCharges"] = test_data["Monthly Charges"].values
    df["Churn Value"] = test_data["Churn Value"].values  # offline only

    horizon = config["value_proxy"]["horizon_months"]
    df["ValueProxy"] = df["MonthlyCharges"] * horizon

    return df


# -------------------------------
# Smooth behavioral weighting
# -------------------------------
def sensitivity_weight(p):
    """
    Smooth weighting:
    - mid-risk customers more persuadable
    - extremes downweighted
    """
    return max(0.0, 1 - abs(p - 0.5) * 2)


# -------------------------------
# Apply policy scores
# -------------------------------
def apply_policy_scores(df):
    df = df.copy()

    # Policy 1: Risk only
    df["score_risk_only"] = df["churn_probability"]

    # Policy 2: Risk × Value
    df["score_risk_value"] = (
        df["churn_probability"] * df["ValueProxy"]
    )

    # Policy 3: Risk × Value × Behavioral weighting
    df["weight"] = df["churn_probability"].apply(sensitivity_weight)

    df["score_risk_value_weighted"] = (
        df["churn_probability"] *
        df["ValueProxy"] *
        df["weight"]
    )

    return df


# -------------------------------
# Simulate policy
# -------------------------------
def simulate_policy(
    df,
    score_col,
    budget_pct=0.10,
    contact_cost=None,
    discount=None,
    rescue_rate=None
):
    df = df.copy()

    if contact_cost is None:
        contact_cost = config["campaign"]["contact_cost"]

    if discount is None:
        discount = config["campaign"]["discount"]

    if rescue_rate is None:
        rescue_rate = config["campaign"]["rescue_rate"]

    df = df.sort_values(score_col, ascending=False)

    cutoff = max(1, int(len(df) * budget_pct))
    targeted = df.head(cutoff).copy()

    # -------------------
    # Expected (model-based)
    # -------------------
    expected_retained_value = (
        targeted["churn_probability"] *
        targeted["ValueProxy"] *
        rescue_rate
    ).sum()

    # -------------------
    # Realized proxy (using actual churn labels)
    # -------------------
    realized_retained_value_proxy = (
        targeted["Churn Value"] *
        targeted["ValueProxy"] *
        rescue_rate
    ).sum()

    total_cost = cutoff * (contact_cost + discount)

    expected_profit = expected_retained_value - total_cost
    realized_profit_proxy = realized_retained_value_proxy - total_cost

    roi = expected_profit / total_cost if total_cost > 0 else 0

    targeted_churn_rate = targeted["Churn Value"].mean()
    avg_targeted_value = targeted["ValueProxy"].mean()

    return {
        "policy": score_col,
        "budget_pct": budget_pct,
        "customers_targeted": cutoff,
        "rescue_rate": rescue_rate,
        "targeted_churn_rate": targeted_churn_rate,
        "avg_targeted_value": avg_targeted_value,
        "expected_retained_value": expected_retained_value,
        "realized_retained_value_proxy": realized_retained_value_proxy,
        "total_cost": total_cost,
        "expected_profit": expected_profit,
        "roi": roi,
        "realized_profit_proxy": realized_profit_proxy,
    }


# -------------------------------
# Run comparison
# -------------------------------
def run_policy_comparison():

    preds = load_predictions("data/predictions/test_predictions.csv")
    test_data = load_test_data("data/processed/test.csv")

    df = build_policy_table(preds, test_data)
    df = apply_policy_scores(df)

    policies = [
        "score_risk_only",
        "score_risk_value",
        "score_risk_value_weighted",
    ]

    budgets = [0.05, 0.10, 0.15]
    rescue_rates = [0.1, 0.2, 0.3]

    results = []

    for policy in policies:
        for b in budgets:
            for r in rescue_rates:

                with mlflow.start_run(
                    run_name=f"{policy}_b{b}_r{r}",
                    nested=True
                ):

                    result = simulate_policy(
                        df=df,
                        score_col=policy,
                        budget_pct=b,
                        rescue_rate=r
                    )

                    results.append(result)

                    # -------------------
                    # LOG PARAMETERS
                    # -------------------
                    mlflow.log_param("policy_type", policy)
                    mlflow.log_param("budget_pct", b)
                    mlflow.log_param("rescue_rate", r)
                    mlflow.log_param("contact_cost", config["campaign"]["contact_cost"])
                    mlflow.log_param("discount", config["campaign"]["discount"])

                    # -------------------
                    # LOG BUSINESS METRICS
                    # -------------------
                    mlflow.log_metric("expected_profit", result["expected_profit"])
                    mlflow.log_metric("roi", result["roi"])
                    mlflow.log_metric("customers_targeted", result["customers_targeted"])

                    mlflow.log_metric("targeted_churn_rate", result["targeted_churn_rate"])
                    mlflow.log_metric("avg_targeted_value", result["avg_targeted_value"])

                    mlflow.log_metric("expected_retained_value", result["expected_retained_value"])
                    mlflow.log_metric("realized_profit_proxy", result["realized_profit_proxy"])

    results_df = pd.DataFrame(results)

    # Save artifact
    results_df.to_csv("policy_comparison.csv", index=False)
    mlflow.log_artifact("policy_comparison.csv")

    summary = (
        results_df.groupby(["policy", "budget_pct"])
        .agg({
            "expected_profit": "mean",
            "roi": "mean"
        })
        .reset_index()
    )

    print("\nPolicy Comparison Summary")
    print(summary)

    return results_df