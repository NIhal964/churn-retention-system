import pandas as pd
import matplotlib.pyplot as plt
import mlflow


def sensitivity_weight(p):
    return max(0.0, 1 - abs(p - 0.5) * 2)


def apply_score(df, policy, prob_col, value_col):
    df = df.copy()

    if policy == "score_risk_only":
        df["score"] = df[prob_col]

    elif policy == "score_risk_value":
        df["score"] = df[prob_col] * df[value_col]

    elif policy == "score_risk_value_weighted":
        df["weight"] = df[prob_col].apply(sensitivity_weight)
        df["score"] = df[prob_col] * df[value_col] * df["weight"]

    else:
        raise ValueError(f"Unknown policy: {policy}")

    return df


def find_optimal_threshold(
    df,
    budget_pct,
    policy,  # 🔥 NEW
    prob_col="churn_probability",
    value_col="ValueProxy",
    contact_cost=5,
    discount=50,
    rescue_rate=0.1,
):
    """
    Convert selected policy into executable threshold.
    """

    df = df.copy()

    # -----------------------------
    # Apply correct scoring logic
    # -----------------------------
    df = apply_score(df, policy, prob_col, value_col)

    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    # -----------------------------
    # Apply budget
    # -----------------------------
    n = max(1, int(len(df) * budget_pct))
    targeted = df.head(n).copy()

    # -----------------------------
    # Expected value
    # -----------------------------
    expected_saved_value = (
        targeted[prob_col] * targeted[value_col] * rescue_rate
    ).sum()

    campaign_cost = n * (contact_cost + discount)
    expected_profit = expected_saved_value - campaign_cost
    roi = expected_profit / campaign_cost if campaign_cost > 0 else 0.0

    # Threshold
    score_threshold = targeted["score"].iloc[-1]

    result = {
        "budget_pct": float(budget_pct),
        "customers_targeted": n,
        "score_threshold": float(score_threshold),
        "expected_saved_value": float(expected_saved_value),
        "campaign_cost": float(campaign_cost),
        "expected_profit": float(expected_profit),
        "expected_profit_per_customer": float(expected_profit / n) if n > 0 else 0.0,
        "roi": float(roi),
        "policy": policy,  # 🔥 important
    }

    # -----------------------------
    # MLflow logging
    # -----------------------------
    mlflow.log_metric("optimal_threshold_profit", result["expected_profit"])
    mlflow.log_metric("optimal_threshold_budget_pct", result["budget_pct"])
    mlflow.log_metric("optimal_threshold_customers", result["customers_targeted"])
    mlflow.log_metric("optimal_threshold_roi", result["roi"])
    mlflow.log_metric(
        "optimal_threshold_profit_per_customer",
        result["expected_profit_per_customer"],
    )
    mlflow.log_metric("score_threshold", result["score_threshold"])

    # -----------------------------
    # Save artifacts
    # -----------------------------
    results_df = pd.DataFrame([result])
    results_df.to_csv("threshold_results.csv", index=False)
    mlflow.log_artifact("threshold_results.csv")

    targeted.to_csv("targeted_customers.csv", index=False)
    mlflow.log_artifact("targeted_customers.csv")

    return result, targeted


def plot_profit_vs_budget(results_df):
    optimal_idx = results_df["expected_profit"].idxmax()
    optimal = results_df.loc[optimal_idx]

    plt.figure(figsize=(8, 5))
    plt.plot(
        results_df["budget_pct"],
        results_df["expected_profit"],
        marker="o"
    )

    plt.scatter(
        optimal["budget_pct"],
        optimal["expected_profit"],
        color="red",
        s=120,
        label="Optimal Policy"
    )

    plt.axvline(
        optimal["budget_pct"],
        linestyle="--",
        color="red",
        alpha=0.6
    )

    plt.xlabel("Targeting Budget (%)")
    plt.ylabel("Expected Profit ($)")
    plt.title("Profit vs Targeting Budget")
    plt.legend()
    plt.grid(True)

    plt.savefig("profit_vs_budget.png")
    mlflow.log_artifact("profit_vs_budget.png")

    plt.show()