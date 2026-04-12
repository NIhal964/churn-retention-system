import pandas as pd
import matplotlib.pyplot as plt

from src.decisioning.policy import (
    load_predictions,
    load_test_data,
    build_policy_table,
    apply_policy_scores,
    simulate_policy
)


def compute_profit_curve():

    preds = load_predictions("data/predictions/test_predictions.csv")
    test_data = load_test_data("data/processed/test.csv")

    df = build_policy_table(preds, test_data)
    df = apply_policy_scores(df)

    policies = [
        "score_risk_only",
        "score_risk_value",
        "score_risk_value_weighted"
    ]

    budgets = [x/100 for x in range(1, 21)]  # 1% → 20%

    results = []

    for policy in policies:
        for b in budgets:

            result = simulate_policy(
                df=df,
                score_col=policy,
                budget_pct=b,
                rescue_rate=0.2
            )

            result["policy"] = policy
            results.append(result)

    results_df = pd.DataFrame(results)

    return results_df


def plot_profit_curve(results_df):

    plt.figure(figsize=(8,6))

    for policy, group in results_df.groupby("policy"):

        plt.plot(
            group["budget_pct"] * 100,
            group["expected_profit"],
            marker="o",
            label=policy
        )

    plt.xlabel("Targeting Budget (%)")
    plt.ylabel("Expected Profit ($)")
    plt.title("Profit vs Retention Budget")

    plt.legend()
    plt.grid(True)

    plt.show()



if __name__ == "__main__":

    results = compute_profit_curve()

    plot_profit_curve(results)
    results.to_csv("data/predictions/profit_curve_results.csv", index=False)