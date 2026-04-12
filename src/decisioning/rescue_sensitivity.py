import pandas as pd
import matplotlib.pyplot as plt

from src.decisioning.policy import (
    load_predictions,
    load_test_data,
    build_policy_table,
    apply_policy_scores,
    simulate_policy
)


def compute_rescue_sensitivity():

    preds = load_predictions("data/predictions/test_predictions.csv")
    test_data = load_test_data("data/processed/test.csv")

    df = build_policy_table(preds, test_data)
    df = apply_policy_scores(df)

    policies = [
        "score_risk_only",
        "score_risk_value",
        "score_risk_value_weighted"
    ]

    rescue_rates = [x/100 for x in range(5, 51, 5)]  # 5% → 50%

    budget = 0.10   # fix budget at 10%

    results = []

    for policy in policies:
        for r in rescue_rates:

            result = simulate_policy(
                df=df,
                score_col=policy,
                budget_pct=budget,
                rescue_rate=r
            )

            result["policy"] = policy
            results.append(result)

    results_df = pd.DataFrame(results)

    return results_df


def plot_rescue_curve(results_df):

    plt.figure(figsize=(8,6))

    for policy, group in results_df.groupby("policy"):

        plt.plot(
            group["rescue_rate"] * 100,
            group["expected_profit"],
            marker="o",
            label=policy
        )

    plt.xlabel("Rescue Rate (%)")
    plt.ylabel("Expected Profit ($)")
    plt.title("Profit vs Campaign Effectiveness")

    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":

    results = compute_rescue_sensitivity()

    plot_rescue_curve(results)