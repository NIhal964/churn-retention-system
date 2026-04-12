import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.decisioning.policy import (
    load_predictions,
    load_test_data,
    build_policy_table,
    apply_policy_scores,
    simulate_policy
)


def compute_profit_heatmap():

    preds = load_predictions("data/predictions/test_predictions.csv")
    test_data = load_test_data("data/processed/test.csv")

    df = build_policy_table(preds, test_data)
    df = apply_policy_scores(df)

    # choose best policy
    policy = "score_risk_value"

    budgets = np.arange(0.05, 0.21, 0.01)     # 5% → 20%
    rescue_rates = np.arange(0.05, 0.51, 0.05) # 5% → 50%

    heatmap = []

    for r in rescue_rates:

        row = []

        for b in budgets:

            result = simulate_policy(
                df=df,
                score_col=policy,
                budget_pct=b,
                rescue_rate=r
            )

            row.append(result["expected_profit"])

        heatmap.append(row)

    heatmap = np.array(heatmap)

    return heatmap, budgets, rescue_rates


def plot_heatmap(heatmap, budgets, rescue_rates):

    plt.figure(figsize=(10,6))

    plt.imshow(
        heatmap,
        aspect="auto",
        origin="lower",
        extent=[
            budgets[0]*100,
            budgets[-1]*100,
            rescue_rates[0]*100,
            rescue_rates[-1]*100
        ]
    )

    plt.colorbar(label="Expected Profit ($)")

    plt.xlabel("Targeting Budget (%)")
    plt.ylabel("Rescue Rate (%)")

    plt.title("Profit Heatmap (Risk × Value Policy)")

    plt.show()


if __name__ == "__main__":

    heatmap, budgets, rescue_rates = compute_profit_heatmap()

    plot_heatmap(heatmap, budgets, rescue_rates)