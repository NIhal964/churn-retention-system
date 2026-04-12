import pandas as pd

from src.decisioning.threshold import (
    find_optimal_threshold,
    plot_profit_vs_budget
)
from src.decisioning.policy import (
    load_predictions,
    load_test_data,
    build_policy_table
)

# load predictions produced earlier
preds = load_predictions("data/predictions/test_predictions.csv")
test_data = load_test_data("data/processed/test.csv")
df = build_policy_table(preds, test_data)

# compute optimal threshold
optimal, results = find_optimal_threshold(df)

print("\nOptimal Targeting Threshold")
print("----------------------------")
print(optimal)

# visualize profit vs threshold
plot_profit_vs_budget(results)