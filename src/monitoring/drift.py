import json
import os
import pandas as pd

LOG_FILE = "logs/predictions.jsonl"
BASELINE_FILE = "configs/monitoring_baseline.json"
MIN_DRIFT_RECORDS = 10
DRIFT_RELATIVE_THRESHOLD = 0.05


def load_logs():
    if not os.path.exists(LOG_FILE):
        raise FileNotFoundError(
            f"Log file not found: {LOG_FILE}. Run the API and generate predictions first."
        )

    data = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)


def load_baseline():
    if not os.path.exists(BASELINE_FILE):
        return None

    with open(BASELINE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def split_history(series: pd.Series):
    total = len(series)
    if total < MIN_DRIFT_RECORDS:
        return None, None

    split_index = total // 2
    baseline = series.iloc[:split_index].mean()
    current = series.iloc[split_index:].mean()
    return baseline, current


def get_recent_window(series: pd.Series):
    if len(series) < MIN_DRIFT_RECORDS:
        return None
    return series.iloc[-(len(series) // 2):]


def check_prediction_drift(df):
    if len(df) < MIN_DRIFT_RECORDS:
        print(f"Not enough records for drift detection. Need at least {MIN_DRIFT_RECORDS} log entries.")
        return

    # Sample size awareness
    if len(df) < 50:
        print("⚠ Not enough data for reliable drift detection")

    recent_probs = get_recent_window(df["churn_probability"])
    current_avg = recent_probs.mean()
    print(f"Current avg churn probability: {current_avg:.4f}")

    # High-risk customers percentage
    high_risk_pct = (df["churn_probability"] > 0.5).mean()
    print(f"% high-risk customers: {high_risk_pct:.2f}")

    baseline = load_baseline()
    if baseline is None:
        print("No saved baseline found. Cannot perform baseline-based drift detection.")
        return

    baseline_avg = baseline["avg_churn_prob"]
    threshold = baseline_avg * DRIFT_RELATIVE_THRESHOLD
    print(f"Baseline avg churn probability: {baseline_avg:.4f}")
    print(f"Drift threshold: ±{threshold:.4f} ({DRIFT_RELATIVE_THRESHOLD:.0%} of baseline)")

    if abs(current_avg - baseline_avg) > threshold:
        direction = "higher" if current_avg > baseline_avg else "lower"
        diff = abs(current_avg - baseline_avg)
        print(f"⚠ Drift detected: current avg is {direction} than baseline by {diff:.4f}")
    else:
        print("Churn probability is stable relative to the saved baseline.")


def check_feature_drift(df):
    if len(df) < MIN_DRIFT_RECORDS:
        print(f"Not enough records for feature drift detection. Need at least {MIN_DRIFT_RECORDS} log entries.")
        return

    monthly = df["input"].apply(lambda x: x["Monthly Charges"])
    recent_monthly = get_recent_window(monthly)
    current_avg = recent_monthly.mean()
    print(f"Current avg Monthly Charges: {current_avg:.2f}")

    baseline = load_baseline()
    if baseline is None:
        print("No saved baseline found. Cannot perform baseline-based feature drift detection.")
        return

    baseline_avg = baseline["avg_monthly_charges"]
    baseline_std = baseline["std_monthly_charges"]
    threshold = baseline_std
    print(f"Baseline avg Monthly Charges: {baseline_avg:.2f}")
    print(f"Baseline std Monthly Charges: {baseline_std:.2f}")

    if abs(current_avg - baseline_avg) > threshold:
        direction = "higher" if current_avg > baseline_avg else "lower"
        diff_pct = abs(current_avg - baseline_avg) / baseline_avg
        print(f"⚠ Feature drift detected in Monthly Charges: recent avg is {direction} than baseline by {diff_pct:.1%}")
    else:
        print("Monthly Charges are stable relative to the saved baseline.")


def check_tenure_drift(df):
    if len(df) < MIN_DRIFT_RECORDS:
        print(f"Not enough records for tenure drift detection. Need at least {MIN_DRIFT_RECORDS} log entries.")
        return

    tenure = df["input"].apply(lambda x: x["Tenure Months"])
    recent_tenure = get_recent_window(tenure)
    current_avg = recent_tenure.mean()
    print(f"Current avg Tenure: {current_avg:.2f}")

    baseline = load_baseline()
    if baseline is None:
        print("No saved baseline found. Cannot perform baseline-based tenure drift detection.")
        return

    if "avg_tenure" not in baseline:
        print("Tenure baseline not available. Skipping tenure drift check.")
        return

    baseline_avg = baseline["avg_tenure"]
    baseline_std = baseline["std_tenure"]
    threshold = baseline_std
    print(f"Baseline avg Tenure: {baseline_avg:.2f}")
    print(f"Baseline std Tenure: {baseline_std:.2f}")

    if abs(current_avg - baseline_avg) > threshold:
        direction = "higher" if current_avg > baseline_avg else "lower"
        diff_pct = abs(current_avg - baseline_avg) / baseline_avg
        print(f"⚠ Feature drift detected in Tenure: recent avg is {direction} than baseline by {diff_pct:.1%}")
    else:
        print("Tenure is stable relative to the saved baseline.")


if __name__ == "__main__":
    try:
        df = load_logs()

        if df.empty:
            print("No prediction logs found in the file.")
        else:
            print("=== Monitoring Report ===")
            check_prediction_drift(df)
            check_feature_drift(df)
            check_tenure_drift(df)
    except FileNotFoundError as e:
        print(str(e))
    except Exception as e:
        print(f"Unexpected error while running drift monitoring: {e}")