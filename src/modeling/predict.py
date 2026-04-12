import pandas as pd
import joblib

TARGET_COLUMN = "Churn Value"


def predict(test_path, raw_data_path, model_path):
    """
    Generate churn probability predictions and attach CustomerID.
    """

    # load processed test data
    test = pd.read_csv(test_path)

    # load raw dataset to retrieve IDs
    raw = pd.read_excel(raw_data_path)

    # keep features only
    X_test = test.drop(columns=[TARGET_COLUMN])

    # load trained calibrated model
    model = joblib.load(model_path)

    # generate probabilities
    probs = model.predict_proba(X_test)[:, 1]

    # attach IDs
    customer_ids = raw.loc[test.index, "CustomerID"].values

    predictions = pd.DataFrame({
        "CustomerID": customer_ids,
        "churn_probability": probs
    })

    return predictions


if __name__ == "__main__":

    preds = predict(
        test_path="data/processed/test.csv",
        raw_data_path="data/raw/Telco_customer_churn.xlsx",
        model_path="models/calibrated_xgb.pkl"
    )

    from pathlib import Path

    output_path = Path("data/predictions")
    output_path.mkdir(parents=True, exist_ok=True)

    save_path = output_path / "test_predictions.csv"

    preds.to_csv(save_path, index=False)

    print(f"Predictions saved to: {save_path}")
    print(preds.head())