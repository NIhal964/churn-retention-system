# Target column
TARGET_COLUMN = "Churn Value"

# Columns to drop completely (leakage / useless)
DROP_COLUMNS = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Score",
    "Churn Reason",
    "CLTV",
    "Churn Label"
]


def get_model_features(df):
    """
    Returns dataframe with safe model features only.
    """
    df = df.drop(columns=[col for col in DROP_COLUMNS if col in df.columns])

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"{TARGET_COLUMN} not found in dataset")

    return df