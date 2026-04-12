from src.data.load import load_raw_data
from src.data.validate import clean_data
from src.features.schema import get_model_features

from pathlib import Path
import pandas as pd

PROCESSED_PATH = Path("data/processed/cleaned_data.csv")


def main():
    df = load_raw_data()
    df = get_model_features(df)
    df = clean_data(df)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print("Cleaned dataset saved to:", PROCESSED_PATH)


if __name__ == "__main__":
    main()