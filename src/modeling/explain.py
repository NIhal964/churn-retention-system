import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

MODEL_PATH = "models/xgboost_pipeline.pkl"
DATA_PATH = "data/processed/train_pool.csv"
TARGET_COLUMN = "Churn Value"

REPORT_DIR = "reports"
Path(REPORT_DIR).mkdir(exist_ok=True)


def run_shap(sample_size: int = 500):

    print("Loading trained model pipeline...")
    pipeline = joblib.load(MODEL_PATH)

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET_COLUMN])

    # sample data for faster SHAP computation
    X_sample = X.sample(n=min(sample_size, len(X)), random_state=42)

    # extract pipeline components
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    # transform features the same way model sees them
    X_transformed = preprocessor.transform(X_sample)

    # get transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    # convert to DataFrame for readable SHAP plots
    X_transformed = pd.DataFrame(
        X_transformed,
        columns=feature_names
    )

    print("Computing SHAP values...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    # SHAP summary plot
    print("Generating SHAP summary plot...")
    plt.figure()
    shap.summary_plot(shap_values, X_transformed, show=False)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/shap_summary.png")

    # SHAP feature importance (bar)
    print("Generating SHAP feature importance plot...")
    plt.figure()
    shap.summary_plot(shap_values, X_transformed, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/shap_importance.png")

    print("SHAP plots saved to /reports")

    return shap_values


if __name__ == "__main__":
    run_shap()