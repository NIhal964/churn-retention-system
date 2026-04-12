import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib
from pathlib import Path
import mlflow
import mlflow.sklearn


from src.features.build import build_preprocessor

TARGET_COLUMN = "Churn Value"


# -------------------------------
# Model Factory
# -------------------------------

def build_model(model_type: str):
    """
    Return model instance based on type.
    """

    if model_type == "logistic":
        return LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            random_state=42
        )

    elif model_type == "xgboost":
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


# -------------------------------
# Training + CV Evaluation
# -------------------------------

def train_model(train_pool_path: str, model_type: str = "logistic"):
    """
    Train and evaluate model using 5-fold CV + MLflow tracking.
    """

    df = pd.read_csv(train_pool_path)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    scale_numeric = True if model_type == "logistic" else False

    preprocessor = build_preprocessor(df, scale_numeric=scale_numeric)
    model = build_model(model_type)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # -------------------
    # Cross-validation
    # -------------------
    roc_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=5,
        scoring="roc_auc"
    )

    pr_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=5,
        scoring="average_precision"
    )

    roc_mean = roc_scores.mean()
    pr_mean = pr_scores.mean()

    print(f"{model_type.upper()} CV ROC-AUC: {roc_mean:.4f}")
    print(f"{model_type.upper()} CV PR-AUC: {pr_mean:.4f}")

    # -------------------
    # Log metrics
    # -------------------
    mlflow.log_metric("cv_roc_auc", roc_mean)
    mlflow.log_metric("cv_pr_auc", pr_mean)

    # -------------------
    # Log parameters
    # -------------------
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("cv_folds", 5)

    # model-specific params
    if model_type == "logistic":
        mlflow.log_param("solver", "liblinear")
        mlflow.log_param("max_iter", 1000)

    elif model_type == "xgboost":
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_depth", 4)
        mlflow.log_param("learning_rate", 0.05)

    # -------------------
    # Fit final model
    # -------------------
    pipeline.fit(X, y)

    # -------------------
    # Log model artifact
    # -------------------
    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model"
    )

    # -------------------
    # Save locally (for API)
    # -------------------
    Path("models").mkdir(exist_ok=True)
    model_path = f"models/{model_type}_pipeline.pkl"
    joblib.dump(pipeline, model_path)

    print(f"Model saved to {model_path}")

    return pipeline


if __name__ == "__main__":
    # Train logistic regression model
    train_model("data/processed/train_pool.csv", "logistic")
    # Train XGBoost model
    train_model("data/processed/train_pool.csv", "xgboost")
