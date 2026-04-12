import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

TARGET_COLUMN = "Churn Value"



def get_feature_types(df: pd.DataFrame):
    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object"]).columns.tolist()

    if TARGET_COLUMN in numeric_features:
        numeric_features.remove(TARGET_COLUMN)

    return numeric_features, categorical_features


def build_preprocessor(df, scale_numeric=False):
    
    numeric_features, categorical_features = get_feature_types(df)

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    pipeline = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

   

    return pipeline