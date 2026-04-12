import joblib

MODEL_PATH = "models/calibrated_xgb.pkl"

model = None


def load_model():
    global model
    if model is None:
        model = joblib.load(MODEL_PATH)
    return model