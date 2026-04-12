from fastapi import FastAPI,HTTPException
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.api.schemas import CustomerFeatures, DecisionResponse
from src.api.model import load_model
from src.config import load_config
# From your pipeline output
import json

with open("configs/decision_config.json") as f:
    decision_config = json.load(f)

SCORE_THRESHOLD = decision_config["score_threshold"]
RESCUE_RATE = decision_config["rescue_rate"]

BEST_POLICY = decision_config["policy"]
budget_pct = decision_config["budget_pct"]
config = load_config()


CONTACT_COST = config["campaign"]["contact_cost"]
DISCOUNT = config["campaign"]["discount"]

app = FastAPI(
    title="Churn Decision API",
    description="Predict churn probability and expected loss",
    version="1.0"
)

config = load_config()
model = load_model()


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/decision", response_model=DecisionResponse)
def decision(customer: CustomerFeatures):
    logger.info("Decision request received")

    try:
        data = pd.DataFrame([customer.model_dump(by_alias=True)])

        # Model prediction
        prob = model.predict_proba(data)[:, 1][0]

        # Value proxy
        horizon = config["value_proxy"]["horizon_months"]
        value_proxy = data["Monthly Charges"].iloc[0] * horizon

        # Score
        score = prob * value_proxy

        # Expected saved value
        expected_saved = prob * value_proxy * RESCUE_RATE
        cost = CONTACT_COST + DISCOUNT
        expected_profit = expected_saved - cost

        # Decision logic (TOP-K proxy via threshold)
        if score >= SCORE_THRESHOLD:
            decision = "TARGET"
        else:
            decision = "DO_NOT_TARGET"

        # Reasoning
        if expected_profit <= 0:
            reason = "Not profitable to target"
        elif score >= SCORE_THRESHOLD:
            reason = "High value and above threshold"
        else:
            reason = "Below targeting threshold"

        return DecisionResponse(
            churn_probability=round(float(prob), 4),
            value_proxy=round(value_proxy, 2),
            score=round(score, 2),
            decision=decision,
            score_threshold=round(SCORE_THRESHOLD, 2),
            budget_pct=round(budget_pct, 2),
            expected_profit=round(expected_profit, 2),
            reason=reason
        )

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.error(f"Error occurred while making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))