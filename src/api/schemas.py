from pydantic import BaseModel, Field, field_validator
from typing import Literal


class CustomerFeatures(BaseModel):

    tenure_months: int = Field(..., alias="Tenure Months",examples=[1, 12, 24, 36, 48, 60])
    monthly_charges: float = Field(..., alias="Monthly Charges",examples=[29.85, 56.95, 53.85, 42.30, 70.70])
    total_charges: float = Field(..., alias="Total Charges")

    contract: str = Field(..., alias="Contract",examples=["Month-to-month", "One year", "Two year"])

    internet_service: str = Field(..., alias="Internet Service",examples=["DSL", "Fiber optic", "No"])

    online_security: str = Field(..., alias="Online Security",examples=["Yes", "No", "No internet service"])

    tech_support: str = Field(..., alias="Tech Support",examples=["Yes", "No", "No internet service"])

    payment_method: str = Field(..., alias="Payment Method",examples=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    

    paperless_billing: str = Field(..., alias="Paperless Billing",examples=["Yes", "No"])

    # Optional / defaulted fields
    gender: Literal["Male", "Female"] = Field("Male", alias="Gender")
    senior_citizen: Literal["Yes", "No"] = Field("No", alias="Senior Citizen")
    partner: Literal["Yes", "No"] = Field("No", alias="Partner")
    dependents: Literal["Yes", "No"] = Field("No", alias="Dependents")

    phone_service: Literal["Yes", "No"] = Field("Yes", alias="Phone Service")
    multiple_lines: Literal["Yes", "No", "No phone service"] = Field("No", alias="Multiple Lines")

    online_backup: Literal["Yes", "No", "No internet service"] = Field("No", alias="Online Backup")
    device_protection: Literal["Yes", "No", "No internet service"] = Field("No", alias="Device Protection")

    streaming_tv: Literal["Yes", "No", "No internet service"] = Field("No", alias="Streaming TV")
    streaming_movies: Literal["Yes", "No", "No internet service"] = Field("No", alias="Streaming Movies")
    
    @field_validator("internet_service")
    @classmethod
    def normalize_internet_service(cls, v):
        v_clean = v.strip().lower()

        mapping = {
            "dsl": "DSL",
            "fiber optic": "Fiber optic",
            "no": "No"
       }

        if v_clean not in mapping:
            raise ValueError(
            "Invalid Internet Service. Allowed values: DSL, Fiber optic, No"
        )

        return mapping[v_clean]
    
    @field_validator("contract")
    @classmethod
    def normalize_contract(cls, v):
        v_clean = v.strip().lower()

        mapping = {
            "month-to-month": "Month-to-month",
            "one year": "One year",
            "two year": "Two year"
       }

        if v_clean not in mapping:
            raise ValueError(
            "Invalid Contract. Allowed values: Month-to-month, One year, Two year"
        )

        return mapping[v_clean]
    
    @field_validator("payment_method")
    @classmethod    
    def normalize_payment_method(cls, v):
        v_clean = v.strip().lower()

        mapping = {
            "electronic check": "Electronic check",
            "mailed check": "Mailed check",
            "bank transfer (automatic)": "Bank transfer (automatic)",
            "credit card (automatic)": "Credit card (automatic)"
       }

        if v_clean not in mapping:
            raise ValueError(
            "Invalid Payment Method. Allowed values: Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)"
        )

        return mapping[v_clean]
    
    @field_validator("paperless_billing")
    @classmethod   
    def normalize_paperless_billing(cls, v):
        v_clean = v.strip().lower()

        mapping = {
            "yes": "Yes",
            "no": "No"
       }

        if v_clean not in mapping:
            raise ValueError(
            "Invalid Paperless Billing. Allowed values: Yes, No"
        )

        return mapping[v_clean]
    
    @field_validator('online_security', 'tech_support', 'online_backup', 'device_protection', 'streaming_tv', 'streaming_movies')
    @classmethod
    def normalize_optional_fields(cls, v):
        v_clean = v.strip().lower()

        mapping = {
            "yes": "Yes",
            "no": "No",
            "no internet service": "No internet service"
        }

        if v_clean not in mapping:
            raise ValueError(
                f"Invalid {cls.__name__}. Allowed values: Yes, No, No internet service"
            )

        return mapping[v_clean]

    class Config:
        populate_by_name = True
        schema_extra = {
            "example": {
                "Tenure Months": 12,
                "Monthly Charges": 56.95,
                "Total Charges": 683.70,
                "Contract": "One year",
                "Internet Service": "Fiber optic",
                "Online Security": "No",
                "Tech Support": "No",
                "Payment Method": "Mailed check",
                "Paperless Billing": "Yes"
            }
        }
    
class PredictionResponse(BaseModel):
    churn_probability: float
    value_proxy: float
    score: float

class DecisionResponse(BaseModel):
    churn_probability: float
    value_proxy: float
    score: float
    decision: str
    score_threshold: float
    budget_pct: float
    expected_profit: float
    reason: str