from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_api_health():
    response = client.get("/")
    assert response.status_code == 200


def test_high_churn_customer():
    payload = {
        "Tenure Months": 1,
        "Monthly Charges": 95,
        "Total Charges": 95,
        "Contract": "Month-to-month",
        "Internet Service": "Fiber optic",
        "Online Security": "No",
        "Tech Support": "No",
        "Payment Method": "Electronic check",
        "Paperless Billing": "Yes",
        "Gender": "Male",
        "Senior Citizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "Phone Service": "Yes",
        "Multiple Lines": "Yes",
        "Online Backup": "No",
        "Device Protection": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "Yes"
    }

    response = client.post("/decision", json=payload)
    data = response.json()

    assert response.status_code == 200
    assert data["churn_probability"] > 0.5


def test_low_churn_customer():
    payload = {
        "Tenure Months": 60,
        "Monthly Charges": 100,
        "Total Charges": 6000,
        "Contract": "Two year",
        "Internet Service": "Fiber optic",
        "Online Security": "Yes",
        "Tech Support": "Yes",
        "Payment Method": "Credit card (automatic)",
        "Paperless Billing": "No",
        "Gender": "Female",
        "Senior Citizen": "No",
        "Partner": "Yes",
        "Dependents": "Yes",
        "Phone Service": "Yes",
        "Multiple Lines": "Yes",
        "Online Backup": "Yes",
        "Device Protection": "Yes",
        "Streaming TV": "Yes",
        "Streaming Movies": "Yes"
    }

    response = client.post("/decision", json=payload)
    data = response.json()

    assert response.status_code == 200
    assert data["churn_probability"] < 0.5