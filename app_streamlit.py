import streamlit as st
import requests

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Churn Decision System", layout="wide")

HORIZON = 12
RESCUE_RATE = 0.2   # from business.yaml
CONTACT_COST = 5    # from business.yaml
DISCOUNT = 50       # from business.yaml
SCORE_THRESHOLD = 300  # derived from policy simulation (score_risk_value, 15% budget)

# Get API URL from Streamlit Secrets
API_URL = st.secrets.get("API_URL", None)

# Toggle demo mode manually if needed
FORCE_DEMO_MODE = False


# ---------------------------
# DEMO DATA (dynamic fallback)
# Mirrors real API logic using actual business.yaml params
# Churn probability is a representative fixed value (0.28) —
# real probability requires the trained model which lives on EC2
# ---------------------------
def get_demo_response(monthly_charges):
    """
    Dynamic demo response that reflects user inputs.
    Churn probability is fixed at a representative value (0.28)
    since the trained model is not available without the backend.
    All other values are computed from actual business logic.
    """
    churn_prob = 0.28  # representative — real model runs on EC2
    value_proxy = round(monthly_charges * HORIZON, 2)
    score = round(churn_prob * value_proxy, 2)
    expected_profit = round(churn_prob * value_proxy * RESCUE_RATE - (CONTACT_COST + DISCOUNT), 2)
    decision = "TARGET" if score >= SCORE_THRESHOLD else "DO NOT TARGET"
    reason = (
        "High value customer with positive expected ROI"
        if score >= SCORE_THRESHOLD
        else "Score below threshold — not worth targeting at current value"
    )

    return {
        "churn_probability": churn_prob,
        "value_proxy": value_proxy,
        "score": score,
        "decision": decision,
        "score_threshold": SCORE_THRESHOLD,
        "expected_profit": expected_profit,
        "reason": reason
    }


# ---------------------------
# TITLE
# ---------------------------
st.title("💰 Churn Decision Intelligence System")
st.markdown("Optimize retention decisions using **ROI-driven targeting**")

st.divider()

# ---------------------------
# INPUTS
# ---------------------------
st.subheader("🧾 Customer Inputs")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure Months", value=12)
    monthly_charges = st.number_input("Monthly Charges", value=70.0)

    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

with col2:
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

st.divider()

# ---------------------------
# BUTTON
# ---------------------------
if st.button("🚀 Get Decision"):

    payload = {
        "Tenure Months": tenure,
        "Monthly Charges": monthly_charges,
        "Total Charges": monthly_charges * tenure,
        "Contract": contract,
        "Internet Service": internet_service,
        "Online Security": online_security,
        "Tech Support": tech_support,
        "Payment Method": payment_method,
        "Paperless Billing": paperless_billing,
    }

    result = None
    using_demo = False

    # ---------------------------
    # TRY REAL API
    # ---------------------------
    if not FORCE_DEMO_MODE and API_URL:
        try:
            response = requests.post(API_URL, json=payload, timeout=5)

            if response.status_code == 200:
                result = response.json()
            else:
                using_demo = True

        except:
            using_demo = True

    else:
        using_demo = True

    # ---------------------------
    # FALLBACK TO DEMO
    # ---------------------------
    if using_demo:
        result = get_demo_response(monthly_charges)
        st.warning(
            "⚠️ Running in DEMO MODE (backend not active). "
            "Churn probability is fixed at a representative value (0.28). "
            "Score, profit, and targeting decision reflect your actual inputs."
        )

    st.divider()

    # ---------------------------
    # METRICS
    # ---------------------------
    st.subheader("📊 Decision Result")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Churn Probability", round(result["churn_probability"], 3))
    col2.metric("Value Proxy ($)", round(result["value_proxy"], 2))
    col3.metric("Score", round(result["score"], 2))
    col4.metric("Expected Profit ($)", round(result["expected_profit"], 2))

    st.progress(result["churn_probability"])
    st.caption(f"Churn Risk: {round(result['churn_probability']*100,1)}%")

    st.divider()

    # ---------------------------
    # DECISION
    # ---------------------------
    if result["decision"] == "TARGET":
        st.success("🎯 TARGET this customer")
    else:
        st.error("❌ DO NOT TARGET")

    st.info(f"💡 Reason: {result['reason']}")

    st.divider()

    # ---------------------------
    # PROFIT EXPLANATION
    # ---------------------------
    st.subheader("💡 Profit Explanation")

    st.markdown("""
**Expected Profit = P(churn) × Value × Rescue Rate − Cost**

✔ Decisions are based on **ROI**, not just probability  
✔ High-value customers can be worth targeting even at moderate risk  
""")

    if result["expected_profit"] > 0:
        st.success("✅ Profitable to target")
    else:
        st.warning("⚠️ Not profitable at current value and assumed rescue rate")

    st.divider()

    # ---------------------------
    # THRESHOLD VISUAL
    # ---------------------------
    st.subheader("📈 Score vs Threshold")

    threshold = result["score_threshold"]
    score = result["score"]

    progress_ratio = min(score / (threshold * 1.5), 1.0)
    st.progress(progress_ratio)

    st.write(f"Score: {round(score, 2)} | Threshold: {round(threshold, 2)}")
    st.caption(
        "Threshold derived from policy simulation: "
        "score of the last customer in top 15% ranked by decision score (score_risk_value policy)"
    )

    if score >= threshold:
        st.success("Above threshold → Selected")
    else:
        st.warning("Below threshold → Not selected")

    st.divider()

    # ---------------------------
    # BUSINESS INSIGHTS
    # ---------------------------
    st.subheader("📊 Business Insights")

    st.markdown("""
✔ High churn ≠ always target  
✔ High value customers drive ROI  
✔ Only top-ranked customers are selected under budget  
✔ Decision = **risk × value + economics**
""")

    st.divider()

    # ---------------------------
    # WHAT-IF ANALYSIS
    # ---------------------------
    st.subheader("🧪 What-if Analysis")

    if "what_if_value" not in st.session_state:
        st.session_state.what_if_value = int(monthly_charges)

    new_charge = st.slider(
        "Adjust Monthly Charges",
        20,
        200,
        int(st.session_state.what_if_value),
        key="what_if_value"
    )

    new_value = new_charge * HORIZON
    estimated_score = round(result["churn_probability"] * new_value, 2)
    estimated_profit = round(result["churn_probability"] * new_value * RESCUE_RATE - (CONTACT_COST + DISCOUNT), 2)

    col1, col2 = st.columns(2)
    col1.metric("New Value Proxy ($)", round(new_value, 2))
    col2.metric("Estimated Score", estimated_score)

    st.write(f"Estimated Profit: ${estimated_profit}")

    if estimated_score >= threshold:
        st.success("🚀 Would become TARGETABLE")
    elif estimated_score >= threshold * 0.8:
        st.info("📈 Close to threshold — small value increase would tip the decision")
    else:
        st.warning("Still below threshold")

    st.caption("Higher value → higher score → higher chance of targeting")

    st.divider()

    # ---------------------------
    # SYSTEM EXPLANATION
    # ---------------------------
    with st.expander("📘 How this system works"):
        st.markdown("""
1. **Predict** churn probability using calibrated XGBoost
2. **Estimate** customer value (Monthly Charges × 12 months)
3. **Compute** decision score = churn probability × value proxy
4. **Rank** all customers by decision score
5. **Apply** profit-based threshold under budget constraints
6. **Output** targeting decision + expected profit

**Campaign parameters (from business.yaml):**
- Contact cost: $5 per customer
- Discount: $50 per customer
- Rescue rate: 20%
- Value horizon: 12 months
- Threshold derived from: score_risk_value policy, 15% budget
""")