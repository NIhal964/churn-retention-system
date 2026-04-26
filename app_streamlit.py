import streamlit as st
import requests

# ---------------------------
# CONFIG
# ---------------------------
API_URL = "http://3.19.131.246:8000/decision"  # your EC2 API

HORIZON = 12  # same as backend value proxy

st.set_page_config(page_title="Churn Decision System", layout="wide")

# ---------------------------
# TITLE
# ---------------------------
st.title("💰 Churn Decision Intelligence System")
st.markdown("Optimize retention decisions using **ROI-driven targeting**")

st.divider()

# ---------------------------
# INPUT SECTION
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
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
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

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:

        result = response.json()

        st.divider()
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
        # DECISION + REASON
        # ---------------------------
        if result["decision"] == "TARGET":
            st.success("🎯 TARGET this customer")
        else:
            st.error("❌ DO NOT TARGET")

        st.info(f"💡 Reason: {result['reason']}")

        # ---------------------------
        # PROFIT EXPLANATION
        # ---------------------------
        st.subheader("💡 Profit Explanation")

        st.markdown(
            """
            **Expected Profit = P(churn) × Value × Rescue Rate − Cost**

            ✔ Only customers with **positive expected profit** should be targeted  
            ✔ Not all high-risk customers are worth targeting  
            ✔ Not all low-risk customers are useless (if value is high)
            """
        )

        if result["expected_profit"] > 0:
            st.success("✅ Profitable to target")
        else:
            st.warning("⚠️ Negative ROI → Avoid targeting")

        st.divider()

        # ---------------------------
        # SCORE VS THRESHOLD VISUAL
        # ---------------------------
        st.subheader("📈 Score vs Targeting Threshold")

        threshold = result["score_threshold"]
        score = result["score"]

        st.write(f"Score: {round(score,2)} | Threshold: {round(threshold,2)}")

        progress_ratio = min(score / threshold, 1.0)
        st.progress(progress_ratio)

        if score >= threshold:
            st.success("Above threshold → Selected")
        else:
            st.warning("Below threshold → Not selected")

        st.divider()

        # ---------------------------
        # BUSINESS INSIGHTS
        # ---------------------------
        st.subheader("📊 Business Insights")

        st.markdown(
            """
            ✔ High churn ≠ always target  
            ✔ High value customers can be targeted even at moderate risk  
            ✔ Decisions are driven by **expected ROI, not probability alone**  
            ✔ Budget constraints define final targeting set  
            """
        )

        st.divider()

        # ---------------------------
        # WHAT-IF ANALYSIS (FIXED)
        # ---------------------------
        st.subheader("🧪 What-if Scenario")

        if "what_if_value" not in st.session_state:
            st.session_state.what_if_value = monthly_charges

        new_charge = st.slider(
            "Adjust Monthly Charges",
            min_value=20,
            max_value=200,
            value=int(st.session_state.what_if_value),
            key="what_if_value"
        )

        new_value = new_charge * HORIZON
        estimated_score = result["churn_probability"] * new_value

        st.write(f"New Value Proxy: ${round(new_value,2)}")
        st.write(f"Estimated Score: {round(estimated_score,2)}")

        if estimated_score > threshold:
            st.success("🚀 This customer WOULD become TARGETABLE")
        else:
            st.warning("Still below targeting threshold")

        st.caption("Higher value → higher score → higher chance of selection")

        st.divider()

        # ---------------------------
        # SYSTEM EXPLANATION
        # ---------------------------
        with st.expander("📘 How this system works"):
            st.markdown(
                """
                1. Predict churn probability using ML model  
                2. Estimate customer value (12-month proxy)  
                3. Compute expected profit  
                4. Apply profit-optimized threshold  
                5. Target only high-value profitable customers  
                """
            )

    else:
        st.error("API Error ❌")