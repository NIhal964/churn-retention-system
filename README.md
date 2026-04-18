#  Value-Aware Churn Optimization & Decision System

##  Overview

Traditional churn models answer:

> “Who will churn?”

This system answers a more important question:

> **“Who should we target to maximize retained revenue under budget constraints?”**

By integrating **churn prediction, customer value, and campaign economics**, this system transforms predictions into **profit-driven decisions**.

---

##  Key Results

* ~**3× lift** in retained value within top-decile targeting
* ~**5–10% higher expected profit** vs probability-only targeting
* **Break-even rescue rate:** ~8–10%
* Optimal targeting range: **10–15% of customers**

 Demonstrates that **value-aware targeting significantly improves ROI**

## 📊 Policy Comparison

![Policy Comparison](assets/policy_comparison.png)

Value-aware targeting significantly outperforms probability-only targeting in expected profit.

## 📈 Profit vs Budget

![Profit vs Budget](assets/profit_vs_budget.png)

Optimal targeting occurs around 10–15% of customers, balancing cost and expected return.

## 🎯 Threshold Optimization

![Threshold Curve](assets/threshold_curve.png)

Profit-based threshold selection avoids arbitrary cutoff decisions.

---

## Core Idea

```text
Prediction ≠ Decision
```

A high churn probability customer is not always worth targeting.

We use:

```text
Decision Score = churn_probability × value_proxy
```

to prioritize **high-value, persuadable customers**

---

##  System Architecture

```text
Data Pipeline → Model → Calibration → Decision Layer → API → Monitoring
```

### Layers

* **Data Pipeline** → ingestion, validation, splitting
* **Modeling** → churn prediction (XGBoost)
* **Calibration** → reliable probabilities (Isotonic Regression)
* **Decision Layer** → profit-aware targeting
* **API** → FastAPI deployment
* **Monitoring** → drift detection using baseline comparison

---

##  Data Pipeline

Modular pipeline design:

```text
src/data/
  ├── load.py
  ├── validate.py
  ├── split.py
  └── prepare.py
```

Ensures:

* reproducibility
* clean train/test separation
* leakage prevention

---

##  Modeling

### Models Evaluated

* Logistic Regression (baseline)
* XGBoost (final)

### Final Model Performance

* ROC-AUC: **0.86**
* PR-AUC: **0.68**

 Selected for strong ranking performance

---

##  Probability Calibration

Tree models are often miscalibrated.

Applied **Isotonic Regression**:

* Calibrated ROC-AUC: **0.859**
* Brier Score: **0.13**

 Enables reliable **financial decision-making**

---

## 📊 Ranking Performance

Retention campaigns target a small fraction of users.

* Precision@5%: **0.83**
* Precision@10%: **0.79**
* Lift@10%: **~3×**

Model identifies significantly more churners than random targeting

---

##  Decision Layer

### Policies

* Risk-only → P(churn)
* Risk × Value → P(churn) × ValueProxy
* Weighted strategy → penalizes extreme cases

### Insight

```text
Value-aware targeting consistently outperforms probability-only strategies
```

---

## 📈 Economic Simulation

Expected profit:

```text
Expected Profit =
Σ(P(churn) × ValueProxy × rescue_rate)
− Campaign Cost
```

Simulates:

* budget constraints
* rescue rate
* campaign cost

---

##  Key Analyses

### Profit vs Budget

* Shows optimal targeting size (~10–15%)

### Profit vs Rescue Rate

* Break-even: ~8–10%

### Profit Heatmap

* Identifies profitable operating regions

---

##  Deployment

### API

```bash
uvicorn src.api.app:app --reload
```

### Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

---
##  CI/CD

Implemented using **GitHub Actions**

Validates:

* dependencies
* API startup
* test cases
* Docker build

 Ensures reliable and reproducible deployments

---

##  Monitoring

Tracks:

* prediction distribution (churn probability)
* feature distribution (e.g., Monthly Charges)
* % high-risk customers

### Approach

* Baseline saved during training
* Live predictions compared to baseline

Example:

```text
Current avg churn probability: 0.1700
Baseline avg churn probability: 0.2667
⚠ Drift detected
```

 Enables early detection of data and prediction drift

---

## 🧪 Testing

```bash
pytest tests/
```

Validates:

* API availability
* prediction behavior (high vs low churn)

---

##  Project Structure

```text
src/
  ├── api/
  ├── data/
  ├── features/
  ├── modeling/
  ├── decisioning/
  ├── pipeline/
  └── monitoring/
```

---

##  Limitations

* Uses value proxy instead of full CLTV
* Simulation assumes fixed rescue rate
* Drift detection based on statistical thresholds

---

## 🔮 Future Work

* uplift modeling (causal targeting)
* automated retraining pipelines
* advanced monitoring (Prometheus/Grafana)

---

##  Key Takeaways

* Optimize **profit**, not just accuracy
* Calibration is critical for decisions
* Decision layer > prediction alone
* Monitoring is essential for production ML
