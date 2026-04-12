# Customer Churn Retention Optimization System

A decision-focused machine learning system designed to **maximize retained revenue from customer retention campaigns under budget constraints**.

Unlike typical churn projects that stop at prediction accuracy, this system connects churn predictions to **real business decisions** by evaluating how different targeting strategies impact expected profit.

---

# Problem

Retention teams often face the question:

> Given a limited retention budget, which customers should we target to maximize saved revenue?

A churn prediction model alone cannot answer this.  
This project combines **churn prediction, customer value estimation, and economic simulation** to determine the most profitable retention strategy.

---

# System Architecture

The project is organized into three layers:
Data → Modeling → Decision
.
├── .github/
│   └── workflows/
│       └── ci.yml                  # CI pipeline (tests, validation checks)
│
├── configs/                        # Configuration files
│
├── data/
│   ├── raw/                        # Original dataset
│   ├── interim/                    # Intermediate artifacts
│   └── processed/                  # Cleaned and split datasets
│
├── src/
│   ├── data/
│   │   ├── load.py                 # Load raw dataset
│   │   ├── validate.py             # Data validation checks
│   │   └── split.py                # Train / validation / test split
│   │
│   ├── features/
│   │   ├── schema.py               # Feature schema and dropped columns
│   │   └── build.py                # Feature engineering pipeline
│   │
│   ├── modeling/
│   │   ├── train.py                # Train models (Logistic + XGBoost)
│   │   ├── calibrate.py            # Probability calibration
│   │   ├── predict.py              # Generate churn predictions
│   │   └── evaluate.py             # Ranking metrics (Precision@K, Lift)
│   │
│   ├── decisioning/
│   │   ├── value.py                # Customer value proxy calculation
│   │   ├── policy.py               # Retention targeting policies
│   │   ├── profit_curve.py         # Profit vs targeting budget simulation
│   │   ├── rescue_sensitivity.py   # Profit vs rescue rate analysis
│   │   └── profit_heatmap.py       # Budget × rescue rate profit heatmap
│   │
│   └── monitoring/
│       ├── drift.py                # Data / prediction drift detection
│       └── reports.py              # Monitoring and evaluation reports
│
├── notebooks/
│   └── 01_eda.ipynb                # Exploratory data analysis
│
├── requirements.txt                # Project dependencies
├── .gitignore
└── README.md

This structure separates **data processing, model development, and business decision logic**, similar to production ML systems.

---

# Dataset

Telecom churn dataset containing customer attributes such as:

- tenure
- contract type
- internet service
- monthly charges
- payment method
- service subscriptions

Target variable:
Churn Value

Leakage-prone columns such as **CLTV, churn reason, and churn score** were removed to ensure realistic modeling.

---

# Modeling

Two models were evaluated:

### Logistic Regression (baseline)
ROC-AUC ≈ 0.856
PR-AUC ≈ 0.675


### XGBoost (final model)
ROC-AUC ≈ 0.858
PR-AUC ≈ 0.682

XGBoost was selected due to **better ranking performance for retention targeting**.

---

# Probability Calibration

Tree models often produce poorly calibrated probabilities.

Predictions were calibrated using **Isotonic Regression** to ensure reliable probabilities for decision simulations.

Calibrated ROC-AUC ≈ 0.859
Brier Score ≈ 0.13


---

# Ranking Performance

Retention campaigns target only a small fraction of customers, so ranking quality is critical.

Example results:
Precision@5% ≈ 0.83
Precision@10% ≈ 0.79
Recall@10% ≈ 0.30
Lift@10% ≈ 2.98

The model captures **nearly 3× more churners than random targeting** in the top 10%.

---

# Retention Targeting Policies

Three targeting strategies were evaluated.

### Risk-Only
Score = P(churn)

Targets customers with the highest churn probability.

### Risk × Value
Score = P(churn) × ValueProxy

Prioritizes **high-value customers at risk of leaving**.

### Sensitivity-Weighted
Score = P(churn) × ValueProxy × W(p)

Downweights customers extremely unlikely to churn or already too likely to leave.

---

# Economic Simulation

Retention campaigns incur costs such as:

- contact cost
- discount incentives

Expected profit is estimated as:
ExpectedProfit =
Σ(P(churn) × ValueProxy × rescue_rate)
− CampaignCost

Where **rescue rate** represents campaign effectiveness.

---

# Key Analyses

### Profit vs Budget

Evaluates how campaign profitability changes with targeting size.

Insight:
Value-aware targeting outperforms risk-only targeting.

---

### Profit vs Rescue Rate

Measures sensitivity to campaign effectiveness.

Insight:
Break-even rescue rate ≈ 8–10%

Campaigns below this effectiveness lose money.

---

### Profit Heatmap

Visualizes expected profit across:
Targeting Budget × Rescue Rate


This identifies the **safe operating region for retention campaigns**.

Example insight:


Budget: 12–15%
Rescue Rate ≥ 30%
→ strong profitability


---

# Key Business Insights

- Value-aware targeting significantly improves retention ROI
- Targeting ~10–15% of customers captures most recoverable value
- Campaign effectiveness must exceed ~8–10% to break even
- High-value customers should be prioritized even if churn risk is slightly lower

---

# Technologies


Python
Pandas
Scikit-learn
XGBoost
NumPy
Matplotlib


---

# Future Improvements

Planned extensions include:

- MLflow experiment tracking
- CI/CD pipelines with GitHub Actions
- Dockerized inference
- model monitoring and automated retraining
