CredTech: Explainable Credit Intelligence

Tagline: Real-time explainable credit scoring with multi-source data, SHAP-like insights, and an interactive dashboard.


---

🔍 Overview

CredTech is an end-to-end platform that delivers credit risk scores with explanations.
It combines structured data (finance, macro indicators) and unstructured data (news headlines, filings) to provide transparent, adaptive, and analyst-friendly insights.

FastAPI backend → real-time credit scoring API

Streamlit dashboard → issuer selector, score timeline, feature breakdown, events feed

Logistic Regression baseline (interpretable) → easily upgradeable to LightGBM/XGBoost + SHAP

Adaptive learning → online updates with river (planned)

Docker-ready → reproducible and deployable anywhere



---

⚡ Quickstart

# Install dependencies
pip install -r requirements.txt

# Run API (Terminal 1)
python main.py api   # http://localhost:8000

# Run Dashboard (Terminal 2)
python main.py ui    # http://localhost:8501


---

📊 Example Output

API Response (POST /score):

{
  "issuer": "TCS",
  "score": 0.72,
  "timestamp": "2025-08-22T12:00:00Z",
  "top_features": [
    {"name": "ret_30", "value": -0.012, "contribution": -0.18},
    {"name": "vol_30", "value": 0.021, "contribution": -0.12}
  ],
  "events": [
    {"headline": "Earnings outlook stable; margin guidance positive", "impact": "+", "confidence": 0.7},
    {"headline": "Sector demand moderation reported by peers", "impact": "-", "confidence": 0.6}
  ]
}


---

🛠 Roadmap

✅ Baseline Logistic model with coef explanations

🔜 Gradient-boosted models (LightGBM/XGBoost) + SHAP explainability

🔜 Incremental learning with river for intra-day updates

🔜 Event-to-factor mapping from unstructured news

🔜 Alerts on score changes / anomalies

🔜 CI/CD pipeline and cloud deployment

