# main.py
# CredTech: Single-file, upload-ready baseline
# --------------------------------------------
# Features:
# - Config & lightweight data bootstrap (yfinance + RSS)
# - Toy feature engineering
# - Logistic Regression baseline (interpretable)
# - Explanations from model coefficients (feature contributions)
# - FastAPI endpoint: /health, /score
# - Streamlit dashboard that calls the API
#
# Usage:
#   python main.py api      # start API (FastAPI)
#   python main.py ui       # start UI  (Streamlit)
#
# Requirements (put these in requirements.txt):
#   pandas
#   numpy
#   scikit-learn
#   fastapi
#   uvicorn
#   streamlit
#   requests
#   yfinance
#   feedparser
#   python-dotenv (optional)

import os
import sys
import time
import json
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

import numpy as np
import pandas as pd

# Optional deps imported where used:
# - yfinance
# - feedparser

# ML
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# API
from fastapi import FastAPI
from pydantic import BaseModel

# --------------------------------------------
# Config
# --------------------------------------------
ISSUERS = [
    {"name": "TCS", "ticker": "TCS.NS"},
    {"name": "Reliance", "ticker": "RELIANCE.NS"},
    {"name": "HDFC Bank", "ticker": "HDFCBANK.NS"},
]

RSS_FEEDS = [
    "https://www.reuters.com/finance/markets/rss",
    "https://www.moneycontrol.com/rss/latestnews.xml",
]

CACHE_DIR = ".cache"
DATA_DIR = "data"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Global model objects (warmed by bootstrap)
MODEL: Optional[LogisticRegression] = None
FEATURE_COLUMNS: List[str] = []
LAST_TRAIN_AUC: Optional[float] = None


# --------------------------------------------
# Data Ingestion
# --------------------------------------------
def fetch_prices_yf(tickers: List[str], period="2y", interval="1d") -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("yfinance not installed. Add it to requirements.txt")

    data = yf.download(tickers=tickers, period=period, interval=interval, auto_adjust=True, threads=True)
    # If multi-index columns (Open/High/Low/Close), take Close
    if isinstance(data, pd.DataFrame):
        if "Close" in data:
            data = data["Close"]
        data = data.reset_index().rename(columns={"Date": "date"})
    return data


def fetch_rss(feed_urls: List[str]) -> pd.DataFrame:
    try:
        import feedparser
    except ImportError:
        raise RuntimeError("feedparser not installed. Add it to requirements.txt")

    rows = []
    for url in feed_urls:
        d = feedparser.parse(url)
        for e in d.entries[:200]:
            rows.append({
                "published": getattr(e, "published", None),
                "title": getattr(e, "title", ""),
                "link": getattr(e, "link", ""),
                "feed": url
            })
    return pd.DataFrame(rows)


# --------------------------------------------
# Feature Engineering
# --------------------------------------------
def make_toy_features(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Input: wide price frame with columns: date, <ticker1>, <ticker2>, ...
    Output: per-date features pooled across tickers (average across issuers)
    """
    df = df_prices.copy()
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    df = df.sort_values("date").reset_index(drop=True)

    price_cols = [c for c in df.columns if c != "date"]
    # returns and volatility windows
    windows = [7, 30, 90]
    for w in windows:
        df[f"ret_{w}"] = df[price_cols].pct_change(w).mean(axis=1)
        df[f"vol_{w}"] = df[price_cols].pct_change().rolling(w).std().mean(axis=1)

    # macro-ish proxy: drawdown over 90d
    df["max_90"] = df[price_cols].rolling(90).max().mean(axis=1)
    df["avg_price"] = df[price_cols].mean(axis=1)
    df["drawdown_90"] = (df["avg_price"] - df["max_90"]) / (df["max_90"] + 1e-9)

    # Toy target: flag if future 5-day average return is negative (classification)
    df["fwd_ret_5"] = df[price_cols].pct_change(5).shift(-5).mean(axis=1)
    df["target"] = (df["fwd_ret_5"] < 0).astype(int)

    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        "ret_7", "ret_30", "ret_90",
        "vol_7", "vol_30", "vol_90",
        "drawdown_90"
    ]
    return df[["date"] + feature_cols + ["target"]], feature_cols


# --------------------------------------------
# Modeling & Explanations
# --------------------------------------------
def train_logistic(df: pd.DataFrame, feature_cols: List[str]) -> (LogisticRegression, float):
    X = df[feature_cols].astype(float)
    y = df["target"].astype(int)
    # time-ordered split (no shuffle) to avoid leakage
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LogisticRegression(max_iter=2000)
    model.fit(Xtr, ytr)
    auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
    return model, auc


def explain_with_coefs(model: LogisticRegression, features: Dict[str, float]) -> List[Dict[str, float]]:
    """
    Local contribution via linear model:
    contribution_i ≈ coef_i * x_i  (bias excluded)
    Sorted by absolute impact.
    """
    coefs = model.coef_.ravel()
    names = list(features.keys())
    vals = np.array([features[k] for k in names], dtype=float)
    contribs = coefs * vals
    items = [{"name": n, "value": float(v), "contribution": float(c)} for n, v, c in zip(names, vals, contribs)]
    items.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    return items


# --------------------------------------------
# Bootstrap pipeline (called by API start)
# --------------------------------------------
def bootstrap():
    global MODEL, FEATURE_COLUMNS, LAST_TRAIN_AUC

    tickers = [x["ticker"] for x in ISSUERS]
    prices_path = os.path.join(DATA_DIR, "prices.csv")
    news_path = os.path.join(DATA_DIR, "news.csv")

    # Download or load cached prices
    if not os.path.exists(prices_path):
        print("Fetching prices via yfinance...")
        dfp = fetch_prices_yf(tickers)
        dfp.to_csv(prices_path, index=False)
    else:
        dfp = pd.read_csv(prices_path, parse_dates=["date"])

    # Fetch RSS (not used in baseline features yet; store for UI)
    if not os.path.exists(news_path):
        print("Fetching news RSS...")
        dfn = fetch_rss(RSS_FEEDS)
        dfn.to_csv(news_path, index=False)

    # Build features and train model
    df_feat, FEATURE_COLUMNS = make_toy_features(dfp)
    MODEL, LAST_TRAIN_AUC = train_logistic(df_feat, FEATURE_COLUMNS)

    # Persist minimal artifacts (optional)
    with open(os.path.join(CACHE_DIR, "model_meta.json"), "w") as f:
        json.dump(
            {
                "trained_at": datetime.utcnow().isoformat(),
                "auc": LAST_TRAIN_AUC,
                "features": FEATURE_COLUMNS,
            },
            f,
            indent=2,
        )
    print(f"Bootstrap complete. AUC={LAST_TRAIN_AUC:.3f}")


# --------------------------------------------
# FastAPI App
# --------------------------------------------
app = FastAPI(title="CredTech API (Single-file)", version="0.1.0")


class ScoreRequest(BaseModel):
    issuer: str


@app.get("/health")
def health():
    meta_path = os.path.join(CACHE_DIR, "model_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
    return {"status": "ok", "time": datetime.utcnow().isoformat(), "model": meta}


@app.post("/score")
def score(req: ScoreRequest):
    """
    For demo, build a simple 'latest' feature row from recent data and apply the model.
    Since our features are pooled, issuer input is used for UI context and event feed filtering in a full version.
    """
    if MODEL is None:
        bootstrap()

    prices_path = os.path.join(DATA_DIR, "prices.csv")
    dfp = pd.read_csv(prices_path, parse_dates=["date"]).sort_values("date")
    df_feat, feature_cols = make_toy_features(dfp)
    latest = df_feat.iloc[-1]
    x = latest[feature_cols].to_dict()

    # Predict
    p1 = float(MODEL.predict_proba(latest[feature_cols].values.reshape(1, -1))[0, 1])

    # Local coef-based explanation (linear model)
    top_feats = explain_with_coefs(MODEL, x)[:6]

    # Minimal event placeholders (for UI)
    events = [
        {"headline": "Earnings outlook stable; margin guidance positive", "impact": "+", "confidence": 0.7},
        {"headline": "Sector demand moderation reported by peers", "impact": "-", "confidence": 0.6},
    ]

    return {
        "issuer": req.issuer,
        "score": p1,
        "timestamp": datetime.utcnow().isoformat(),
        "top_features": top_feats,
        "events": events,
    }


# --------------------------------------------
# Streamlit UI entry (launched with: python main.py ui)
# --------------------------------------------
def run_streamlit():
    import streamlit as st
    import requests

    st.set_page_config(page_title="CredTech Dashboard", layout="wide")
    st.title("CredTech: Explainable Credit Intelligence (Baseline)")

    issuers = [x["name"] for x in ISSUERS]
    issuer = st.selectbox("Select issuer", issuers)

    api_url = st.text_input("API URL", "http://localhost:8000")
    st.caption("Start the API in another terminal:  `python main.py api`")

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Get latest score"):
            try:
                resp = requests.post(f"{api_url}/score", json={"issuer": issuer}, timeout=20)
                if resp.status_code == 200:
                    data = resp.json()
                    st.metric(label=f"{issuer} score (prob of stress)", value=round(data["score"], 3))
                    st.subheader("Top feature contributions")
                    st.dataframe(pd.DataFrame(data["top_features"]))
                    st.subheader("Recent events (demo)")
                    st.dataframe(pd.DataFrame(data["events"]))
                else:
                    st.error(f"API error: {resp.status_code} - {resp.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

    with col2:
        st.subheader("Model Health")
        try:
            h = requests.get(f"{api_url}/health", timeout=10).json()
            st.json(h)
        except Exception:
            st.info("API not reachable yet.")

        # Show latest features for transparency
        prices_path = os.path.join(DATA_DIR, "prices.csv")
        if os.path.exists(prices_path):
            dfp = pd.read_csv(prices_path, parse_dates=["date"]).sort_values("date")
            df_feat, feature_cols = make_toy_features(dfp)
            st.subheader("Latest engineered features (pooled)")
            st.dataframe(df_feat[feature_cols].tail(5).reset_index(drop=True))

    st.markdown("---")
    st.caption("Explainability uses linear model coefficients (no LLM). Upgrade path: LightGBM + SHAP.")


# --------------------------------------------
# CLI Entrypoint
# --------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CredTech single-file app")
    parser.add_argument("mode", choices=["api", "ui"], help="Run mode: api|ui")
    parser.add_argument("--host", default="0.0.0.0", help="API host (api mode)")
    parser.add_argument("--port", default=8000, type=int, help="API port (api mode)")
    args = parser.parse_args()

    if args.mode == "api":
        # ensure model/artifacts are ready
        bootstrap()
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.mode == "ui":
        # Launch Streamlit app defined above
        # Equivalent to: streamlit run main.py -- (but we call function directly)
        run_streamlit()


if __name__ == "__main__":
    main()
