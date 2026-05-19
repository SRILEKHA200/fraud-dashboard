# Fraud Detection Dashboard

**Live Demo:** https://fraud-dashboard-srilekha.streamlit.app

## Overview
End-to-end fraud detection system with ML model, SHAP explainability, and interactive Streamlit dashboard.

## Features
- **Multi-page Streamlit app**: Overview, Investigate, Explain
- **Business Impact metrics**: Fraud prevented, Detection rate, False positives
- **Interactive Plotly charts**: Risk tiers, hourly fraud rates
- **SHAP explainability**: Waterfall plots for individual transaction decisions
- **Case management**: Search, filter, and investigate flagged transactions

## Tech Stack
Streamlit • Plotly • SHAP • XGBoost • Scikit-learn • Pandas

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
