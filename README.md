# 🎮 PUBG Analytics Dashboard

A Machine Learning powered analytics dashboard built on **4.4 Million real PUBG match records**.

🔴 **Live App:** [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pubg-analytics-qk5qyu4k32cusjav7s5upe.streamlit.app/)

---

## 📌 Project Overview

This project analyzes real PUBG match data to build two ML models:
- **Survival Predictor** — Predicts if a player can finish in Top 10
- **Player Segmentation** — Classifies players as Noob, Average, or Pro

---

## 🚀 Features

- 🎯 **Survival Predictor** — Enter your match stats and get your Top 10 probability with actionable tips
- 👥 **Player Segmentation** — Find out your player type using K-Means Clustering
- 📊 **EDA & Insights** — Interactive charts from 4.4M match records

---

## 📂 Dataset

- **Source:** [PUBG Finish Placement Prediction — Kaggle](https://www.kaggle.com/c/pubg-finish-placement-prediction/data?select=train_V2.csv)
- **Size:** 4,446,966 rows × 29 columns
- **Target:** `winPlacePerc` — Player's finish placement percentile

---

## 🤖 Models

### Model 1 — Survival Predictor
- **Algorithm:** Random Forest Classifier
- **Features:** 15 gameplay features (kills, walkDistance, boosts, heals etc.)
- **Threshold:** 0.70 (optimized for Precision)
- **Accuracy:** 88.47%
- **Winner Precision:** 0.47 | **Winner Recall:** 0.75

### Model 2 — Player Segmentation
- **Algorithm:** K-Means Clustering (K=3)
- **Features:** 11 gameplay behavior features
- **Training Data:** Full 4.4M rows (MiniBatchKMeans)
- **Clusters:** Noob | Average | Pro

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| ML | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Deployment | Streamlit Cloud |
| Version Control | GitHub |

---

## 📊 Key Insights

- **walkDistance** is the most important feature (41% importance) — survival > kills
- Winners use **336% more boosts** than losers
- Winners walk **193% more distance** than losers
- Solo matches have higher kills but Squad matches have better survival rates

---

## 🗂️ Project Structure
pubg-analytics/
├── app.py                  — Streamlit app
├── rf_survival_model.pkl   — Survival Predictor model
├── kmeans_full_model.pkl   — Player Segmentation model
├── scaler_full.pkl         — StandardScaler
├── model_config.json       — Threshold config
├── winner_stats.csv        — EDA data
├── match_stats.csv         — EDA data
├── corr_matrix.csv         — Correlation data
├── feature_importance.csv  — Feature importance data
└── requirements.txt        — Dependencies

---

## 👨‍💻 Author

**Aryan Kasundra**
- GitHub: [@aryankasundra509](https://github.com/aryankasundra509)