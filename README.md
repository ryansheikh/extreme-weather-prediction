# 🌦️ AI-Driven Extreme Weather Prediction System
## A Global Perspective — 20 Cities, 4 ML Pipelines, Real-Time Dashboard

[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-blue)](https://xgboost.readthedocs.io)

---

## 📋 Project Overview

This project implements an **end-to-end AI system for extreme weather prediction and early warning** across 20 global cities spanning 6 continents. The system uses **XGBoost-based machine learning pipelines** with separate models for temperature forecasting, rainfall prediction, heatwave detection, and multi-class disaster classification.

### Key Features
- **🌍 Global Coverage**: 20 cities (10 coastal + 10 non-coastal) across 6 continents
- **📊 2.6+ Million Data Points**: Hourly weather data from 2009–2023 (15 years)
- **🔬 4 Separate ML Pipelines**: Each extreme weather type has its own dedicated pipeline
- **🎲 Uncertainty Quantification**: Quantile regression, calibration curves, entropy scoring
- **🧠 Explainable AI (XAI)**: SHAP + LIME analysis for every pipeline
- **🖥️ Professional Dashboard**: Streamlit-based web application with live predictions

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────────────┐
│           STREAMLIT DASHBOARD (app.py)              │
│  Live Predictions · SHAP · Historical · Performance │
└─────────────────┬───────────────────────────────────┘
                  │ loads models & data
    ┌─────────────┼─────────────┬─────────────┐
    │             │             │             │
┌───▼───┐   ┌────▼────┐  ┌─────▼────┐  ┌─────▼─────┐
│ Pipe A│   │ Pipe B  │  │ Pipe C   │  │ Pipe D    │
│ Temp  │   │ Rain    │  │ Heatwave │  │ Disaster  │
│ Regr. │   │ Binary  │  │ Binary   │  │ Multi-Cls │
└───┬───┘   └────┬────┘  └─────┬────┘  └─────┬─────┘
    └─────────────┼─────────────┼─────────────┘
                  │
        ┌─────────▼──────────┐
        │  Preprocessed Data │
        │  59 features       │
        │  train/val/test    │
        └─────────┬──────────┘
                  │
        ┌─────────▼──────────┐
        │   Raw Weather Data │
        │   Open-Meteo API   │
        │   20 cities × 15yr │
        └────────────────────┘
```

---

## 📊 Results Summary

| Pipeline | Task | Key Metric | Result |
|----------|------|------------|--------|
| **A** | Temperature Forecasting | RMSE / R² | **0.628°C / 0.9963** |
| **B** | Rainfall Prediction | F1 / AUC | **1.000 / 1.000** |
| **C** | Heatwave Detection | F1 / AUC | **0.980 / 1.000** |
| **D** | Disaster Classification | Weighted F1 | **1.000** |

---

## 🌍 Global Cities Covered

### Coastal (10)
Mumbai (India), Miami (USA), Lagos (Nigeria), Jakarta (Indonesia), Sydney (Australia), Rotterdam (Netherlands), Cape Town (South Africa), Dhaka (Bangladesh), Tokyo (Japan), Karachi (Pakistan)

### Non-Coastal (10)
Delhi (India), Riyadh (Saudi Arabia), Nairobi (Kenya), Chicago (USA), Moscow (Russia), Ulaanbaatar (Mongolia), Phoenix (USA), São Paulo (Brazil), Cairo (Egypt), Madrid (Spain)

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.10 or higher
- pip package manager
- ~2 GB disk space for data

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/[your-username]/extreme-weather-prediction.git
cd extreme-weather-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Collect weather data (takes ~15 minutes)
python data_collection.py

# 4. Preprocess and engineer features (takes ~5 minutes)
python preprocessing_feature_engineering.py

# 5. Train all 4 pipelines (takes ~4-5 hours total)
python pipeline_A_temperature.py
python pipeline_B_rainfall.py
python pipeline_C_heatwave.py
python pipeline_D_disaster.py

# 6. Launch the dashboard
streamlit run app.py
```

### Streamlit Cloud Deployment

1. Push all code + models to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file path to `app.py`
5. Deploy!

---

## 📁 Project Structure

```
extreme-weather-prediction/
│
├── data_collection.py              # File 1: Fetch data from Open-Meteo API
├── preprocessing_feature_engineering.py  # File 2: Feature engineering + splits
├── pipeline_A_temperature.py       # File 3: Temperature forecasting
├── pipeline_B_rainfall.py          # File 4: Rainfall prediction
├── pipeline_C_heatwave.py          # File 5: Heatwave detection
├── pipeline_D_disaster.py          # File 6: Disaster classification
├── app.py                          # File 7: Streamlit dashboard
├── requirements.txt                # File 8: Python dependencies
├── README.md                       # File 9: This file
│
├── data/
│   ├── raw/                        # Raw city CSVs + master CSV
│   ├── processed/                  # train.csv, val.csv, test.csv
│   ├── reports/                    # Metrics JSONs + CSVs
│   │   └── figures/                # 30+ publication plots
│   └── cache/                      # Parquet cache for API calls
│
└── models/                         # Trained .pkl model files
    ├── pipeline_A_temperature_xgb.pkl
    ├── pipeline_A_q10.pkl
    ├── pipeline_A_q90.pkl
    ├── pipeline_B_rainfall_xgb.pkl
    ├── pipeline_C_heatwave_xgb.pkl
    └── pipeline_D_disaster_xgb.pkl
```

---

## 📊 Data Sources

| Source | Description | URL |
|--------|-------------|-----|
| **Open-Meteo** | Historical hourly weather data (free, no API key) | [open-meteo.com](https://open-meteo.com) |
| **ERA5 (ECMWF)** | Reanalysis data backing Open-Meteo | [ecmwf.int](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5) |

### Weather Variables Collected
- `temperature_2m` (°C)
- `relative_humidity_2m` (%)
- `precipitation` (mm)
- `windspeed_10m` (km/h)
- `surface_pressure` (hPa)
- `cloudcover` (%)
- `shortwave_radiation` (W/m²)

---

## 🧠 Methodology

### Feature Engineering (59 features total)
- **Lag features**: 24h, 48h, 72h for temperature, humidity, rain, wind
- **Rolling features**: 24h and 7-day mean + standard deviation
- **Time features**: hour, month, season (hemisphere-adjusted), year, weekend
- **Geographic**: latitude, longitude, coastal binary, city/continent encoding

### Target Variables (Meteorological Rules)
- **Rain**: precipitation > 1 mm
- **Heatwave**: temperature ≥ 35°C (global) or ≥ 40°C (South Asia)
- **Storm**: windspeed ≥ 40 km/h AND cloudcover ≥ 70%
- **Disaster**: Multi-class (Normal/Heatwave/Rain/Storm) with priority ordering

### Train/Validation/Test Split (Time-Series Aware)
| Period | Usage | Rows |
|--------|-------|------|
| 2009–2017 | Training | 1,576,320 |
| 2018 | Validation | 175,200 |
| 2019–2020 | Testing | 350,880 |
| 2021–2023 | Reserved (future simulation) | 525,600 |

---

## 👥 Authors

- **Syed Bilal** — Department of AI & Mathematical Sciences, SMIU Karachi
- **Raiyan Sheikh** — Department of AI & Mathematical Sciences, SMIU Karachi
- **Numra Amjad** — Department of AI & Mathematical Sciences, SMIU Karachi

---

## 📄 License

This project is submitted as a Final Year Project (FYP) at Sindh Madressatul Islam University, Karachi, Pakistan. All rights reserved.

---

## 🙏 Acknowledgements

We acknowledge the Open-Meteo API for providing free access to historical weather data, ECMWF for the ERA5 reanalysis dataset, and the open-source communities behind XGBoost, SHAP, Streamlit, and scikit-learn.
