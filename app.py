#!/usr/bin/env python3
"""
==============================================================================
FILE 7: app.py — Streamlit Dashboard
==============================================================================
Project : AI-Driven Extreme Weather Prediction — A Global Perspective
Author  : [Your Name]

Dashboard Features:
    TAB 1 — Live Predictions (temperature, rain, heatwave, disaster + alerts)
    TAB 2 — SHAP Explainability (per-pipeline analysis)
    TAB 3 — Historical Trends (temperature, heatwave, rainfall charts)
    TAB 4 — Model Performance (metrics, confusion matrices, ROC curves)
    TAB 5 — Data Explorer (interactive data table, stats, correlation)

Run locally:  streamlit run app.py
Deploy:       Push to GitHub → connect to Streamlit Cloud
==============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Extreme Weather Prediction",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS (dark weather theme)
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        padding: 0.5rem 0;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #90A4AE;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.3rem 0;
    }
    .alert-green { background: linear-gradient(135deg, #1b5e20, #2e7d32); padding: 1rem; border-radius: 10px; color: white; text-align: center; font-size: 1.2rem; font-weight: bold; }
    .alert-yellow { background: linear-gradient(135deg, #f57f17, #fbc02d); padding: 1rem; border-radius: 10px; color: black; text-align: center; font-size: 1.2rem; font-weight: bold; }
    .alert-orange { background: linear-gradient(135deg, #e65100, #ff6d00); padding: 1rem; border-radius: 10px; color: white; text-align: center; font-size: 1.2rem; font-weight: bold; }
    .alert-red { background: linear-gradient(135deg, #b71c1c, #d32f2f); padding: 1rem; border-radius: 10px; color: white; text-align: center; font-size: 1.2rem; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PATH DETECTION
# ============================================================
@st.cache_data
def detect_paths():
    """Auto-detect data and model paths."""
    possible_data = [
        Path.cwd() / "data",
        Path.home() / "data",
        Path.home() / "Desktop" / "FYP 2026" / "data",
        Path("data"),
    ]
    possible_models = [
        Path.cwd() / "models",
        Path.home() / "models",
        Path.home() / "Desktop" / "FYP 2026" / "models",
        Path("models"),
    ]

    data_dir = next((p for p in possible_data if p.exists()), None)
    models_dir = next((p for p in possible_models if p.exists()), None)

    return data_dir, models_dir


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_test_data(data_dir):
    """Load test set for predictions and analysis."""
    test_path = data_dir / "processed" / "test.csv"
    if not test_path.exists():
        return None
    df = pd.read_csv(test_path, low_memory=False)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df


@st.cache_data
def load_master_data(data_dir):
    """Load master raw data for historical trends."""
    master_path = data_dir / "raw" / "master_weather_data.csv"
    if not master_path.exists():
        return None
    df = pd.read_csv(master_path, low_memory=False)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df


@st.cache_data
def load_city_metadata(data_dir):
    """Load city metadata."""
    meta_path = data_dir / "raw" / "city_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


@st.cache_resource
def load_models(models_dir):
    """Load all trained models."""
    models = {}
    model_files = {
        "temperature": "pipeline_A_temperature_xgb.pkl",
        "temp_q10": "pipeline_A_q10.pkl",
        "temp_q90": "pipeline_A_q90.pkl",
        "rainfall": "pipeline_B_rainfall_xgb.pkl",
        "heatwave": "pipeline_C_heatwave_xgb.pkl",
        "disaster": "pipeline_D_disaster_xgb.pkl",
    }
    for key, filename in model_files.items():
        path = models_dir / filename
        if path.exists():
            models[key] = joblib.load(path)
    return models


@st.cache_data
def load_metrics(data_dir):
    """Load all pipeline metrics."""
    metrics = {}
    reports_dir = data_dir / "reports"
    for pipeline in ["A", "B", "C", "D"]:
        path = reports_dir / f"pipeline_{pipeline}_metrics.json"
        if path.exists():
            with open(path) as f:
                metrics[pipeline] = json.load(f)
    return metrics


def get_feature_columns(df):
    exclude = {"datetime", "city", "country", "continent", "climate_zone",
               "heatwave_threshold", "target_rain", "target_heatwave",
               "target_storm", "target_disaster", "target_temperature_next"}
    return [c for c in df.columns if c not in exclude]


# ============================================================
# ALERT SYSTEM
# ============================================================
def get_alert_level(temp_pred, rain_prob, heatwave_prob, disaster_class):
    """Determine alert level based on predictions."""
    if disaster_class == 3 or rain_prob > 0.8:
        return "RED", "🚨 SEVERE WEATHER ALERT", "alert-red"
    elif disaster_class == 1 or heatwave_prob > 0.7 or temp_pred > 42:
        return "ORANGE", "⚠️ HEATWAVE WARNING", "alert-orange"
    elif disaster_class == 2 or rain_prob > 0.5:
        return "YELLOW", "⛈️ RAIN ADVISORY", "alert-yellow"
    else:
        return "GREEN", "✅ NORMAL CONDITIONS", "alert-green"


# ============================================================
# MAIN APP
# ============================================================
def main():
    # Header
    st.markdown('<div class="main-header">🌦️ AI Extreme Weather Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">20 Global Cities · 4 ML Pipelines · Real-Time Predictions with Uncertainty & Explainability</div>', unsafe_allow_html=True)

    # Detect paths
    data_dir, models_dir = detect_paths()

    if data_dir is None:
        st.error("❌ Data directory not found. Please ensure the 'data' folder exists.")
        st.stop()

    # Load everything
    test_df = load_test_data(data_dir)
    master_df = load_master_data(data_dir)
    city_meta = load_city_metadata(data_dir)
    models = load_models(models_dir) if models_dir else {}
    metrics = load_metrics(data_dir)

    if test_df is None:
        st.error("❌ Test data not found. Run preprocessing (File 2) first.")
        st.stop()

    # ── SIDEBAR ──
    st.sidebar.image("https://img.icons8.com/clouds/100/000000/partly-cloudy-day.png", width=80)
    st.sidebar.title("🌍 Controls")

    cities = sorted(test_df["city"].unique())
    selected_city = st.sidebar.selectbox("🏙️ Select City", cities, index=cities.index("Karachi") if "Karachi" in cities else 0)

    # City info
    if selected_city in city_meta:
        info = city_meta[selected_city]
        st.sidebar.markdown(f"""
        **Country:** {info.get('country', 'N/A')}
        **Continent:** {info.get('continent', 'N/A')}
        **Climate:** {info.get('climate_zone', 'N/A')}
        **Coastal:** {'Yes 🌊' if info.get('coastal') else 'No 🏔️'}
        """)

    coastal_filter = st.sidebar.radio("🌊 Filter", ["All Cities", "Coastal Only", "Non-Coastal Only"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Model Status**")
    for key, name in [("temperature", "Pipeline A (Temp)"), ("rainfall", "Pipeline B (Rain)"),
                      ("heatwave", "Pipeline C (Heat)"), ("disaster", "Pipeline D (Disaster)")]:
        status = "✅" if key in models else "❌"
        st.sidebar.markdown(f"{status} {name}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("*Built by Syed Bilal, Raiyan Sheikh & Numra Amjad*")
    st.sidebar.markdown("*SMIU, Karachi · 2025*")

    # Filter data by city
    city_df = test_df[test_df["city"] == selected_city].copy()
    features = get_feature_columns(test_df)

    # ── TABS ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 Live Predictions",
        "🧠 SHAP Explainability",
        "📈 Historical Trends",
        "📊 Model Performance",
        "🔍 Data Explorer",
    ])

    # ============================================================
    # TAB 1 — LIVE PREDICTIONS
    # ============================================================
    with tab1:
        st.header(f"🔮 Predictions for {selected_city}")

        if len(city_df) == 0:
            st.warning("No data available for this city.")
        else:
            # Use last available data point for "current" prediction
            latest = city_df.iloc[-1:]
            X_latest = latest[features]

            # Make predictions
            temp_pred = models["temperature"].predict(X_latest)[0] if "temperature" in models else latest["temperature_2m"].values[0]
            temp_q10 = models["temp_q10"].predict(X_latest)[0] if "temp_q10" in models else temp_pred - 2
            temp_q90 = models["temp_q90"].predict(X_latest)[0] if "temp_q90" in models else temp_pred + 2

            rain_prob = models["rainfall"].predict_proba(X_latest)[0][1] if "rainfall" in models else 0.0
            heat_prob = models["heatwave"].predict_proba(X_latest)[0][1] if "heatwave" in models else 0.0

            disaster_class = int(models["disaster"].predict(X_latest)[0]) if "disaster" in models else 0
            disaster_probs = models["disaster"].predict_proba(X_latest)[0] if "disaster" in models else [1, 0, 0, 0]
            disaster_names = {0: "Normal", 1: "Heatwave", 2: "Heavy Rain", 3: "Storm"}
            disaster_confidence = float(max(disaster_probs))

            # Alert system
            alert_level, alert_text, alert_class = get_alert_level(temp_pred, rain_prob, heat_prob, disaster_class)
            st.markdown(f'<div class="{alert_class}">{alert_text} — Alert Level: {alert_level}</div>', unsafe_allow_html=True)
            st.markdown("")

            # Prediction cards
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:0.9rem; opacity:0.8;">🌡️ Temperature</div>
                    <div style="font-size:2rem; font-weight:bold;">{temp_pred:.1f}°C</div>
                    <div style="font-size:0.8rem; opacity:0.7;">CI: [{temp_q10:.1f} — {temp_q90:.1f}]°C</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                rain_color = "#4CAF50" if rain_prob < 0.3 else "#FF9800" if rain_prob < 0.7 else "#F44336"
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:0.9rem; opacity:0.8;">🌧️ Rainfall Probability</div>
                    <div style="font-size:2rem; font-weight:bold; color:{rain_color};">{rain_prob*100:.0f}%</div>
                    <div style="font-size:0.8rem; opacity:0.7;">{'Heavy rain likely' if rain_prob > 0.5 else 'Low risk'}</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                heat_color = "#4CAF50" if heat_prob < 0.3 else "#FF9800" if heat_prob < 0.7 else "#F44336"
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:0.9rem; opacity:0.8;">☀️ Heatwave Risk</div>
                    <div style="font-size:2rem; font-weight:bold; color:{heat_color};">{heat_prob*100:.0f}%</div>
                    <div style="font-size:0.8rem; opacity:0.7;">{'⚠️ High risk!' if heat_prob > 0.5 else 'Low risk'}</div>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                disaster_color = {"Normal": "#4CAF50", "Heatwave": "#FF9800", "Heavy Rain": "#2196F3", "Storm": "#F44336"}
                dc_name = disaster_names[disaster_class]
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:0.9rem; opacity:0.8;">🚨 Disaster Classification</div>
                    <div style="font-size:1.5rem; font-weight:bold; color:{disaster_color.get(dc_name, 'white')};">{dc_name}</div>
                    <div style="font-size:0.8rem; opacity:0.7;">Confidence: {disaster_confidence*100:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")

            # Temperature forecast with uncertainty band (last 7 days)
            st.subheader("🌡️ Temperature Forecast with Uncertainty")
            n_show = min(24 * 7, len(city_df))
            recent = city_df.tail(n_show).copy()

            if "temperature" in models and len(recent) > 0:
                X_recent = recent[features]
                recent["pred"] = models["temperature"].predict(X_recent)
                recent["q10"] = models["temp_q10"].predict(X_recent) if "temp_q10" in models else recent["pred"] - 2
                recent["q90"] = models["temp_q90"].predict(X_recent) if "temp_q90" in models else recent["pred"] + 2

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=recent["datetime"], y=recent["temperature_2m"],
                                         mode="lines", name="Actual", line=dict(color="#FF5722", width=2)))
                fig.add_trace(go.Scatter(x=recent["datetime"], y=recent["pred"],
                                         mode="lines", name="Predicted", line=dict(color="#2196F3", width=2)))
                fig.add_trace(go.Scatter(
                    x=pd.concat([recent["datetime"], recent["datetime"][::-1]]),
                    y=pd.concat([recent["q90"], recent["q10"][::-1]]),
                    fill="toself", fillcolor="rgba(33,150,243,0.15)",
                    line=dict(width=0), name="80% Confidence Interval",
                ))
                fig.update_layout(
                    title=f"Temperature Forecast — {selected_city} (Last 7 Days)",
                    xaxis_title="Date", yaxis_title="Temperature (°C)",
                    template="plotly_dark", height=450,
                )
                st.plotly_chart(fig, use_container_width=True)

            # Disaster probability breakdown
            st.subheader("🚨 Disaster Probability Breakdown")
            prob_df = pd.DataFrame({
                "Category": list(disaster_names.values()),
                "Probability": [float(p) for p in disaster_probs],
            })
            fig_prob = px.bar(prob_df, x="Category", y="Probability",
                              color="Category",
                              color_discrete_map={"Normal": "#4CAF50", "Heatwave": "#FF9800",
                                                  "Heavy Rain": "#2196F3", "Storm": "#F44336"},
                              template="plotly_dark", title="Disaster Class Probabilities")
            fig_prob.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_prob, use_container_width=True)

    # ============================================================
    # TAB 2 — SHAP EXPLAINABILITY
    # ============================================================
    with tab2:
        st.header("🧠 SHAP Explainability")

        pipeline_choice = st.selectbox("Select Pipeline", [
            "Pipeline A — Temperature",
            "Pipeline B — Rainfall",
            "Pipeline C — Heatwave",
            "Pipeline D — Disaster",
        ])

        pipeline_map = {"Pipeline A — Temperature": "A", "Pipeline B — Rainfall": "B",
                        "Pipeline C — Heatwave": "C", "Pipeline D — Disaster": "D"}
        pl = pipeline_map[pipeline_choice]

        figures_dir = data_dir / "reports" / "figures"

        col1, col2 = st.columns(2)

        # SHAP Beeswarm
        beeswarm = figures_dir / f"pipeline_{pl}_shap_beeswarm.png"
        if beeswarm.exists():
            with col1:
                st.image(str(beeswarm), caption=f"SHAP Beeswarm — Pipeline {pl}", use_container_width=True)
        else:
            with col1:
                st.info(f"Beeswarm plot not found for Pipeline {pl}")

        # SHAP Bar
        bar_file = figures_dir / f"pipeline_{pl}_shap_bar.png"
        if not bar_file.exists():
            bar_file = figures_dir / f"pipeline_{pl}_shap_bar_overall.png"
        if bar_file.exists():
            with col2:
                st.image(str(bar_file), caption=f"Feature Importance — Pipeline {pl}", use_container_width=True)

        # Waterfall
        waterfall = figures_dir / f"pipeline_{pl}_shap_waterfall.png"
        if waterfall.exists():
            st.image(str(waterfall), caption=f"SHAP Waterfall (Single Prediction) — Pipeline {pl}", use_container_width=True)

        # Top features table
        top_path = data_dir / "reports" / f"pipeline_{pl}_top_features.csv"
        if top_path.exists():
            st.subheader(f"📋 Top Features — Pipeline {pl}")
            top_df = pd.read_csv(top_path)
            st.dataframe(top_df, use_container_width=True, hide_index=True)

        # LIME (Pipeline B only)
        if pl == "B":
            st.subheader("🧪 LIME Explanations")
            for i in range(1, 4):
                lime_path = figures_dir / f"pipeline_B_lime_instance_{i}.png"
                if lime_path.exists():
                    st.image(str(lime_path), caption=f"LIME — Instance {i}", use_container_width=True)

    # ============================================================
    # TAB 3 — HISTORICAL TRENDS
    # ============================================================
    with tab3:
        st.header("📈 Historical Trends")

        data_source = master_df if master_df is not None else test_df
        city_hist = data_source[data_source["city"] == selected_city].copy()

        if len(city_hist) == 0:
            st.warning("No historical data for this city.")
        else:
            city_hist["year"] = city_hist["datetime"].dt.year
            city_hist["month"] = city_hist["datetime"].dt.month

            # Temperature trend
            st.subheader("🌡️ Temperature Trend (2009–2023)")
            yearly_temp = city_hist.groupby("year")["temperature_2m"].agg(["mean", "min", "max"]).reset_index()
            fig_temp = go.Figure()
            fig_temp.add_trace(go.Scatter(x=yearly_temp["year"], y=yearly_temp["mean"],
                                          mode="lines+markers", name="Mean", line=dict(color="#FF5722", width=3)))
            fig_temp.add_trace(go.Scatter(x=yearly_temp["year"], y=yearly_temp["max"],
                                          mode="lines", name="Max", line=dict(color="#F44336", dash="dot")))
            fig_temp.add_trace(go.Scatter(x=yearly_temp["year"], y=yearly_temp["min"],
                                          mode="lines", name="Min", line=dict(color="#2196F3", dash="dot")))
            fig_temp.update_layout(title=f"Temperature Trend — {selected_city}",
                                   xaxis_title="Year", yaxis_title="Temperature (°C)",
                                   template="plotly_dark", height=400)
            st.plotly_chart(fig_temp, use_container_width=True)

            # Heatwave frequency
            st.subheader("☀️ Heatwave Hours per Year")
            threshold = 40.0 if selected_city in ["Karachi", "Delhi", "Mumbai", "Dhaka"] else 35.0
            city_hist["is_heatwave"] = (city_hist["temperature_2m"] >= threshold).astype(int)
            hw_yearly = city_hist.groupby("year")["is_heatwave"].sum().reset_index()
            hw_yearly.columns = ["year", "heatwave_hours"]

            fig_hw = px.bar(hw_yearly, x="year", y="heatwave_hours",
                            color="heatwave_hours", color_continuous_scale="YlOrRd",
                            template="plotly_dark",
                            title=f"Heatwave Hours per Year — {selected_city} (threshold: {threshold}°C)")
            fig_hw.update_layout(height=400)
            st.plotly_chart(fig_hw, use_container_width=True)

            # Rainfall distribution
            st.subheader("🌧️ Monthly Rainfall Distribution")
            monthly_rain = city_hist.groupby("month")["precipitation"].sum().reset_index()
            month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            monthly_rain["month_name"] = monthly_rain["month"].apply(lambda x: month_names[x-1])

            fig_rain = px.bar(monthly_rain, x="month_name", y="precipitation",
                              color="precipitation", color_continuous_scale="Blues",
                              template="plotly_dark",
                              title=f"Total Precipitation by Month — {selected_city}")
            fig_rain.update_layout(height=400, xaxis_title="Month", yaxis_title="Total Precipitation (mm)")
            st.plotly_chart(fig_rain, use_container_width=True)

            # Wind speed extremes
            st.subheader("💨 Wind Speed Distribution")
            fig_wind = px.histogram(city_hist, x="windspeed_10m", nbins=50,
                                     color_discrete_sequence=["#26A69A"],
                                     template="plotly_dark",
                                     title=f"Wind Speed Distribution — {selected_city}")
            fig_wind.add_vline(x=40, line_dash="dash", line_color="red",
                               annotation_text="Storm threshold (40 km/h)")
            fig_wind.update_layout(height=350, xaxis_title="Wind Speed (km/h)", yaxis_title="Count")
            st.plotly_chart(fig_wind, use_container_width=True)

    # ============================================================
    # TAB 4 — MODEL PERFORMANCE
    # ============================================================
    with tab4:
        st.header("📊 Model Performance Dashboard")

        if not metrics:
            st.warning("No metrics found. Run all 4 pipelines first.")
        else:
            # Summary table
            st.subheader("📋 All Pipelines — Performance Summary")
            summary_rows = []
            if "A" in metrics:
                m = metrics["A"].get("test_metrics", {})
                summary_rows.append({
                    "Pipeline": "A — Temperature",
                    "Task": "Regression",
                    "Primary Metric": f"RMSE = {m.get('rmse', 'N/A')}°C",
                    "R²": m.get("r2", "N/A"),
                    "Secondary": f"MAE = {m.get('mae', 'N/A')}°C",
                })
            if "B" in metrics:
                m = metrics["B"].get("test_metrics", {})
                summary_rows.append({
                    "Pipeline": "B — Rainfall",
                    "Task": "Binary Classification",
                    "Primary Metric": f"F1 = {m.get('f1_score', 'N/A')}",
                    "R²": f"AUC = {m.get('roc_auc', 'N/A')}",
                    "Secondary": f"Brier = {m.get('brier_score', 'N/A')}",
                })
            if "C" in metrics:
                m = metrics["C"].get("test_metrics", {})
                summary_rows.append({
                    "Pipeline": "C — Heatwave",
                    "Task": "Binary Classification",
                    "Primary Metric": f"F1 = {m.get('f1_score', 'N/A')}",
                    "R²": f"AUC = {m.get('roc_auc', 'N/A')}",
                    "Secondary": f"Brier = {m.get('brier_score', 'N/A')}",
                })
            if "D" in metrics:
                m = metrics["D"].get("test_metrics", {})
                summary_rows.append({
                    "Pipeline": "D — Disaster",
                    "Task": "Multi-Class",
                    "Primary Metric": f"wF1 = {m.get('weighted_f1', 'N/A')}",
                    "R²": f"mF1 = {m.get('macro_f1', 'N/A')}",
                    "Secondary": f"Acc = {m.get('accuracy', 'N/A')}",
                })

            if summary_rows:
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

            # Show figures
            st.subheader("📊 Evaluation Figures")
            figures_dir = data_dir / "reports" / "figures"

            fig_col1, fig_col2 = st.columns(2)

            # Confusion matrices
            for pl, col in [("B", fig_col1), ("C", fig_col2)]:
                cm_path = figures_dir / f"pipeline_{pl}_confusion_matrix.png"
                if cm_path.exists():
                    with col:
                        st.image(str(cm_path), caption=f"Confusion Matrix — Pipeline {pl}", use_container_width=True)

            # ROC curves
            fig_col3, fig_col4 = st.columns(2)
            for pl, col in [("C", fig_col3), ("D", fig_col4)]:
                roc_path = figures_dir / f"pipeline_{pl}_roc_curve.png"
                if not roc_path.exists():
                    roc_path = figures_dir / f"pipeline_{pl}_roc_per_class.png"
                if roc_path.exists():
                    with col:
                        st.image(str(roc_path), caption=f"ROC Curve — Pipeline {pl}", use_container_width=True)

            # Calibration curves
            fig_col5, fig_col6 = st.columns(2)
            for pl, col in [("B", fig_col5), ("C", fig_col6)]:
                cal_path = figures_dir / f"pipeline_{pl}_calibration_curve.png"
                if cal_path.exists():
                    with col:
                        st.image(str(cal_path), caption=f"Calibration — Pipeline {pl}", use_container_width=True)

            # Pipeline A specific
            for fname, caption in [
                ("pipeline_A_predictions_vs_actual.png", "Predictions vs Actual (Temperature)"),
                ("pipeline_A_residuals.png", "Residual Analysis (Temperature)"),
                ("pipeline_A_baseline_comparison.png", "Model vs Baseline (Temperature)"),
            ]:
                fpath = figures_dir / fname
                if fpath.exists():
                    st.image(str(fpath), caption=caption, use_container_width=True)

            # Uncertainty plots
            for fname, caption in [
                ("pipeline_A_uncertainty_Karachi.png", "Uncertainty — Karachi"),
                ("pipeline_A_uncertainty_Mumbai.png", "Uncertainty — Mumbai"),
                ("pipeline_D_entropy_uncertainty.png", "Entropy Uncertainty — Disaster"),
            ]:
                fpath = figures_dir / fname
                if fpath.exists():
                    st.image(str(fpath), caption=caption, use_container_width=True)

    # ============================================================
    # TAB 5 — DATA EXPLORER
    # ============================================================
    with tab5:
        st.header("🔍 Data Explorer")

        data_source_choice = st.radio("Data Source", ["Test Set (2019-2020)", "Full Dataset (2009-2023)"])
        explore_df = test_df if "Test" in data_source_choice else (master_df if master_df is not None else test_df)

        # City filter
        explore_city = st.selectbox("Select City for Exploration", ["All Cities"] + sorted(explore_df["city"].unique()))
        if explore_city != "All Cities":
            explore_df = explore_df[explore_df["city"] == explore_city]

        st.subheader(f"📋 Data Preview ({len(explore_df):,} rows)")
        st.dataframe(explore_df.head(500), use_container_width=True, height=400)

        # Descriptive statistics
        st.subheader("📊 Descriptive Statistics")
        numeric_cols = ["temperature_2m", "relative_humidity_2m", "precipitation",
                        "windspeed_10m", "surface_pressure", "cloudcover", "shortwave_radiation"]
        available_cols = [c for c in numeric_cols if c in explore_df.columns]
        if available_cols:
            st.dataframe(explore_df[available_cols].describe().round(2), use_container_width=True)

        # Correlation heatmap
        st.subheader("🔥 Correlation Heatmap")
        if available_cols and len(explore_df) > 0:
            corr = explore_df[available_cols].corr()
            fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                 title="Feature Correlation Matrix", template="plotly_dark",
                                 aspect="auto")
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)

        # Download option
        st.subheader("📥 Download Data")
        csv = explore_df.head(10000).to_csv(index=False)
        st.download_button("Download CSV (first 10,000 rows)", csv,
                           f"weather_data_{explore_city}.csv", "text/csv")


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    main()
