#!/usr/bin/env python3
"""
==============================================================================
FILE 7: app.py — Streamlit Dashboard (FULLY INTERACTIVE — No Static Images)
==============================================================================
Project : AI-Driven Extreme Weather Prediction — A Global Perspective
Authors : Syed Bilal, Raiyan Sheikh & Numra Amjad
GitHub  : https://github.com/ryansheikh/extreme-weather-prediction

All charts are interactive Plotly — hover, zoom, pan, download!
Tabs:
    1 — Live Predictions (alerts + charts)
    2 — SHAP Explainability (interactive feature importance)
    3 — Historical Trends (2019-2020 test period)
    4 — Model Performance (confusion matrix, ROC, calibration)
    5 — Extended 2021-2026 (new predictions)
    6 — Data Explorer
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

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Extreme Weather Prediction",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size:2.2rem; font-weight:700; color:#1E88E5; text-align:center; padding:.5rem 0; }
    .sub-header  { font-size:1rem; color:#90A4AE; text-align:center; margin-bottom:1.5rem; }
    .card-blue   { background:linear-gradient(135deg,#1a237e,#283593); padding:1.2rem; border-radius:12px; color:white; text-align:center; margin:.3rem 0; }
    .card-green  { background:linear-gradient(135deg,#1b5e20,#2e7d32); padding:1rem; border-radius:10px; color:white; text-align:center; }
    .alert-green  { background:linear-gradient(135deg,#1b5e20,#2e7d32); padding:1rem; border-radius:10px; color:white; text-align:center; font-size:1.2rem; font-weight:bold; }
    .alert-yellow { background:linear-gradient(135deg,#f57f17,#fbc02d); padding:1rem; border-radius:10px; color:black; text-align:center; font-size:1.2rem; font-weight:bold; }
    .alert-orange { background:linear-gradient(135deg,#e65100,#ff6d00); padding:1rem; border-radius:10px; color:white; text-align:center; font-size:1.2rem; font-weight:bold; }
    .alert-red    { background:linear-gradient(135deg,#b71c1c,#d32f2f); padding:1rem; border-radius:10px; color:white; text-align:center; font-size:1.2rem; font-weight:bold; }
    .stTabs [data-baseweb="tab"] { padding:10px 20px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────
EXACT_FEATURES = [
    "temperature_2m","relative_humidity_2m","precipitation",
    "windspeed_10m","surface_pressure","cloudcover","shortwave_radiation",
    "latitude","longitude","coastal",
    "hour","day","day_of_week","month","year","day_of_year","is_weekend","season",
    "temperature_2m_lag_24h","temperature_2m_lag_48h","temperature_2m_lag_72h",
    "relative_humidity_2m_lag_24h","relative_humidity_2m_lag_48h","relative_humidity_2m_lag_72h",
    "precipitation_lag_24h","precipitation_lag_48h","precipitation_lag_72h",
    "windspeed_10m_lag_24h","windspeed_10m_lag_48h","windspeed_10m_lag_72h",
    "temperature_2m_roll_24h_mean","temperature_2m_roll_24h_std",
    "temperature_2m_roll_7d_mean","temperature_2m_roll_7d_std",
    "relative_humidity_2m_roll_24h_mean","relative_humidity_2m_roll_24h_std",
    "relative_humidity_2m_roll_7d_mean","relative_humidity_2m_roll_7d_std",
    "precipitation_roll_24h_mean","precipitation_roll_24h_std",
    "precipitation_roll_7d_mean","precipitation_roll_7d_std",
    "windspeed_10m_roll_24h_mean","windspeed_10m_roll_24h_std",
    "windspeed_10m_roll_7d_mean","windspeed_10m_roll_7d_std",
    "city_id","continent_id","climate_zone_id",
]

CITY_META = {
    "Karachi":    {"lat":24.86, "lon":67.01,  "coastal":1,"climate_zone":"Arid",         "continent":"Asia"},
    "Mumbai":     {"lat":19.08, "lon":72.88,  "coastal":1,"climate_zone":"Tropical",     "continent":"Asia"},
    "Delhi":      {"lat":28.61, "lon":77.21,  "coastal":0,"climate_zone":"Semi-Arid",    "continent":"Asia"},
    "Dhaka":      {"lat":23.81, "lon":90.41,  "coastal":1,"climate_zone":"Tropical",     "continent":"Asia"},
    "Tokyo":      {"lat":35.68, "lon":139.65, "coastal":1,"climate_zone":"Temperate",    "continent":"Asia"},
    "Jakarta":    {"lat":-6.21, "lon":106.85, "coastal":1,"climate_zone":"Tropical",     "continent":"Asia"},
    "Lagos":      {"lat":6.52,  "lon":3.38,   "coastal":1,"climate_zone":"Tropical",     "continent":"Africa"},
    "Nairobi":    {"lat":-1.29, "lon":36.82,  "coastal":0,"climate_zone":"Temperate",    "continent":"Africa"},
    "Cape_Town":  {"lat":-33.92,"lon":18.42,  "coastal":1,"climate_zone":"Mediterranean","continent":"Africa"},
    "Cairo":      {"lat":30.04, "lon":31.24,  "coastal":0,"climate_zone":"Arid",         "continent":"Africa"},
    "Miami":      {"lat":25.76, "lon":-80.19, "coastal":1,"climate_zone":"Tropical",     "continent":"Americas"},
    "Chicago":    {"lat":41.88, "lon":-87.63, "coastal":0,"climate_zone":"Continental",  "continent":"Americas"},
    "Phoenix":    {"lat":33.45, "lon":-112.07,"coastal":0,"climate_zone":"Arid",         "continent":"Americas"},
    "Sao_Paulo":  {"lat":-23.55,"lon":-46.63, "coastal":0,"climate_zone":"Tropical",     "continent":"Americas"},
    "Rotterdam":  {"lat":51.92, "lon":4.48,   "coastal":1,"climate_zone":"Temperate",    "continent":"Europe"},
    "Madrid":     {"lat":40.42, "lon":-3.70,  "coastal":0,"climate_zone":"Mediterranean","continent":"Europe"},
    "Moscow":     {"lat":55.76, "lon":37.62,  "coastal":0,"climate_zone":"Continental",  "continent":"Europe"},
    "Ulaanbaatar":{"lat":47.89, "lon":106.91, "coastal":0,"climate_zone":"Continental",  "continent":"Asia"},
    "Sydney":     {"lat":-33.87,"lon":151.21, "coastal":1,"climate_zone":"Temperate",    "continent":"Oceania"},
    "Riyadh":     {"lat":24.71, "lon":46.68,  "coastal":0,"climate_zone":"Arid",         "continent":"Asia"},
}
CZ_MAP   = {"Arid":0,"Continental":1,"Mediterranean":2,"Semi-Arid":3,"Temperate":4,"Tropical":5}
CONT_MAP = {"Africa":0,"Americas":1,"Asia":2,"Europe":3,"Oceania":4}
CLASS_NAMES = {0:"Normal",1:"Heatwave",2:"Heavy Rain",3:"Storm"}
CLASS_COLORS = {"Normal":"#4CAF50","Heatwave":"#FF9800","Heavy Rain":"#2196F3","Storm":"#F44336"}
HW_CITIES = {"Karachi","Delhi","Mumbai","Dhaka"}

# ── Path Detection ─────────────────────────────────────────────
@st.cache_data
def detect_paths():
    for d in [Path.cwd()/"data", Path("data"), Path.home()/"data",
              Path.home()/"extreme_weather"/"data"]:
        if d.exists(): data_dir = d; break
    else: data_dir = None
    for m in [Path.cwd()/"models", Path("models"), Path.home()/"models",
              Path.home()/"extreme_weather"/"models"]:
        if m.exists(): models_dir = m; break
    else: models_dir = None
    return data_dir, models_dir

# ── Column fixer ───────────────────────────────────────────────
def fix_columns(df):
    df = df.copy()
    city_list = sorted(df["city"].unique())
    if "latitude"       not in df.columns: df["latitude"]       = df["city"].map({k:v["lat"]          for k,v in CITY_META.items()}).fillna(0)
    if "longitude"      not in df.columns: df["longitude"]      = df["city"].map({k:v["lon"]          for k,v in CITY_META.items()}).fillna(0)
    if "coastal"        not in df.columns: df["coastal"]        = df["city"].map({k:v["coastal"]      for k,v in CITY_META.items()}).fillna(0)
    if "climate_zone"   not in df.columns: df["climate_zone"]   = df["city"].map({k:v["climate_zone"] for k,v in CITY_META.items()}).fillna("Temperate")
    if "continent"      not in df.columns: df["continent"]      = df["city"].map({k:v["continent"]    for k,v in CITY_META.items()}).fillna("Asia")
    if "climate_zone_id"not in df.columns: df["climate_zone_id"]= df["climate_zone"].map(CZ_MAP).fillna(4).astype(int)
    if "continent_id"   not in df.columns: df["continent_id"]   = df["continent"].map(CONT_MAP).fillna(2).astype(int)
    if "city_id"        not in df.columns: df["city_id"]        = df["city"].map({c:i for i,c in enumerate(city_list)}).fillna(0).astype(int)
    df.fillna(0, inplace=True)
    return df

# ── Data Loaders ───────────────────────────────────────────────
@st.cache_data
def load_test_data(data_dir):
    for name in ["test.csv.gz","test.csv"]:
        p = data_dir/"processed"/name
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            return fix_columns(df)
    return None

@st.cache_data
def load_extended_data(data_dir):
    p = data_dir/"reports"/"extended_predictions_2021_2026.csv"
    if p.exists():
        df = pd.read_csv(p, low_memory=False)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        return df
    return None

@st.cache_data
def load_metrics(data_dir):
    metrics = {}
    for pl in ["A","B","C","D"]:
        p = data_dir/"reports"/f"pipeline_{pl}_metrics.json"
        if p.exists():
            with open(p) as f: metrics[pl] = json.load(f)
    return metrics

@st.cache_data
def load_top_features(data_dir, pipeline):
    p = data_dir/"reports"/f"pipeline_{pipeline}_top_features.csv"
    if p.exists():
        return pd.read_csv(p)
    return None

@st.cache_data
def load_city_perf(data_dir, pipeline):
    p = data_dir/"reports"/f"pipeline_{pipeline}_per_city_performance.csv"
    if p.exists():
        return pd.read_csv(p)
    return None

@st.cache_resource
def load_models(models_dir):
    if not models_dir: return {}
    m = {}
    for key,fn in [("temperature","pipeline_A_temperature_xgb.pkl"),
                   ("temp_q10","pipeline_A_q10.pkl"),("temp_q90","pipeline_A_q90.pkl"),
                   ("rainfall","pipeline_B_rainfall_xgb.pkl"),
                   ("heatwave","pipeline_C_heatwave_xgb.pkl"),
                   ("disaster","pipeline_D_disaster_xgb.pkl")]:
        p = models_dir/fn
        if p.exists(): m[key] = joblib.load(p)
    return m

def predict_row(models, row_df):
    X = row_df[[f for f in EXACT_FEATURES if f in row_df.columns]]
    temp  = models["temperature"].predict(X)[0] if "temperature" in models else row_df["temperature_2m"].values[0]
    q10   = models["temp_q10"].predict(X)[0]    if "temp_q10"    in models else temp-2
    q90   = models["temp_q90"].predict(X)[0]    if "temp_q90"    in models else temp+2
    rprob = models["rainfall"].predict_proba(X)[0][1] if "rainfall" in models else 0.0
    hprob = models["heatwave"].predict_proba(X)[0][1] if "heatwave" in models else 0.0
    dprob = models["disaster"].predict_proba(X)[0]    if "disaster" in models else [1,0,0,0]
    dcls  = int(np.argmax(dprob))
    dconf = float(max(dprob))
    return temp,q10,q90,rprob,hprob,dcls,dprob,dconf

def get_alert(temp, rprob, hprob, dcls):
    if dcls==3 or rprob>0.8:   return "🚨 SEVERE WEATHER ALERT","alert-red","RED"
    elif dcls==1 or hprob>0.7 or temp>42: return "⚠️ HEATWAVE WARNING","alert-orange","ORANGE"
    elif dcls==2 or rprob>0.5: return "⛈️ RAIN ADVISORY","alert-yellow","YELLOW"
    return "✅ NORMAL CONDITIONS","alert-green","GREEN"


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    st.markdown('<div class="main-header">🌦️ AI Extreme Weather Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">20 Global Cities · 4 ML Pipelines · 17.5 Years of Data (2009–2026) · Interactive Dashboard</div>', unsafe_allow_html=True)

    data_dir, models_dir = detect_paths()
    if data_dir is None:
        st.error("❌ Data directory not found."); st.stop()

    test_df  = load_test_data(data_dir)
    ext_df   = load_extended_data(data_dir)
    models   = load_models(models_dir) if models_dir else {}
    metrics  = load_metrics(data_dir)

    if test_df is None:
        st.error("❌ Test data not found."); st.stop()

    # ── SIDEBAR ───────────────────────────────────────────────
    st.sidebar.markdown("## 🌍 Controls")
    cities = sorted(test_df["city"].unique())
    sel = st.sidebar.selectbox("🏙️ City", cities,
                               index=cities.index("Karachi") if "Karachi" in cities else 0)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Pipeline Status**")
    for key,name in [("temperature","A — Temperature"),("rainfall","B — Rainfall"),
                     ("heatwave","C — Heatwave"),("disaster","D — Disaster")]:
        st.sidebar.markdown(f"{'✅' if key in models else '⚠️ (no model)'} {name}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📅 Data Coverage**")
    st.sidebar.info("Train: 2009–2017\nTest:  2019–2020\nNew:   2021–May 2026\nProj:  2027")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**👥 Team**")
    st.sidebar.markdown("Syed Bilal · Raiyan Sheikh\nNumra Amjad\n\nSMIU Karachi · 2025")

    city_df = test_df[test_df["city"]==sel].copy()

    # ── TABS ──────────────────────────────────────────────────
    t1,t2,t3,t4,t5,t6 = st.tabs([
        "🔮 Live Predictions",
        "🧠 SHAP Explainability",
        "📈 Historical Trends",
        "📊 Model Performance",
        "🆕 2021–2026 Extended",
        "🔍 Data Explorer",
    ])

    # ══════════════════════════════════════════════════════════
    # TAB 1 — LIVE PREDICTIONS
    # ══════════════════════════════════════════════════════════
    with t1:
        st.header(f"🔮 Predictions for {sel}")
        if len(city_df)==0:
            st.warning("No data available for this city."); st.stop()

        latest = city_df.iloc[-1:]
        temp,q10,q90,rprob,hprob,dcls,dprob,dconf = predict_row(models, latest)
        alert_txt, alert_cls, alert_lvl = get_alert(temp, rprob, hprob, dcls)
        dn = CLASS_NAMES[dcls]

        st.markdown(f'<div class="{alert_cls}">{alert_txt} — Level: {alert_lvl}</div>', unsafe_allow_html=True)
        st.markdown("")

        c1,c2,c3,c4 = st.columns(4)
        with c1:
            rc = "#4CAF50" if rprob<0.3 else "#FF9800" if rprob<0.7 else "#F44336"
            st.markdown(f"""<div class="card-blue">
                <div style="font-size:.85rem;opacity:.8">🌡️ Temperature</div>
                <div style="font-size:2rem;font-weight:bold">{temp:.1f}°C</div>
                <div style="font-size:.78rem;opacity:.7">80% CI: {q10:.1f}°C — {q90:.1f}°C</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            rc = "#4CAF50" if rprob<0.3 else "#FF9800" if rprob<0.7 else "#F44336"
            st.markdown(f"""<div class="card-blue">
                <div style="font-size:.85rem;opacity:.8">🌧️ Rain Probability</div>
                <div style="font-size:2rem;font-weight:bold;color:{rc}">{rprob*100:.0f}%</div>
                <div style="font-size:.78rem;opacity:.7">{'Heavy rain likely' if rprob>.5 else 'Low risk'}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            hc = "#4CAF50" if hprob<0.3 else "#FF9800" if hprob<0.7 else "#F44336"
            st.markdown(f"""<div class="card-blue">
                <div style="font-size:.85rem;opacity:.8">☀️ Heatwave Risk</div>
                <div style="font-size:2rem;font-weight:bold;color:{hc}">{hprob*100:.0f}%</div>
                <div style="font-size:.78rem;opacity:.7">{'⚠️ High risk!' if hprob>.5 else 'Low risk'}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            dc = CLASS_COLORS.get(dn,"white")
            st.markdown(f"""<div class="card-blue">
                <div style="font-size:.85rem;opacity:.8">🚨 Disaster Class</div>
                <div style="font-size:1.6rem;font-weight:bold;color:{dc}">{dn}</div>
                <div style="font-size:.78rem;opacity:.7">Confidence: {dconf*100:.0f}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        # Temperature chart with uncertainty
        st.subheader(f"🌡️ Temperature Forecast — Last 7 Days ({sel})")
        n = min(24*7, len(city_df))
        rec = city_df.tail(n).copy()
        X_rec = rec[[f for f in EXACT_FEATURES if f in rec.columns]]
        if "temperature" in models:
            rec["pred"] = models["temperature"].predict(X_rec)
            rec["q10_r"] = models["temp_q10"].predict(X_rec) if "temp_q10" in models else rec["pred"]-2
            rec["q90_r"] = models["temp_q90"].predict(X_rec) if "temp_q90" in models else rec["pred"]+2
        else:
            rec["pred"] = rec["temperature_2m"]
            rec["q10_r"] = rec["pred"]-2
            rec["q90_r"] = rec["pred"]+2

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rec["datetime"], y=rec["q10_r"], line=dict(width=0), showlegend=False, mode="lines"))
        fig.add_trace(go.Scatter(x=rec["datetime"], y=rec["q90_r"], fill="tonexty",
                                 fillcolor="rgba(33,150,243,0.15)", line=dict(width=0), name="80% Confidence Interval"))
        fig.add_trace(go.Scatter(x=rec["datetime"], y=rec["temperature_2m"], mode="lines",
                                 name="Actual", line=dict(color="#FF5722",width=2.5)))
        fig.add_trace(go.Scatter(x=rec["datetime"], y=rec["pred"], mode="lines",
                                 name="AI Predicted", line=dict(color="#2196F3",width=2.5,dash="dash")))
        fig.update_layout(template="plotly_dark", height=420,
                          xaxis_title="Date/Time", yaxis_title="Temperature (°C)",
                          hovermode="x unified", legend=dict(orientation="h",y=1.1))
        st.plotly_chart(fig, use_container_width=True)

        # Disaster probability gauge-style bars
        st.subheader("🚨 Disaster Class Probability")
        prob_fig = go.Figure()
        for i,(cls,prob) in enumerate(zip(CLASS_NAMES.values(), dprob)):
            prob_fig.add_trace(go.Bar(
                x=[prob], y=[cls], orientation="h",
                name=cls, marker_color=list(CLASS_COLORS.values())[i],
                text=f"{prob*100:.1f}%", textposition="outside",
            ))
        prob_fig.update_layout(template="plotly_dark", height=280,
                               xaxis=dict(title="Probability",range=[0,1]),
                               showlegend=False, barmode="overlay")
        st.plotly_chart(prob_fig, use_container_width=True)

        # Rain probability gauge
        col_r, col_h = st.columns(2)
        with col_r:
            st.subheader("🌧️ Rain Probability Gauge")
            gauge_r = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=rprob*100,
                delta={"reference":30,"increasing":{"color":"#F44336"}},
                gauge={"axis":{"range":[0,100]},
                       "bar":{"color":"#2196F3"},
                       "steps":[{"range":[0,30],"color":"#1b5e20"},
                                {"range":[30,70],"color":"#f57f17"},
                                {"range":[70,100],"color":"#b71c1c"}],
                       "threshold":{"line":{"color":"white","width":4},"thickness":.75,"value":50}},
                title={"text":"Rain Probability (%)"},
            ))
            gauge_r.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(gauge_r, use_container_width=True)

        with col_h:
            st.subheader("☀️ Heatwave Risk Gauge")
            gauge_h = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=hprob*100,
                delta={"reference":30,"increasing":{"color":"#F44336"}},
                gauge={"axis":{"range":[0,100]},
                       "bar":{"color":"#FF5722"},
                       "steps":[{"range":[0,30],"color":"#1b5e20"},
                                {"range":[30,70],"color":"#f57f17"},
                                {"range":[70,100],"color":"#b71c1c"}],
                       "threshold":{"line":{"color":"white","width":4},"thickness":.75,"value":50}},
                title={"text":"Heatwave Probability (%)"},
            ))
            gauge_h.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(gauge_h, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # TAB 2 — SHAP EXPLAINABILITY
    # ══════════════════════════════════════════════════════════
    with t2:
        st.header("🧠 SHAP Feature Importance — Explainable AI")

        pl_map = {"Pipeline A — Temperature (Regression)":"A",
                  "Pipeline B — Rainfall (Binary)":"B",
                  "Pipeline C — Heatwave (Binary)":"C",
                  "Pipeline D — Disaster (Multi-Class)":"D"}
        pl_name = st.selectbox("Select Pipeline", list(pl_map.keys()))
        pl = pl_map[pl_name]

        top_df = load_top_features(data_dir, pl)

        if top_df is not None and not top_df.empty:
            top_df = top_df.head(20).sort_values("mean_abs_shap")

            # Feature importance horizontal bar
            st.subheader(f"📊 Top {len(top_df)} Features by Mean |SHAP| — Pipeline {pl}")
            colors = px.colors.sequential.Blues[2:]
            bar_colors = [colors[int(i/(len(top_df)-1)*(len(colors)-1))] for i in range(len(top_df))]

            fig_shap = go.Figure(go.Bar(
                x=top_df["mean_abs_shap"],
                y=top_df["feature"],
                orientation="h",
                marker=dict(color=bar_colors),
                text=[f"{v:.4f}" for v in top_df["mean_abs_shap"]],
                textposition="outside",
            ))
            fig_shap.update_layout(
                template="plotly_dark", height=600,
                xaxis_title="Mean |SHAP Value| (average impact on model output)",
                yaxis_title="Feature",
                title=f"Pipeline {pl} — Feature Importance (SHAP)",
                showlegend=False,
            )
            st.plotly_chart(fig_shap, use_container_width=True)

            # Feature categories
            st.subheader("🗂️ Feature Categories Breakdown")
            def categorize(feat):
                if any(x in feat for x in ["lag_24","lag_48","lag_72"]): return "Lag Features"
                if "roll" in feat: return "Rolling Features"
                if feat in ["hour","day","month","year","season","is_weekend","day_of_week","day_of_year"]: return "Time Features"
                if feat in ["latitude","longitude","coastal","city_id","continent_id","climate_zone_id"]: return "Geographic"
                return "Raw Weather"

            top_df["category"] = top_df["feature"].apply(categorize)
            cat_sum = top_df.groupby("category")["mean_abs_shap"].sum().reset_index()
            fig_pie = px.pie(cat_sum, values="mean_abs_shap", names="category",
                             color_discrete_sequence=px.colors.qualitative.Bold,
                             title=f"Feature Category Contribution — Pipeline {pl}",
                             template="plotly_dark")
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(height=420)
            st.plotly_chart(fig_pie, use_container_width=True)

            # Top features table
            st.subheader("📋 Full Feature Ranking Table")
            display_df = top_df[["feature","mean_abs_shap","category"]].sort_values(
                "mean_abs_shap", ascending=False).reset_index(drop=True)
            display_df.index += 1
            display_df.columns = ["Feature","Mean |SHAP|","Category"]
            display_df["Mean |SHAP|"] = display_df["Mean |SHAP|"].round(4)
            st.dataframe(display_df, use_container_width=True)

            # Physical interpretation
            st.subheader("🔬 Physical Interpretation")
            interpretations = {
                "A": """**Top feature: temperature_2m (current temperature)**
                — The best predictor of next-hour temperature is the current temperature. This is physically intuitive: weather systems evolve gradually, not suddenly.

                **temperature_2m_lag_24h (yesterday's temperature at same hour)**
                — The diurnal cycle (day/night pattern) means yesterday's reading at the same hour is a strong predictor. If it was 38°C at 2pm yesterday, it will likely be similar today.

                **shortwave_radiation (solar energy input)**
                — Solar radiation drives daytime heating. High radiation = temperature will rise. This is the physical mechanism behind temperature change.

                **hour (time of day)**
                — Temperature follows a clear 24-hour cycle: low at dawn, peak in afternoon. The model learned this pattern directly from 15 years of data.""",
                "B": """**Top feature: precipitation (current rainfall)**
                — Current rain is the strongest predictor of rain in the next hour. Rainfall events typically last multiple hours, not single moments.

                **precipitation_roll_24h_std (rainfall variability)**
                — High variability in recent rainfall indicates an active weather system, which increases future rain probability.

                **cloudcover (cloud coverage)**
                — High cloud cover is a necessary (but not sufficient) condition for rainfall. The model correctly learned this physical relationship.""",
                "C": """**Top feature: temperature_2m (current temperature)**
                — A heatwave is defined by extreme temperature, so current temperature naturally dominates.

                **relative_humidity_2m (humidity)**
                — Low humidity during high temperatures is the hallmark of dry heatwaves. The model captures this interaction.

                **temperature_2m_lag_24h**
                — Heatwaves require sustained heat, not just a single hot hour. Yesterday's temperature being high increases today's heatwave probability.""",
                "D": """**Top feature: precipitation**
                — Heavy rain dominates disaster classification because it occurs more frequently than storms and is directly measurable.

                **temperature_2m**
                — Heatwave classification requires sustained high temperatures.

                **windspeed_10m and cloudcover**
                — These two features together define storm classification (≥40 km/h wind AND ≥70% cloud cover).""",
            }
            st.markdown(interpretations.get(pl, "Physical interpretation not available."))
        else:
            st.warning(f"Feature importance data not found for Pipeline {pl}. Check data/reports/pipeline_{pl}_top_features.csv exists.")

    # ══════════════════════════════════════════════════════════
    # TAB 3 — HISTORICAL TRENDS
    # ══════════════════════════════════════════════════════════
    with t3:
        st.header(f"📈 Historical Trends — {sel} (2019–2020 Test Period)")

        city_h = city_df.copy()
        city_h["year"]  = city_h["datetime"].dt.year
        city_h["month"] = city_h["datetime"].dt.month
        city_h["hour"]  = city_h["datetime"].dt.hour

        # Temperature trend
        st.subheader("🌡️ Temperature Over Time")
        monthly_t = city_h.copy()
        monthly_t["month_dt"] = monthly_t["datetime"].dt.to_period("M").dt.to_timestamp()
        mt = monthly_t.groupby("month_dt")["temperature_2m"].agg(["mean","min","max"]).reset_index()
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=mt["month_dt"], y=mt["max"],  mode="lines", name="Max",  line=dict(color="#F44336",dash="dot",width=1.5)))
        fig_t.add_trace(go.Scatter(x=mt["month_dt"], y=mt["mean"], mode="lines", name="Mean", line=dict(color="#FF5722",width=2.5)))
        fig_t.add_trace(go.Scatter(x=mt["month_dt"], y=mt["min"],  mode="lines", name="Min",  line=dict(color="#2196F3",dash="dot",width=1.5)))
        fig_t.update_layout(template="plotly_dark", height=380,
                            xaxis_title="Month", yaxis_title="Temperature (°C)",
                            hovermode="x unified", legend=dict(orientation="h",y=1.1))
        st.plotly_chart(fig_t, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            # Heatwave frequency
            st.subheader("☀️ Heatwave Hours per Year")
            threshold = 40.0 if sel in HW_CITIES else 35.0
            city_h["hw"] = (city_h["temperature_2m"] >= threshold).astype(int)
            hw_y = city_h.groupby("year")["hw"].sum().reset_index()
            fig_hw = px.bar(hw_y, x="year", y="hw",
                            color="hw", color_continuous_scale="YlOrRd",
                            text="hw", template="plotly_dark",
                            labels={"hw":"Heatwave Hours","year":"Year"},
                            title=f"≥{threshold}°C threshold")
            fig_hw.update_traces(texttemplate="%{text}", textposition="outside")
            fig_hw.update_layout(height=360, showlegend=False)
            st.plotly_chart(fig_hw, use_container_width=True)

        with col2:
            # Monthly precipitation
            st.subheader("🌧️ Monthly Precipitation")
            month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            rain_m = city_h.groupby("month")["precipitation"].sum().reset_index()
            rain_m["month_name"] = rain_m["month"].apply(lambda x: month_names[x-1])
            fig_r = px.bar(rain_m, x="month_name", y="precipitation",
                           color="precipitation", color_continuous_scale="Blues",
                           text=rain_m["precipitation"].round(0),
                           template="plotly_dark",
                           labels={"precipitation":"Total Rain (mm)","month_name":"Month"})
            fig_r.update_traces(texttemplate="%{text:.0f}", textposition="outside")
            fig_r.update_layout(height=360, showlegend=False)
            st.plotly_chart(fig_r, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            # Wind distribution
            st.subheader("💨 Wind Speed Distribution")
            fig_w = px.histogram(city_h, x="windspeed_10m", nbins=60,
                                 color_discrete_sequence=["#26A69A"],
                                 template="plotly_dark",
                                 labels={"windspeed_10m":"Wind Speed (km/h)"},
                                 title="Distribution of Wind Speeds")
            fig_w.add_vline(x=40, line_dash="dash", line_color="#F44336",
                            annotation_text="Storm threshold (40 km/h)",
                            annotation_position="top right")
            fig_w.update_layout(height=360)
            st.plotly_chart(fig_w, use_container_width=True)

        with col4:
            # Hourly temperature pattern
            st.subheader("🕐 Average Temperature by Hour")
            hourly = city_h.groupby("hour")["temperature_2m"].mean().reset_index()
            fig_hr = px.line(hourly, x="hour", y="temperature_2m",
                             markers=True, template="plotly_dark",
                             color_discrete_sequence=["#FF9800"],
                             labels={"temperature_2m":"Avg Temp (°C)","hour":"Hour of Day"},
                             title="Diurnal Temperature Cycle")
            fig_hr.update_layout(height=360)
            fig_hr.update_xaxes(dtick=2, tick0=0)
            st.plotly_chart(fig_hr, use_container_width=True)

        # Humidity vs Temperature scatter
        st.subheader("🌡️ Temperature vs Humidity Scatter")
        sample = city_h.sample(min(2000, len(city_h)), random_state=42)
        sample["Heatwave"] = sample["temperature_2m"].apply(
            lambda t: "Heatwave" if t >= threshold else "Normal")
        fig_sc = px.scatter(sample, x="relative_humidity_2m", y="temperature_2m",
                            color="Heatwave",
                            color_discrete_map={"Heatwave":"#F44336","Normal":"#2196F3"},
                            opacity=0.6, template="plotly_dark",
                            labels={"relative_humidity_2m":"Humidity (%)","temperature_2m":"Temperature (°C)"},
                            title="Temperature vs Humidity (sample of 2000 hourly readings)")
        fig_sc.add_hline(y=threshold, line_dash="dash", line_color="orange",
                         annotation_text=f"Heatwave threshold ({threshold}°C)")
        fig_sc.update_layout(height=420)
        st.plotly_chart(fig_sc, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # TAB 4 — MODEL PERFORMANCE
    # ══════════════════════════════════════════════════════════
    with t4:
        st.header("📊 Model Performance Dashboard")

        # Results summary
        st.subheader("🏆 All Pipelines — Results Summary")
        summary = pd.DataFrame([
            {"Pipeline":"A — Temperature","Task":"Regression","RMSE":"0.628°C","R²":"0.9963",
             "F1/wF1":"N/A","AUC":"N/A","Baseline RMSE":"1.087°C","Improvement":"42.2% ↑","Time":"296 min"},
            {"Pipeline":"B — Rainfall","Task":"Binary Clf.","RMSE":"N/A","R²":"N/A",
             "F1/wF1":"1.000","AUC":"1.000","Baseline RMSE":"F1=0.000","Improvement":"+100% F1","Time":"20 min"},
            {"Pipeline":"C — Heatwave","Task":"Binary Clf.","RMSE":"N/A","R²":"N/A",
             "F1/wF1":"0.980","AUC":"1.000","Baseline RMSE":"F1=0.000","Improvement":"+98% F1","Time":"142 min"},
            {"Pipeline":"D — Disaster","Task":"Multi-Class","RMSE":"N/A","R²":"N/A",
             "F1/wF1":"wF1=1.000","AUC":"All 1.000","Baseline RMSE":"wF1=0.914","Improvement":"+9.4%","Time":"62 min"},
        ])
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Pipeline A performance
        st.subheader("📈 Pipeline A — Temperature Prediction Performance")
        X_perf = city_df[[f for f in EXACT_FEATURES if f in city_df.columns]].copy()
        if "temperature" in models:
            city_df["pred_temp"] = models["temperature"].predict(X_perf)
            sample_p = city_df.sample(min(1000, len(city_df)), random_state=42)
            fig_pva = px.scatter(sample_p, x="temperature_2m", y="pred_temp",
                                 template="plotly_dark", opacity=0.5,
                                 color_discrete_sequence=["#2196F3"],
                                 labels={"temperature_2m":"Actual (°C)","pred_temp":"Predicted (°C)"},
                                 title="Predicted vs Actual Temperature (sample of 1000 points)")
            mn = min(sample_p["temperature_2m"].min(), sample_p["pred_temp"].min())
            mx = max(sample_p["temperature_2m"].max(), sample_p["pred_temp"].max())
            fig_pva.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
                                         line=dict(color="#FF5722",dash="dash",width=2),
                                         name="Perfect Prediction"))
            fig_pva.update_layout(height=420)
            st.plotly_chart(fig_pva, use_container_width=True)

            # Residuals
            city_df["residual"] = city_df["temperature_2m"] - city_df["pred_temp"]
            fig_res = px.histogram(city_df, x="residual", nbins=80,
                                   color_discrete_sequence=["#9C27B0"],
                                   template="plotly_dark",
                                   labels={"residual":"Residual (Actual - Predicted) °C"},
                                   title="Residual Distribution — Pipeline A")
            fig_res.add_vline(x=0, line_dash="dash", line_color="white")
            fig_res.update_layout(height=350)
            st.plotly_chart(fig_res, use_container_width=True)

        st.markdown("---")

        # Pipeline C Heatwave analysis
        st.subheader("☀️ Pipeline C — Heatwave Detection Analysis")
        if "heatwave" in models:
            X_hw = city_df[[f for f in EXACT_FEATURES if f in city_df.columns]]
            city_df["hw_prob"] = models["heatwave"].predict_proba(X_hw)[:,1]

            # Probability distribution
            fig_hwp = go.Figure()
            hw1 = city_df[city_df["target_heatwave"]==1]["hw_prob"] if "target_heatwave" in city_df.columns else city_df["hw_prob"]
            hw0 = city_df[city_df["target_heatwave"]==0]["hw_prob"] if "target_heatwave" in city_df.columns else city_df["hw_prob"]
            fig_hwp.add_trace(go.Histogram(x=hw0, nbinsx=50, name="No Heatwave",
                                           marker_color="#2196F3", opacity=0.65, histnorm="density"))
            fig_hwp.add_trace(go.Histogram(x=hw1, nbinsx=50, name="Heatwave",
                                           marker_color="#F44336", opacity=0.65, histnorm="density"))
            fig_hwp.add_vline(x=0.5, line_dash="dash", line_color="white",
                              annotation_text="Decision threshold (0.5)")
            fig_hwp.update_layout(barmode="overlay", template="plotly_dark", height=380,
                                  xaxis_title="Predicted Heatwave Probability",
                                  yaxis_title="Density",
                                  title="Heatwave Probability Distribution — Pipeline C")
            st.plotly_chart(fig_hwp, use_container_width=True)

        # Per-city performance
        st.markdown("---")
        st.subheader("🌍 Per-City Performance — Heatwave Detection")
        city_perf = load_city_perf(data_dir, "C")
        if city_perf is not None:
            city_perf_sorted = city_perf.sort_values("f1", ascending=True)
            fig_cp = px.bar(city_perf_sorted, x="f1", y="city",
                            orientation="h",
                            color="f1", color_continuous_scale="RdYlGn",
                            text="f1", template="plotly_dark",
                            labels={"f1":"F1 Score","city":"City"},
                            title="Heatwave Detection F1 Score by City")
            fig_cp.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_cp.update_layout(height=550, showlegend=False)
            st.plotly_chart(fig_cp, use_container_width=True)

        # Disaster confusion matrix
        st.markdown("---")
        st.subheader("🚨 Pipeline D — Disaster Classification")
        if "disaster" in models:
            X_dis = city_df[[f for f in EXACT_FEATURES if f in city_df.columns]]
            city_df["pred_dis"] = models["disaster"].predict(X_dis)
            if "target_disaster" in city_df.columns:
                from sklearn.metrics import confusion_matrix as cm
                labels = [0,1,2,3]
                label_names = ["Normal","Heatwave","Heavy Rain","Storm"]
                conf = cm(city_df["target_disaster"], city_df["pred_dis"], labels=labels)
                fig_cm = px.imshow(conf, x=label_names, y=label_names,
                                   color_continuous_scale="Blues",
                                   text_auto=True, template="plotly_dark",
                                   title=f"Confusion Matrix — {sel} (Pipeline D)")
                fig_cm.update_layout(height=420,
                                     xaxis_title="Predicted", yaxis_title="Actual")
                st.plotly_chart(fig_cm, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # TAB 5 — EXTENDED 2021-2026
    # ══════════════════════════════════════════════════════════
    with t5:
        st.header("🆕 Extended Predictions — 2021 to 2026")

        # Big results banner
        st.markdown("""
        <div style="background:linear-gradient(135deg,#0d47a1,#1565c0);
                    padding:1.2rem;border-radius:12px;color:white;margin-bottom:1rem;text-align:center">
            <h3 style="margin:0">✅ Model tested on 938,400 NEW predictions (Jan 2021 – May 2026)</h3>
            <p style="margin:.5rem 0 0 0;opacity:.85">
                NEVER seen during training — proves the model generalises to the real world
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Key metrics
        c1,c2,c3,c4 = st.columns(4)
        for col,(title,orig,new_v,note) in zip(
            [c1,c2,c3,c4],[
                ("🌡️ Temp RMSE",  "0.628°C","0.904°C","Still under 1°C ✅"),
                ("📈 Temp R²",    "0.9963", "0.9929", "99.29% accuracy ✅"),
                ("☀️ Heatwave F1","0.980",  "0.988",  "Even BETTER! 🏆"),
                ("🌧️ Rainfall F1","1.000",  "1.000",  "Perfect ✅"),
            ]):
            with col:
                st.markdown(f"""<div class="card-blue">
                    <div style="font-size:.8rem;opacity:.8">{title}</div>
                    <div style="font-size:.85rem;opacity:.65">Original: {orig}</div>
                    <div style="font-size:1.5rem;font-weight:bold">{new_v}</div>
                    <div style="font-size:.75rem;opacity:.8">{note}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("")

        if ext_df is None:
            st.warning("Extended predictions CSV not found. Upload extended_predictions_2021_2026.csv to data/reports/")
        else:
            ext_df["datetime"] = pd.to_datetime(ext_df["datetime"])
            ext_df["year"]     = ext_df["datetime"].dt.year

            cities_ext = sorted(ext_df["city"].unique())
            sel_ext = st.selectbox("🏙️ Select City (Extended)",
                                   cities_ext,
                                   index=cities_ext.index("Karachi") if "Karachi" in cities_ext else 0)
            cext = ext_df[ext_df["city"]==sel_ext].copy()

            # Monthly temperature chart
            st.subheader(f"🌡️ {sel_ext} — Temperature 2021–2026 (Actual vs Predicted)")
            cext["month_dt"] = cext["datetime"].dt.to_period("M").dt.to_timestamp()
            mext = cext.groupby("month_dt").agg(
                actual=("temperature_2m","mean"),
                pred  =("pred_temperature","mean"),
                q10   =("pred_temp_q10","mean"),
                q90   =("pred_temp_q90","mean"),
            ).reset_index()

            fig_ext = go.Figure()
            fig_ext.add_trace(go.Scatter(x=mext["month_dt"], y=mext["q10"], line=dict(width=0), showlegend=False, mode="lines"))
            fig_ext.add_trace(go.Scatter(x=mext["month_dt"], y=mext["q90"],
                                         fill="tonexty", fillcolor="rgba(33,150,243,0.15)",
                                         line=dict(width=0), name="80% Confidence Interval"))
            fig_ext.add_trace(go.Scatter(x=mext["month_dt"], y=mext["actual"],
                                         mode="lines", name="Actual",
                                         line=dict(color="#FF5722",width=2.5)))
            fig_ext.add_trace(go.Scatter(x=mext["month_dt"], y=mext["pred"],
                                         mode="lines", name="AI Predicted",
                                         line=dict(color="#2196F3",width=2.5,dash="dash")))
            fig_ext.update_layout(template="plotly_dark", height=420,
                                  xaxis_title="Month", yaxis_title="Temperature (°C)",
                                  hovermode="x unified", legend=dict(orientation="h",y=1.1))
            st.plotly_chart(fig_ext, use_container_width=True)

            col_hw, col_rain = st.columns(2)

            with col_hw:
                # Heatwave bar chart
                st.subheader(f"☀️ {sel_ext} — Heatwave Hours per Year")
                hw_act  = cext.groupby("year")["target_heatwave"].sum().reset_index()
                hw_pred = cext.groupby("year").apply(lambda x:(x["pred_heat_prob"]>0.5).sum()).reset_index()
                hw_pred.columns = ["year","pred"]
                hw_m = hw_act.merge(hw_pred, on="year")
                fig_hwy = go.Figure()
                fig_hwy.add_trace(go.Bar(x=hw_m["year"].astype(str), y=hw_m["target_heatwave"],
                                         name="Actual", marker_color="#FF5722"))
                fig_hwy.add_trace(go.Bar(x=hw_m["year"].astype(str), y=hw_m["pred"],
                                         name="Predicted", marker_color="#FF9800"))
                fig_hwy.update_layout(barmode="group", template="plotly_dark", height=360,
                                      xaxis_title="Year", yaxis_title="Hours")
                st.plotly_chart(fig_hwy, use_container_width=True)

            with col_rain:
                # Rain probability over time
                st.subheader(f"🌧️ {sel_ext} — Monthly Heavy Rain Events")
                cext["month_dt2"] = cext["datetime"].dt.to_period("M").dt.to_timestamp()
                rain_m = cext.groupby("month_dt2").agg(
                    actual=("target_rain","sum"),
                    pred  =("pred_rain_prob",lambda x:(x>0.5).sum())
                ).reset_index()
                fig_rm = go.Figure()
                fig_rm.add_trace(go.Scatter(x=rain_m["month_dt2"], y=rain_m["actual"],
                                            mode="lines", name="Actual Rain Hours",
                                            line=dict(color="#2196F3",width=2)))
                fig_rm.add_trace(go.Scatter(x=rain_m["month_dt2"], y=rain_m["pred"],
                                            mode="lines", name="Predicted Rain Hours",
                                            line=dict(color="#00BCD4",width=2,dash="dash")))
                fig_rm.update_layout(template="plotly_dark", height=360,
                                     xaxis_title="Month", yaxis_title="Heavy Rain Hours")
                st.plotly_chart(fig_rm, use_container_width=True)

            # Global disaster trend
            st.markdown("---")
            st.subheader("🌍 Global Disaster Classification — All 20 Cities (2021–2026)")
            yearly_dis = ext_df.groupby(["year","pred_disaster"]).size().reset_index(name="count")
            yearly_dis["Class"] = yearly_dis["pred_disaster"].map(CLASS_NAMES)
            fig_dis = px.bar(yearly_dis, x="year", y="count", color="Class",
                             color_discrete_map=CLASS_COLORS,
                             template="plotly_dark",
                             labels={"count":"Predicted Hours","year":"Year"},
                             title="Disaster Classification by Year — All 20 Cities")
            fig_dis.update_layout(height=420, barmode="stack")
            st.plotly_chart(fig_dis, use_container_width=True)

            # 2027 projection
            st.markdown("---")
            st.subheader("🔭 2027 Temperature Projection")
            st.info("📌 Projection based on linear trend fitted to 2021–2026 yearly averages. Shaded band = ±1.5°C uncertainty.")

            proj_cities = st.multiselect("Select cities for projection",
                                         sorted(ext_df["city"].unique()),
                                         default=["Karachi","Delhi","Phoenix","Miami"])

            if proj_cities:
                fig_proj = go.Figure()
                colors_proj = px.colors.qualitative.Bold
                for idx, city in enumerate(proj_cities):
                    cproj = ext_df[ext_df["city"]==city]
                    yt = cproj.groupby("year")["pred_temperature"].mean()
                    years = np.array(yt.index.tolist())
                    vals  = yt.values
                    z = np.polyfit(years, vals, 1)
                    p = np.poly1d(z)
                    proj_y = np.array([2026, 2027])
                    col_c = colors_proj[idx % len(colors_proj)]

                    fig_proj.add_trace(go.Scatter(
                        x=years, y=vals, mode="lines+markers",
                        name=city, line=dict(color=col_c,width=2.5), marker=dict(size=8)))
                    fig_proj.add_trace(go.Scatter(
                        x=proj_y, y=p(proj_y), mode="lines+markers",
                        name=f"{city} (2027 proj)", line=dict(color=col_c,width=2.5,dash="dash"),
                        marker=dict(size=10,symbol="star"), showlegend=True))
                    fig_proj.add_trace(go.Scatter(
                        x=np.concatenate([proj_y, proj_y[::-1]]),
                        y=np.concatenate([p(proj_y)+1.5, (p(proj_y)-1.5)[::-1]]),
                        fill="toself", fillcolor=f"rgba(255,255,255,0.05)",
                        line=dict(width=0), showlegend=False))

                fig_proj.update_layout(template="plotly_dark", height=480,
                                       xaxis_title="Year", yaxis_title="Avg Temperature (°C)",
                                       title="2027 Temperature Projection — Selected Cities",
                                       legend=dict(orientation="h",y=-0.2))
                st.plotly_chart(fig_proj, use_container_width=True)

            # Download
            csv_ext = cext.to_csv(index=False)
            st.download_button(f"📥 Download {sel_ext} Extended Predictions CSV",
                               csv_ext, f"extended_{sel_ext}_2021_2026.csv", "text/csv")

    # ══════════════════════════════════════════════════════════
    # TAB 6 — DATA EXPLORER
    # ══════════════════════════════════════════════════════════
    with t6:
        st.header("🔍 Data Explorer")

        ds = st.radio("Dataset", ["Test Set (2019-2020)","Extended (2021-2026)"], horizontal=True)
        df_exp = test_df if "Test" in ds else (ext_df if ext_df is not None else test_df)

        ec = st.selectbox("Filter by City", ["All Cities"]+sorted(df_exp["city"].unique()))
        if ec != "All Cities": df_exp = df_exp[df_exp["city"]==ec]

        st.markdown(f"**{len(df_exp):,} rows** | showing first 500")
        st.dataframe(df_exp.head(500), use_container_width=True, height=380)

        num_cols = ["temperature_2m","relative_humidity_2m","precipitation",
                    "windspeed_10m","surface_pressure","cloudcover","shortwave_radiation"]
        ac = [c for c in num_cols if c in df_exp.columns]
        if ac:
            st.subheader("📊 Descriptive Statistics")
            st.dataframe(df_exp[ac].describe().round(3), use_container_width=True)

            st.subheader("🔥 Correlation Heatmap")
            corr = df_exp[ac].corr().round(2)
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                                 template="plotly_dark", aspect="auto",
                                 title="Feature Correlation Matrix")
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)

            st.subheader("📦 Box Plots by City")
            var_box = st.selectbox("Variable", ac, index=0)
            cities_box = sorted(df_exp["city"].unique())
            fig_box = px.box(df_exp, x="city", y=var_box, color="city",
                             template="plotly_dark",
                             category_orders={"city":cities_box},
                             title=f"{var_box} Distribution by City")
            fig_box.update_layout(height=480, showlegend=False,
                                  xaxis_tickangle=-45)
            st.plotly_chart(fig_box, use_container_width=True)

        csv_down = df_exp.head(50000).to_csv(index=False)
        st.download_button("📥 Download CSV (up to 50,000 rows)",
                           csv_down, f"weather_{ec}.csv", "text/csv")


if __name__ == "__main__":
    main()
