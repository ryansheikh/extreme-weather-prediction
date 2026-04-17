#!/usr/bin/env python3
"""
==============================================================================
FILE 3: pipeline_A_temperature.py
==============================================================================
Project : AI-Driven Extreme Weather Prediction — A Global Perspective
Pipeline: A — Temperature Forecasting (Regression)
Mode    : ULTRA-THOROUGH (publication-quality)
Author  : [Your Name]

Purpose:
    Predict next-hour temperature using XGBoost regression.
    This pipeline is fully self-contained with:
      • Hyperparameter tuning (time-series cross-validation)
      • Point prediction evaluation: RMSE, MAE, R², MAPE
      • Uncertainty quantification via QUANTILE REGRESSION
          - 10th, 50th, 90th percentile predictors
          - Prediction intervals with coverage calibration
      • SHAP explainability (summary, beeswarm, bar, waterfall, dependence)
      • Baseline comparison (persistence model)
      • Comprehensive publication-ready plots

Outputs:
    models/pipeline_A_temperature_xgb.pkl
    models/pipeline_A_q10.pkl
    models/pipeline_A_q90.pkl
    reports/pipeline_A_metrics.json
    reports/figures/pipeline_A_*.png
    reports/pipeline_A_top_features.csv
==============================================================================
"""

# ============================================================================
# 1. IMPORTS
# ============================================================================
import json
import time
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import joblib
import shap

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error,
)
from scipy import stats

warnings.filterwarnings("ignore")

# ============================================================================
# 2. LOGGING & STYLE
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "figure.figsize": (10, 6),
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

# ============================================================================
# 3. PATH SETUP
# ============================================================================
possible = [
    Path.cwd() / "data",
    Path.home() / "data",
    Path.home() / "Desktop" / "FYP 2026" / "data",
]
DATA_DIR = next((p for p in possible if (p / "processed" / "train.csv").exists()), None)
if DATA_DIR is None:
    raise FileNotFoundError("Cannot find data/processed/. Run File 2 first.")

PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR.parent / "models"
REPORTS_DIR = DATA_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for d in [MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logger.info(f"📁 Data:     {DATA_DIR}")
logger.info(f"📁 Models:   {MODELS_DIR}")
logger.info(f"📁 Reports:  {REPORTS_DIR}")

# ============================================================================
# 4. CONFIGURATION
# ============================================================================
TARGET = "target_temperature_next"  # created on-the-fly (next-hour temp)
RANDOM_SEED = 42

# Ultra-thorough hyperparameter search (coarse→fine strategy)
HP_SEARCH_SPACE = [
    # (n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight)
    (400, 6, 0.05, 0.9, 0.9, 3),
    (600, 6, 0.05, 0.9, 0.9, 3),
    (800, 7, 0.05, 0.85, 0.85, 3),
    (1000, 7, 0.03, 0.85, 0.85, 5),
    (1200, 8, 0.03, 0.8, 0.8, 5),
    (1500, 8, 0.02, 0.8, 0.8, 5),
]

# Time-series cross-validation folds
CV_FOLDS = 5

# Quantile levels for uncertainty estimation
QUANTILES = {"q10": 0.10, "q50": 0.50, "q90": 0.90}

# Sample size cap for SHAP (computing SHAP on 350k rows would take hours)
SHAP_SAMPLE_SIZE = 5000


# ============================================================================
# 5. DATA LOADING & TARGET PREPARATION
# ============================================================================

def load_splits():
    """Load train/val/test splits from File 2 output."""
    logger.info("\n📥 Loading preprocessed datasets...")

    train = pd.read_csv(PROCESSED_DIR / "train.csv", low_memory=False)
    val = pd.read_csv(PROCESSED_DIR / "val.csv", low_memory=False)
    test = pd.read_csv(PROCESSED_DIR / "test.csv", low_memory=False)

    for df in [train, val, test]:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df.sort_values(["city", "datetime"], inplace=True)

    logger.info(f"   ✓ train: {len(train):>10,} rows")
    logger.info(f"   ✓ val:   {len(val):>10,} rows")
    logger.info(f"   ✓ test:  {len(test):>10,} rows")

    return train, val, test


def create_next_hour_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create regression target = next-hour temperature (per city).
    We shift temperature_2m by -1 within each city to avoid data leakage.
    """
    df[TARGET] = df.groupby("city")["temperature_2m"].shift(-1)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Identify columns that should be model inputs.
    Exclude: targets, raw metadata, datetime.
    """
    exclude = {
        "datetime", "city", "country", "continent", "climate_zone",
        "heatwave_threshold",
        "target_rain", "target_heatwave", "target_storm", "target_disaster",
        TARGET,
    }
    return [c for c in df.columns if c not in exclude]


# ============================================================================
# 6. HYPERPARAMETER TUNING WITH TIME-SERIES CV
# ============================================================================

def hyperparameter_search(X_train, y_train, X_val, y_val):
    """
    Evaluate each candidate config. We use a single train/val hold-out
    (time-series-respecting since the split is already chronological)
    plus cross-validation for the best candidate.
    """
    logger.info("\n🔍 Hyperparameter tuning (ultra-thorough)...")
    logger.info(f"   Candidates: {len(HP_SEARCH_SPACE)}")

    results = []
    best_rmse = np.inf
    best_params = None

    for i, (n_est, depth, lr, subsample, colsample, mcw) in enumerate(HP_SEARCH_SPACE, 1):
        t0 = time.time()
        model = XGBRegressor(
            n_estimators=n_est,
            max_depth=depth,
            learning_rate=lr,
            subsample=subsample,
            colsample_bytree=colsample,
            min_child_weight=mcw,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            early_stopping_rounds=50,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        y_val_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        mae = mean_absolute_error(y_val, y_val_pred)
        r2 = r2_score(y_val, y_val_pred)
        elapsed = time.time() - t0

        results.append({
            "config": i,
            "n_estimators": n_est,
            "max_depth": depth,
            "learning_rate": lr,
            "subsample": subsample,
            "colsample": colsample,
            "min_child_weight": mcw,
            "val_rmse": round(rmse, 4),
            "val_mae": round(mae, 4),
            "val_r2": round(r2, 5),
            "time_sec": round(elapsed, 1),
        })

        logger.info(
            f"   [{i}/{len(HP_SEARCH_SPACE)}] n={n_est} depth={depth} lr={lr} "
            f"→ RMSE={rmse:.3f}, R²={r2:.4f}, time={elapsed:.0f}s"
        )

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = dict(
                n_estimators=n_est,
                max_depth=depth,
                learning_rate=lr,
                subsample=subsample,
                colsample_bytree=colsample,
                min_child_weight=mcw,
            )

    pd.DataFrame(results).to_csv(
        REPORTS_DIR / "pipeline_A_hp_search_results.csv", index=False
    )
    logger.info(f"\n   🏆 Best config: {best_params}")
    logger.info(f"   🏆 Best validation RMSE: {best_rmse:.4f}°C")

    return best_params


# ============================================================================
# 7. CROSS-VALIDATION OF CHAMPION MODEL
# ============================================================================

def cross_validate_best(X, y, best_params):
    """Time-series CV on full training set to confirm stability."""
    logger.info(f"\n🔁 Time-series cross-validation ({CV_FOLDS} folds)...")

    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    cv_rmse, cv_mae, cv_r2 = [], [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        Xtr, Xvl = X.iloc[train_idx], X.iloc[val_idx]
        ytr, yvl = y.iloc[train_idx], y.iloc[val_idx]

        # Drop early_stopping since we're not providing eval_set here
        params_cv = {k: v for k, v in best_params.items()}
        model = XGBRegressor(
            **params_cv,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        model.fit(Xtr, ytr, verbose=False)
        preds = model.predict(Xvl)

        cv_rmse.append(np.sqrt(mean_squared_error(yvl, preds)))
        cv_mae.append(mean_absolute_error(yvl, preds))
        cv_r2.append(r2_score(yvl, preds))

        logger.info(
            f"   Fold {fold}/{CV_FOLDS}: RMSE={cv_rmse[-1]:.3f}, "
            f"MAE={cv_mae[-1]:.3f}, R²={cv_r2[-1]:.4f}"
        )

    summary = {
        "cv_rmse_mean": float(np.mean(cv_rmse)),
        "cv_rmse_std": float(np.std(cv_rmse)),
        "cv_mae_mean": float(np.mean(cv_mae)),
        "cv_r2_mean": float(np.mean(cv_r2)),
    }
    logger.info(
        f"\n   CV Summary: RMSE = {summary['cv_rmse_mean']:.3f} ± "
        f"{summary['cv_rmse_std']:.3f}"
    )
    return summary


# ============================================================================
# 8. FINAL MODEL TRAINING (POINT PREDICTION)
# ============================================================================

def train_final_model(X_train, y_train, X_val, y_val, best_params):
    """Train final point-prediction model on train, early-stop on val."""
    logger.info("\n🏋️ Training final point-prediction model...")
    t0 = time.time()

    model = XGBRegressor(
        **best_params,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        early_stopping_rounds=80,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    logger.info(f"   ✓ Trained in {time.time()-t0:.0f}s")
    logger.info(f"   ✓ Best iteration: {model.best_iteration}")
    return model


# ============================================================================
# 9. QUANTILE REGRESSION MODELS (UNCERTAINTY)
# ============================================================================

def train_quantile_models(X_train, y_train, X_val, y_val, best_params):
    """
    Train three quantile regressors (10th, 50th, 90th percentile) for
    predicting full uncertainty intervals.
    """
    logger.info("\n🎲 Training quantile regression models (uncertainty)...")
    quantile_models = {}

    for name, q in QUANTILES.items():
        t0 = time.time()
        params_q = {k: v for k, v in best_params.items()}

        model = XGBRegressor(
            **params_q,
            objective="reg:quantileerror",
            quantile_alpha=q,
            tree_method="hist",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            early_stopping_rounds=80,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        quantile_models[name] = model
        logger.info(
            f"   ✓ {name} (α={q}): trained in {time.time()-t0:.0f}s"
        )

    return quantile_models


def evaluate_prediction_intervals(y_true, q10, q50, q90):
    """Evaluate prediction interval quality."""
    # Coverage: fraction of true values within [q10, q90]
    coverage = ((y_true >= q10) & (y_true <= q90)).mean()
    # Interval width (sharpness): narrower = better if coverage is high
    avg_width = (q90 - q10).mean()
    # Pinball loss for q10 and q90 (lower is better)
    def pinball(y, qpred, alpha):
        diff = y - qpred
        return np.mean(np.maximum(alpha * diff, (alpha - 1) * diff))
    pb10 = pinball(y_true, q10, 0.10)
    pb90 = pinball(y_true, q90, 0.90)

    return {
        "coverage_80pct": float(coverage),
        "target_coverage": 0.80,
        "coverage_gap": float(coverage - 0.80),
        "avg_interval_width_C": float(avg_width),
        "pinball_loss_q10": float(pb10),
        "pinball_loss_q90": float(pb90),
    }


# ============================================================================
# 10. BASELINE: PERSISTENCE MODEL
# ============================================================================

def persistence_baseline(X_test, y_test):
    """
    Persistence baseline: next-hour temperature = current temperature.
    This is the naive 'tomorrow = today' model that ML must beat.
    """
    logger.info("\n📏 Computing persistence baseline...")
    y_pred = X_test["temperature_2m"].values
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"   ✓ Baseline RMSE: {rmse:.3f}°C, MAE: {mae:.3f}°C, R²: {r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2, "predictions": y_pred}


# ============================================================================
# 11. EVALUATION
# ============================================================================

def evaluate_regression(y_true, y_pred, name="model"):
    """Compute standard regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # MAPE (robust: skip near-zero temperatures)
    mask = np.abs(y_true) > 1.0
    mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100 if mask.sum() > 0 else np.nan

    return {
        "name": name,
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "mape_pct": float(mape) if not np.isnan(mape) else None,
    }


def statistical_significance_test(y_true, pred_model, pred_baseline):
    """Wilcoxon signed-rank test: is model significantly better than baseline?"""
    errors_model = np.abs(y_true - pred_model)
    errors_baseline = np.abs(y_true - pred_baseline)
    stat, pvalue = stats.wilcoxon(errors_model, errors_baseline)
    return {
        "wilcoxon_statistic": float(stat),
        "p_value": float(pvalue),
        "model_better": bool(errors_model.mean() < errors_baseline.mean()),
        "significant_at_0.01": bool(pvalue < 0.01),
    }


def per_city_evaluation(test_df, y_true, y_pred):
    """Evaluate performance per city to check generalization."""
    logger.info("\n🌍 Per-city evaluation...")
    results = []
    for city in sorted(test_df["city"].unique()):
        mask = test_df["city"].values == city
        if mask.sum() < 10:
            continue
        y_t = y_true[mask]
        y_p = y_pred[mask]
        results.append({
            "city": city,
            "test_rows": int(mask.sum()),
            "rmse": round(float(np.sqrt(mean_squared_error(y_t, y_p))), 3),
            "mae": round(float(mean_absolute_error(y_t, y_p)), 3),
            "r2": round(float(r2_score(y_t, y_p)), 4),
        })
    city_df = pd.DataFrame(results).sort_values("rmse")
    city_df.to_csv(REPORTS_DIR / "pipeline_A_per_city_performance.csv", index=False)
    logger.info(f"   ✓ Best city:  {city_df.iloc[0]['city']} (RMSE={city_df.iloc[0]['rmse']}°C)")
    logger.info(f"   ✓ Worst city: {city_df.iloc[-1]['city']} (RMSE={city_df.iloc[-1]['rmse']}°C)")
    return city_df


def seasonal_evaluation(test_df, y_true, y_pred):
    """Evaluate performance per season."""
    results = []
    season_names = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Autumn"}
    for season_id, season_name in season_names.items():
        mask = test_df["season"].values == season_id
        if mask.sum() < 10:
            continue
        results.append({
            "season": season_name,
            "rows": int(mask.sum()),
            "rmse": round(float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))), 3),
            "mae": round(float(mean_absolute_error(y_true[mask], y_pred[mask])), 3),
            "r2": round(float(r2_score(y_true[mask], y_pred[mask])), 4),
        })
    return pd.DataFrame(results)


# ============================================================================
# 12. SHAP EXPLAINABILITY
# ============================================================================

def run_shap_analysis(model, X_sample, feature_names):
    """Compute SHAP values and generate publication-ready plots."""
    logger.info(f"\n🔬 SHAP analysis (sample={len(X_sample)})...")
    t0 = time.time()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    logger.info(f"   ✓ SHAP values computed in {time.time()-t0:.0f}s")

    # --- Plot 1: Summary (beeswarm) ---
    plt.figure(figsize=(11, 8))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        show=False, max_display=20,
    )
    plt.title("Pipeline A — SHAP Feature Impact (Beeswarm)",
              fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pipeline_A_shap_beeswarm.png")
    plt.close()
    logger.info("   ✓ Saved: pipeline_A_shap_beeswarm.png")

    # --- Plot 2: Bar chart ---
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False, max_display=20,
    )
    plt.title("Pipeline A — Top Features by Mean |SHAP|",
              fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pipeline_A_shap_bar.png")
    plt.close()
    logger.info("   ✓ Saved: pipeline_A_shap_bar.png")

    # --- Plot 3: Waterfall for a single prediction ---
    plt.figure(figsize=(10, 8))
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value,
        shap_values[0],
        X_sample.iloc[0],
        feature_names=feature_names,
        max_display=15,
        show=False,
    )
    plt.title("Pipeline A — SHAP Waterfall (Single Prediction)",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pipeline_A_shap_waterfall.png")
    plt.close()
    logger.info("   ✓ Saved: pipeline_A_shap_waterfall.png")

    # --- Top features table ---
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_features = (
        pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
        })
        .sort_values("mean_abs_shap", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
    top_features["mean_abs_shap"] = top_features["mean_abs_shap"].round(4)
    top_features.to_csv(REPORTS_DIR / "pipeline_A_top_features.csv", index=False)
    logger.info(
        f"   ✓ Top 5 features: {top_features['feature'].head(5).tolist()}"
    )

    # --- Plot 4: Dependence plot for top feature ---
    top_feat = top_features.iloc[0]["feature"]
    top_feat_idx = list(feature_names).index(top_feat)
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        top_feat_idx, shap_values, X_sample,
        feature_names=feature_names,
        show=False,
    )
    plt.title(f"Pipeline A — SHAP Dependence: {top_feat}",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pipeline_A_shap_dependence.png")
    plt.close()
    logger.info("   ✓ Saved: pipeline_A_shap_dependence.png")

    return top_features


# ============================================================================
# 13. VISUALIZATION
# ============================================================================

def plot_predictions_vs_actual(y_true, y_pred, title_suffix=""):
    """Scatter plot of predictions vs actual values."""
    plt.figure(figsize=(9, 9))
    idx = np.random.RandomState(42).choice(len(y_true), min(5000, len(y_true)), replace=False)
    plt.scatter(y_true[idx], y_pred[idx], alpha=0.25, s=8, c="#1f77b4", edgecolors="none")
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], "r--", lw=2, label="Ideal (y=x)")
    plt.xlabel("Actual Temperature (°C)")
    plt.ylabel("Predicted Temperature (°C)")
    plt.title(f"Pipeline A — Predictions vs Actual {title_suffix}",
              fontsize=13, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pipeline_A_predictions_vs_actual.png")
    plt.close()
    logger.info("   ✓ Saved: pipeline_A_predictions_vs_actual.png")


def plot_residuals(y_true, y_pred):
    """Residual analysis."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(residuals, bins=60, color="#2ca02c", edgecolor="black", alpha=0.8)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Residual (°C)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Residual Distribution")

    idx = np.random.RandomState(42).choice(len(y_pred), min(5000, len(y_pred)), replace=False)
    axes[1].scatter(y_pred[idx], residuals[idx], alpha=0.25, s=8, c="#ff7f0e")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Predicted Temperature (°C)")
    axes[1].set_ylabel("Residual (°C)")
    axes[1].set_title("Residuals vs Predictions")

    plt.suptitle("Pipeline A — Residual Analysis",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pipeline_A_residuals.png")
    plt.close()
    logger.info("   ✓ Saved: pipeline_A_residuals.png")


def plot_uncertainty_intervals(test_df, y_true, q10, q50, q90, city="Karachi"):
    """Plot predictions with uncertainty bands for one city."""
    mask = test_df["city"].values == city
    if mask.sum() == 0:
        return

    t = test_df.loc[mask, "datetime"].values
    y_c = y_true[mask]
    q10_c = q10[mask]
    q50_c = q50[mask]
    q90_c = q90[mask]

    # Show first 2 weeks for readability
    n_show = min(24 * 14, len(t))

    plt.figure(figsize=(14, 6))
    plt.plot(t[:n_show], y_c[:n_show], "k-", label="Actual", linewidth=1.5, alpha=0.9)
    plt.plot(t[:n_show], q50_c[:n_show], "b-", label="Predicted (median)", linewidth=1.5, alpha=0.9)
    plt.fill_between(
        t[:n_show], q10_c[:n_show], q90_c[:n_show],
        alpha=0.25, color="blue", label="80% Prediction Interval (10th–90th)",
    )
    plt.xlabel("Date (UTC)")
    plt.ylabel("Temperature (°C)")
    plt.title(f"Pipeline A — Uncertainty Forecast ({city}, 2 weeks)",
              fontsize=13, fontweight="bold")
    plt.legend(loc="best")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"pipeline_A_uncertainty_{city}.png")
    plt.close()
    logger.info(f"   ✓ Saved: pipeline_A_uncertainty_{city}.png")


def plot_model_vs_baseline(metrics_model, metrics_baseline):
    """Bar chart comparing XGBoost vs Persistence baseline."""
    metrics = ["rmse", "mae"]
    model_vals = [metrics_model[m] for m in metrics]
    base_vals = [metrics_baseline[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(8, 6))
    plt.bar(x - width/2, base_vals, width, label="Persistence Baseline", color="#d62728", alpha=0.85)
    plt.bar(x + width/2, model_vals, width, label="XGBoost (Pipeline A)", color="#2ca02c", alpha=0.85)
    plt.xticks(x, [m.upper() for m in metrics])
    plt.ylabel("Error (°C, lower is better)")
    plt.title("Pipeline A — Model vs Baseline",
              fontsize=13, fontweight="bold")
    plt.legend()
    for i, (b, m) in enumerate(zip(base_vals, model_vals)):
        plt.text(i - width/2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=10)
        plt.text(i + width/2, m, f"{m:.2f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pipeline_A_baseline_comparison.png")
    plt.close()
    logger.info("   ✓ Saved: pipeline_A_baseline_comparison.png")


# ============================================================================
# 14. MAIN
# ============================================================================

def main():
    print(r"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║   PIPELINE A — TEMPERATURE FORECASTING                         ║
    ║   Model: XGBoost Regressor                                     ║
    ║   Uncertainty: Quantile Regression (10/50/90)                  ║
    ║   Explainability: SHAP (beeswarm + bar + waterfall)            ║
    ║   Mode: ULTRA-THOROUGH (publication quality)                   ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    total_start = time.time()

    # ── Load data ──
    train, val, test = load_splits()

    # ── Create regression target ──
    logger.info("\n🎯 Creating next-hour temperature target...")
    train = create_next_hour_target(train)
    val = create_next_hour_target(val)
    test = create_next_hour_target(test)

    # Drop rows where next-hour target is NaN (last hour per city)
    train = train.dropna(subset=[TARGET]).reset_index(drop=True)
    val = val.dropna(subset=[TARGET]).reset_index(drop=True)
    test = test.dropna(subset=[TARGET]).reset_index(drop=True)

    features = get_feature_columns(train)
    logger.info(f"   ✓ Using {len(features)} input features")
    logger.info(f"   ✓ Target: {TARGET}")

    X_train, y_train = train[features], train[TARGET]
    X_val, y_val = val[features], val[TARGET]
    X_test, y_test = test[features], test[TARGET]

    # ── Hyperparameter search ──
    best_params = hyperparameter_search(X_train, y_train, X_val, y_val)

    # ── Cross-validation ──
    cv_summary = cross_validate_best(
        pd.concat([X_train, X_val]),
        pd.concat([y_train, y_val]),
        best_params,
    )

    # ── Train final point-prediction model ──
    model = train_final_model(X_train, y_train, X_val, y_val, best_params)

    # ── Train quantile models ──
    quantile_models = train_quantile_models(
        X_train, y_train, X_val, y_val, best_params
    )

    # ── Evaluate on test set ──
    logger.info("\n📊 Evaluating on test set (2019–2020)...")
    y_test_pred = model.predict(X_test)
    metrics_test = evaluate_regression(y_test.values, y_test_pred, "XGBoost (test)")
    logger.info(
        f"   ✓ Test RMSE={metrics_test['rmse']:.3f}°C, "
        f"MAE={metrics_test['mae']:.3f}°C, "
        f"R²={metrics_test['r2']:.4f}"
    )

    # ── Baseline comparison ──
    baseline = persistence_baseline(X_test, y_test.values)
    sig_test = statistical_significance_test(
        y_test.values, y_test_pred, baseline["predictions"]
    )
    logger.info(
        f"   ✓ Wilcoxon p-value vs baseline: {sig_test['p_value']:.2e} "
        f"(significant: {sig_test['significant_at_0.01']})"
    )

    # ── Uncertainty evaluation ──
    logger.info("\n🎲 Evaluating uncertainty intervals...")
    q10_pred = quantile_models["q10"].predict(X_test)
    q50_pred = quantile_models["q50"].predict(X_test)
    q90_pred = quantile_models["q90"].predict(X_test)
    uncertainty_metrics = evaluate_prediction_intervals(
        y_test.values, q10_pred, q50_pred, q90_pred
    )
    logger.info(
        f"   ✓ 80% interval coverage: {uncertainty_metrics['coverage_80pct']:.1%} "
        f"(target: 80%)"
    )
    logger.info(
        f"   ✓ Average interval width: "
        f"{uncertainty_metrics['avg_interval_width_C']:.2f}°C"
    )

    # ── Per-city and seasonal analysis ──
    city_perf = per_city_evaluation(test, y_test.values, y_test_pred)
    season_perf = seasonal_evaluation(test, y_test.values, y_test_pred)
    season_perf.to_csv(REPORTS_DIR / "pipeline_A_seasonal_performance.csv", index=False)

    # ── SHAP analysis ──
    X_shap = X_test.sample(SHAP_SAMPLE_SIZE, random_state=RANDOM_SEED)
    top_features = run_shap_analysis(model, X_shap, features)

    # ── Visualizations ──
    logger.info("\n🎨 Generating publication figures...")
    plot_predictions_vs_actual(y_test.values, y_test_pred, "(Test Set)")
    plot_residuals(y_test.values, y_test_pred)
    plot_uncertainty_intervals(
        test, y_test.values, q10_pred, q50_pred, q90_pred, city="Karachi"
    )
    plot_uncertainty_intervals(
        test, y_test.values, q10_pred, q50_pred, q90_pred, city="Mumbai"
    )
    plot_model_vs_baseline(metrics_test, baseline)

    # ── Save models ──
    logger.info("\n💾 Saving models...")
    joblib.dump(model, MODELS_DIR / "pipeline_A_temperature_xgb.pkl")
    for name, qm in quantile_models.items():
        joblib.dump(qm, MODELS_DIR / f"pipeline_A_{name}.pkl")
    logger.info(f"   ✓ Saved 4 models to {MODELS_DIR}")

    # ── Save metrics ──
    final_metrics = {
        "pipeline": "A",
        "task": "temperature_forecasting",
        "best_hyperparameters": best_params,
        "cross_validation": cv_summary,
        "test_metrics": metrics_test,
        "persistence_baseline": {
            "rmse": baseline["rmse"],
            "mae": baseline["mae"],
            "r2": baseline["r2"],
        },
        "improvement_over_baseline_pct": {
            "rmse": round((baseline["rmse"] - metrics_test["rmse"]) / baseline["rmse"] * 100, 2),
            "mae": round((baseline["mae"] - metrics_test["mae"]) / baseline["mae"] * 100, 2),
        },
        "statistical_significance": sig_test,
        "uncertainty_metrics": uncertainty_metrics,
        "top_5_features": top_features["feature"].head(5).tolist(),
        "per_city_performance_file": "pipeline_A_per_city_performance.csv",
        "seasonal_performance_file": "pipeline_A_seasonal_performance.csv",
    }
    with open(REPORTS_DIR / "pipeline_A_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    logger.info(f"   ✓ Saved: pipeline_A_metrics.json")

    # ── Final summary ──
    elapsed_min = (time.time() - total_start) / 60
    print(f"\n{'='*70}")
    print("  ✅ PIPELINE A — TEMPERATURE FORECASTING — COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time:                  {elapsed_min:.1f} minutes")
    print(f"  Test RMSE:                   {metrics_test['rmse']:.3f}°C")
    print(f"  Test MAE:                    {metrics_test['mae']:.3f}°C")
    print(f"  Test R²:                     {metrics_test['r2']:.4f}")
    print(f"  Baseline RMSE:               {baseline['rmse']:.3f}°C")
    print(f"  Improvement over baseline:   "
          f"{final_metrics['improvement_over_baseline_pct']['rmse']:.1f}% RMSE reduction")
    print(f"  80% interval coverage:       "
          f"{uncertainty_metrics['coverage_80pct']:.1%}")
    print(f"  Top feature:                 {top_features.iloc[0]['feature']}")
    print(f"  Outputs:")
    print(f"    · Models:   {MODELS_DIR}")
    print(f"    · Reports:  {REPORTS_DIR}")
    print(f"    · Figures:  {FIGURES_DIR}")
    print(f"{'='*70}\n")

    return model, quantile_models, final_metrics


if __name__ == "__main__":
    model, quantile_models, metrics = main()
