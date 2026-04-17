#!/usr/bin/env python3
"""
==============================================================================
FILE 4: pipeline_B_rainfall.py
==============================================================================
Project : AI-Driven Extreme Weather Prediction — A Global Perspective
Pipeline: B — Rainfall / Heavy Rain Prediction (Binary Classification)
Author  : [Your Name]

Purpose:
    Predict whether heavy rainfall (>1mm/hr) will occur using XGBoost
    binary classifier. Includes:
      • Class imbalance handling (scale_pos_weight)
      • Hyperparameter tuning with time-series CV
      • Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC
      • Uncertainty: Calibration curve + Brier Score
      • Explainability: SHAP + LIME
      • Baseline comparison + statistical significance
      • Per-city and seasonal analysis

Outputs:
    models/pipeline_B_rainfall_xgb.pkl
    reports/pipeline_B_metrics.json
    reports/figures/pipeline_B_*.png
    reports/pipeline_B_top_features.csv
==============================================================================
"""

import json
import time
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    brier_score_loss, log_loss,
)
from sklearn.calibration import calibration_curve
from scipy import stats

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "figure.figsize": (10, 6), "axes.titlesize": 13, "axes.labelsize": 11,
})

# ============================================================
# PATHS
# ============================================================
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

logger.info(f"📁 Data: {DATA_DIR}")

# ============================================================
# CONFIG
# ============================================================
TARGET = "target_rain"
RANDOM_SEED = 42
SHAP_SAMPLE_SIZE = 5000
LIME_SAMPLES = 3

# Ultra-thorough hyperparameter search
HP_SEARCH_SPACE = [
    # (n_estimators, max_depth, learning_rate, subsample, colsample, min_child_weight, gamma)
    (400, 5, 0.05, 0.9, 0.9, 3, 0.0),
    (600, 6, 0.05, 0.85, 0.85, 3, 0.1),
    (800, 6, 0.03, 0.85, 0.85, 5, 0.1),
    (1000, 7, 0.03, 0.8, 0.8, 5, 0.2),
    (1200, 7, 0.02, 0.8, 0.8, 5, 0.2),
    (1500, 8, 0.02, 0.75, 0.75, 7, 0.3),
]

CV_FOLDS = 5


# ============================================================
# DATA LOADING
# ============================================================
def load_splits():
    logger.info("\n📥 Loading preprocessed datasets...")
    train = pd.read_csv(PROCESSED_DIR / "train.csv", low_memory=False)
    val = pd.read_csv(PROCESSED_DIR / "val.csv", low_memory=False)
    test = pd.read_csv(PROCESSED_DIR / "test.csv", low_memory=False)
    for df in [train, val, test]:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df.sort_values(["city", "datetime"], inplace=True)
    logger.info(f"   ✓ train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")
    return train, val, test


def get_feature_columns(df):
    exclude = {
        "datetime", "city", "country", "continent", "climate_zone",
        "heatwave_threshold",
        "target_rain", "target_heatwave", "target_storm", "target_disaster",
        "target_temperature_next",
    }
    return [c for c in df.columns if c not in exclude]


# ============================================================
# HYPERPARAMETER TUNING
# ============================================================
def hyperparameter_search(X_train, y_train, X_val, y_val, scale_pos_wt):
    logger.info(f"\n🔍 Hyperparameter tuning ({len(HP_SEARCH_SPACE)} configs)...")
    logger.info(f"   scale_pos_weight = {scale_pos_wt:.2f}")

    results = []
    best_f1 = -1
    best_params = None

    for i, (n_est, depth, lr, sub, col, mcw, gamma) in enumerate(HP_SEARCH_SPACE, 1):
        t0 = time.time()
        model = XGBClassifier(
            n_estimators=n_est, max_depth=depth, learning_rate=lr,
            subsample=sub, colsample_bytree=col, min_child_weight=mcw,
            gamma=gamma, scale_pos_weight=scale_pos_wt,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", random_state=RANDOM_SEED, n_jobs=-1,
            early_stopping_rounds=50,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        f1 = f1_score(y_val, y_pred)
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_prob)
        elapsed = time.time() - t0

        results.append({
            "config": i, "n_estimators": n_est, "max_depth": depth,
            "learning_rate": lr, "val_f1": round(f1, 4),
            "val_auc": round(auc, 4), "val_recall": round(rec, 4),
            "time_sec": round(elapsed, 1),
        })

        logger.info(
            f"   [{i}/{len(HP_SEARCH_SPACE)}] n={n_est} d={depth} lr={lr} "
            f"→ F1={f1:.3f}, AUC={auc:.3f}, Recall={rec:.3f}, time={elapsed:.0f}s"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_params = dict(
                n_estimators=n_est, max_depth=depth, learning_rate=lr,
                subsample=sub, colsample_bytree=col, min_child_weight=mcw,
                gamma=gamma,
            )

    pd.DataFrame(results).to_csv(REPORTS_DIR / "pipeline_B_hp_search_results.csv", index=False)
    logger.info(f"\n   🏆 Best F1: {best_f1:.4f}")
    logger.info(f"   🏆 Best config: {best_params}")
    return best_params


# ============================================================
# CROSS-VALIDATION
# ============================================================
def cross_validate_best(X, y, best_params, scale_pos_wt):
    logger.info(f"\n🔁 Time-series cross-validation ({CV_FOLDS} folds)...")
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    cv_f1, cv_auc = [], []

    for fold, (tr_idx, vl_idx) in enumerate(tscv.split(X), 1):
        model = XGBClassifier(
            **best_params, scale_pos_weight=scale_pos_wt,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", random_state=RANDOM_SEED, n_jobs=-1,
        )
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx], verbose=False)
        preds = model.predict(X.iloc[vl_idx])
        probs = model.predict_proba(X.iloc[vl_idx])[:, 1]

        cv_f1.append(f1_score(y.iloc[vl_idx], preds))
        cv_auc.append(roc_auc_score(y.iloc[vl_idx], probs))
        logger.info(f"   Fold {fold}: F1={cv_f1[-1]:.3f}, AUC={cv_auc[-1]:.3f}")

    logger.info(f"\n   CV F1 = {np.mean(cv_f1):.3f} ± {np.std(cv_f1):.3f}")
    return {"cv_f1_mean": float(np.mean(cv_f1)), "cv_f1_std": float(np.std(cv_f1)),
            "cv_auc_mean": float(np.mean(cv_auc))}


# ============================================================
# FINAL MODEL
# ============================================================
def train_final_model(X_train, y_train, X_val, y_val, best_params, scale_pos_wt):
    logger.info("\n🏋️ Training final rainfall classifier...")
    t0 = time.time()
    model = XGBClassifier(
        **best_params, scale_pos_weight=scale_pos_wt,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", random_state=RANDOM_SEED, n_jobs=-1,
        early_stopping_rounds=80,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    logger.info(f"   ✓ Trained in {time.time()-t0:.0f}s, best iter: {model.best_iteration}")
    return model


# ============================================================
# EVALUATION
# ============================================================
def evaluate_classifier(y_true, y_pred, y_prob, name="model"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    logloss = log_loss(y_true, y_prob)

    return {
        "name": name, "accuracy": float(acc), "precision": float(prec),
        "recall": float(rec), "f1_score": float(f1), "roc_auc": float(auc),
        "brier_score": float(brier), "log_loss": float(logloss),
    }


def baseline_majority(y_train, y_test):
    """Majority-class baseline: always predict 'no rain'."""
    logger.info("\n📏 Computing majority-class baseline...")
    majority = int(y_train.value_counts().idxmax())
    y_pred = np.full(len(y_test), majority)
    y_prob = np.full(len(y_test), y_train.mean())
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    logger.info(f"   ✓ Baseline (always '{majority}'): Acc={acc:.3f}, F1={f1:.3f}")
    return {"accuracy": float(acc), "f1": float(f1), "predictions": y_pred}


def statistical_significance_test(y_true, pred_model, pred_baseline):
    correct_model = (pred_model == y_true).astype(int)
    correct_baseline = (pred_baseline == y_true).astype(int)
    stat, pvalue = stats.mcnemar(
        [[((correct_model == 1) & (correct_baseline == 1)).sum(),
          ((correct_model == 1) & (correct_baseline == 0)).sum()],
         [((correct_model == 0) & (correct_baseline == 1)).sum(),
          ((correct_model == 0) & (correct_baseline == 0)).sum()]],
    ) if hasattr(stats, 'mcnemar') else (0, 0)
    # Fallback: use simple chi-squared approximation
    n01 = ((correct_model == 1) & (correct_baseline == 0)).sum()
    n10 = ((correct_model == 0) & (correct_baseline == 1)).sum()
    if n01 + n10 > 0:
        chi2 = (abs(n01 - n10) - 1)**2 / (n01 + n10)
        pvalue = 1 - stats.chi2.cdf(chi2, df=1)
    else:
        chi2, pvalue = 0, 1.0
    return {
        "mcnemar_chi2": float(chi2), "p_value": float(pvalue),
        "significant_at_0.01": bool(pvalue < 0.01),
    }


def per_city_evaluation(test_df, y_true, y_pred, y_prob):
    logger.info("\n🌍 Per-city evaluation...")
    results = []
    for city in sorted(test_df["city"].unique()):
        mask = test_df["city"].values == city
        if mask.sum() < 10 or y_true[mask].sum() == 0:
            continue
        results.append({
            "city": city,
            "test_rows": int(mask.sum()),
            "rain_pct": round(float(y_true[mask].mean() * 100), 2),
            "f1": round(float(f1_score(y_true[mask], y_pred[mask], zero_division=0)), 3),
            "auc": round(float(roc_auc_score(y_true[mask], y_prob[mask])), 3),
            "recall": round(float(recall_score(y_true[mask], y_pred[mask])), 3),
        })
    city_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    city_df.to_csv(REPORTS_DIR / "pipeline_B_per_city_performance.csv", index=False)
    logger.info(f"   ✓ Best:  {city_df.iloc[0]['city']} (F1={city_df.iloc[0]['f1']})")
    logger.info(f"   ✓ Worst: {city_df.iloc[-1]['city']} (F1={city_df.iloc[-1]['f1']})")
    return city_df


# ============================================================
# SHAP
# ============================================================
def run_shap_analysis(model, X_sample, feature_names):
    logger.info(f"\n🔬 SHAP analysis (sample={len(X_sample)})...")
    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    logger.info(f"   ✓ SHAP computed in {time.time()-t0:.0f}s")

    # Beeswarm
    plt.figure(figsize=(11, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      show=False, max_display=20)
    plt.title("Pipeline B — SHAP Feature Impact (Rainfall)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pipeline_B_shap_beeswarm.png")
    plt.close()
    logger.info("   ✓ Saved: pipeline_B_shap_beeswarm.png")

    # Bar
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      plot_type="bar", show=False, max_display=20)
    plt.title("Pipeline B — Top Features by Mean |SHAP|", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pipeline_B_shap_bar.png")
    plt.close()
    logger.info("   ✓ Saved: pipeline_B_shap_bar.png")

    # Waterfall
    plt.figure(figsize=(10, 8))
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value, shap_values[0], X_sample.iloc[0],
        feature_names=feature_names, max_display=15, show=False,
    )
    plt.title("Pipeline B — SHAP Waterfall (Single Prediction)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pipeline_B_shap_waterfall.png")
    plt.close()
    logger.info("   ✓ Saved: pipeline_B_shap_waterfall.png")

    # Top features table
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_features = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False).head(15).reset_index(drop=True)
    )
    top_features["mean_abs_shap"] = top_features["mean_abs_shap"].round(4)
    top_features.to_csv(REPORTS_DIR / "pipeline_B_top_features.csv", index=False)
    logger.info(f"   ✓ Top 5: {top_features['feature'].head(5).tolist()}")

    return top_features


# ============================================================
# LIME (for a few individual predictions)
# ============================================================
def run_lime_analysis(model, X_sample, feature_names):
    logger.info(f"\n🧪 LIME analysis ({LIME_SAMPLES} instances)...")
    try:
        import lime
        import lime.lime_tabular
    except ImportError:
        logger.warning("   ⚠ LIME not installed. Run: pip install lime")
        logger.warning("   Skipping LIME analysis.")
        return

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_sample.values, feature_names=feature_names,
        class_names=["No Rain", "Rain"], mode="classification",
        random_state=RANDOM_SEED,
    )

    for i in range(min(LIME_SAMPLES, len(X_sample))):
        exp = explainer.explain_instance(
            X_sample.iloc[i].values, model.predict_proba, num_features=10,
        )
        fig = exp.as_pyplot_figure()
        fig.set_size_inches(10, 6)
        plt.title(f"Pipeline B — LIME Explanation (Instance {i+1})", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"pipeline_B_lime_instance_{i+1}.png")
        plt.close()
        logger.info(f"   ✓ Saved: pipeline_B_lime_instance_{i+1}.png")


# ============================================================
# PLOTS
# ============================================================
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                xticklabels=["No Rain", "Rain"], yticklabels=["No Rain", "Rain"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Pipeline B — Confusion Matrix (Rainfall)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pipeline_B_confusion_matrix.png")
    plt.close()
    logger.info("   ✓ Saved: pipeline_B_confusion_matrix.png")


def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, "b-", lw=2, label=f"XGBoost (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "r--", lw=1, label="Random (AUC = 0.500)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Pipeline B — ROC Curve (Rainfall)", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pipeline_B_roc_curve.png")
    plt.close()
    logger.info("   ✓ Saved: pipeline_B_roc_curve.png")


def plot_calibration_curve(y_true, y_prob):
    fraction_pos, mean_predicted = calibration_curve(y_true, y_prob, n_bins=10)
    brier = brier_score_loss(y_true, y_prob)

    plt.figure(figsize=(8, 8))
    plt.plot(mean_predicted, fraction_pos, "bs-", lw=2, label=f"XGBoost (Brier={brier:.4f})")
    plt.plot([0, 1], [0, 1], "r--", lw=1, label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Pipeline B — Calibration Curve (Rainfall)", fontsize=13, fontweight="bold")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pipeline_B_calibration_curve.png")
    plt.close()
    logger.info("   ✓ Saved: pipeline_B_calibration_curve.png")


def plot_probability_distribution(y_true, y_prob):
    plt.figure(figsize=(10, 6))
    plt.hist(y_prob[y_true == 0], bins=50, alpha=0.6, label="No Rain", color="#2196F3", density=True)
    plt.hist(y_prob[y_true == 1], bins=50, alpha=0.6, label="Rain", color="#FF5722", density=True)
    plt.xlabel("Predicted Probability of Rain")
    plt.ylabel("Density")
    plt.title("Pipeline B — Predicted Probability Distribution", fontsize=13, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pipeline_B_probability_distribution.png")
    plt.close()
    logger.info("   ✓ Saved: pipeline_B_probability_distribution.png")


# ============================================================
# MAIN
# ============================================================
def main():
    print(r"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║   PIPELINE B — RAINFALL / HEAVY RAIN PREDICTION                ║
    ║   Model: XGBoost Binary Classifier                             ║
    ║   Uncertainty: Calibration Curve + Brier Score                 ║
    ║   Explainability: SHAP + LIME                                  ║
    ║   Mode: ULTRA-THOROUGH (publication quality)                   ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    total_start = time.time()

    # Load
    train, val, test = load_splits()
    features = get_feature_columns(train)
    logger.info(f"   ✓ {len(features)} features, target: {TARGET}")

    X_train, y_train = train[features], train[TARGET]
    X_val, y_val = val[features], val[TARGET]
    X_test, y_test = test[features], test[TARGET]

    # Class imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_wt = n_neg / n_pos
    logger.info(f"\n⚖️ Class balance: {n_neg:,} neg / {n_pos:,} pos "
                f"(ratio {n_neg/n_pos:.1f}:1)")
    logger.info(f"   scale_pos_weight = {scale_pos_wt:.2f}")

    # HP search
    best_params = hyperparameter_search(X_train, y_train, X_val, y_val, scale_pos_wt)

    # CV
    cv_summary = cross_validate_best(
        pd.concat([X_train, X_val]), pd.concat([y_train, y_val]),
        best_params, scale_pos_wt,
    )

    # Train final
    model = train_final_model(X_train, y_train, X_val, y_val, best_params, scale_pos_wt)

    # Evaluate on test
    logger.info("\n📊 Evaluating on test set (2019–2020)...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_classifier(y_test.values, y_pred, y_prob, "XGBoost (test)")
    logger.info(
        f"   ✓ Acc={metrics['accuracy']:.3f}, F1={metrics['f1_score']:.3f}, "
        f"AUC={metrics['roc_auc']:.3f}, Brier={metrics['brier_score']:.4f}"
    )

    # Baseline
    baseline = baseline_majority(y_train, y_test.values)
    sig = statistical_significance_test(y_test.values, y_pred, baseline["predictions"])
    logger.info(f"   ✓ McNemar p-value: {sig['p_value']:.2e} (significant: {sig['significant_at_0.01']})")

    # Per-city
    city_perf = per_city_evaluation(test, y_test.values, y_pred, y_prob)

    # SHAP
    X_shap = X_test.sample(SHAP_SAMPLE_SIZE, random_state=RANDOM_SEED)
    top_features = run_shap_analysis(model, X_shap, features)

    # LIME
    X_lime = X_test.sample(50, random_state=RANDOM_SEED)
    run_lime_analysis(model, X_lime, features)

    # Plots
    logger.info("\n🎨 Generating publication figures...")
    plot_confusion_matrix(y_test.values, y_pred)
    plot_roc_curve(y_test.values, y_prob)
    plot_calibration_curve(y_test.values, y_prob)
    plot_probability_distribution(y_test.values, y_prob)

    # Save model
    logger.info("\n💾 Saving model...")
    joblib.dump(model, MODELS_DIR / "pipeline_B_rainfall_xgb.pkl")
    logger.info(f"   ✓ Saved to {MODELS_DIR}")

    # Save metrics
    final_metrics = {
        "pipeline": "B",
        "task": "rainfall_prediction",
        "target": TARGET,
        "class_balance": {"negative": int(n_neg), "positive": int(n_pos),
                          "scale_pos_weight": round(scale_pos_wt, 2)},
        "best_hyperparameters": best_params,
        "cross_validation": cv_summary,
        "test_metrics": metrics,
        "baseline": baseline["accuracy"],
        "improvement_f1_over_baseline": round(metrics["f1_score"] - baseline["f1"], 4),
        "statistical_significance": sig,
        "top_5_features": top_features["feature"].head(5).tolist(),
    }
    with open(REPORTS_DIR / "pipeline_B_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    # Summary
    elapsed_min = (time.time() - total_start) / 60
    print(f"\n{'='*70}")
    print("  ✅ PIPELINE B — RAINFALL PREDICTION — COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time:             {elapsed_min:.1f} minutes")
    print(f"  Test Accuracy:          {metrics['accuracy']:.3f}")
    print(f"  Test Precision:         {metrics['precision']:.3f}")
    print(f"  Test Recall:            {metrics['recall']:.3f}")
    print(f"  Test F1:                {metrics['f1_score']:.3f}")
    print(f"  Test ROC-AUC:           {metrics['roc_auc']:.3f}")
    print(f"  Brier Score:            {metrics['brier_score']:.4f}")
    print(f"  Baseline Accuracy:      {baseline['accuracy']:.3f}")
    print(f"  Top feature:            {top_features.iloc[0]['feature']}")
    print(f"  Outputs:")
    print(f"    · Model:    {MODELS_DIR}")
    print(f"    · Reports:  {REPORTS_DIR}")
    print(f"    · Figures:  {FIGURES_DIR}")
    print(f"{'='*70}\n")

    return model, final_metrics


if __name__ == "__main__":
    model, metrics = main()
