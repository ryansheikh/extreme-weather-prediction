#!/usr/bin/env python3
"""
==============================================================================
FILE 5: pipeline_C_heatwave.py
==============================================================================
Project : AI-Driven Extreme Weather Prediction — A Global Perspective
Pipeline: C — Heatwave Detection (Binary Classification)
Author  : [Your Name]

Purpose:
    Detect heatwave events (regional thresholds: 35°C global, 40°C South Asia)
    using XGBoost binary classifier. Includes:
      • Class imbalance handling
      • Hyperparameter tuning
      • Evaluation: Accuracy, Precision, Recall, F1
      • Uncertainty: Probability calibration
      • Explainability: SHAP beeswarm + waterfall plots
      • Per-city and seasonal analysis

Outputs:
    models/pipeline_C_heatwave_xgb.pkl
    reports/pipeline_C_metrics.json
    reports/figures/pipeline_C_*.png
==============================================================================
"""

import json, time, logging, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, shap

from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, brier_score_loss, log_loss,
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from scipy import stats

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight", "figure.figsize": (10, 6)})

# Paths
possible = [Path.cwd()/"data", Path.home()/"data", Path.home()/"Desktop"/"FYP 2026"/"data"]
DATA_DIR = next((p for p in possible if (p/"processed"/"train.csv").exists()), None)
if DATA_DIR is None: raise FileNotFoundError("Cannot find data/processed/")
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR.parent / "models"
REPORTS_DIR = DATA_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
for d in [MODELS_DIR, REPORTS_DIR, FIGURES_DIR]: d.mkdir(parents=True, exist_ok=True)

TARGET = "target_heatwave"
RANDOM_SEED = 42
SHAP_SAMPLE_SIZE = 5000

HP_SEARCH_SPACE = [
    (400, 5, 0.05, 0.9, 0.9, 3, 0.0),
    (600, 6, 0.05, 0.85, 0.85, 3, 0.1),
    (800, 6, 0.03, 0.85, 0.85, 5, 0.1),
    (1000, 7, 0.03, 0.8, 0.8, 5, 0.2),
    (1200, 7, 0.02, 0.8, 0.8, 5, 0.2),
    (1500, 8, 0.02, 0.75, 0.75, 7, 0.3),
]
CV_FOLDS = 5


def load_splits():
    logger.info("\n📥 Loading preprocessed datasets...")
    train = pd.read_csv(PROCESSED_DIR/"train.csv", low_memory=False)
    val = pd.read_csv(PROCESSED_DIR/"val.csv", low_memory=False)
    test = pd.read_csv(PROCESSED_DIR/"test.csv", low_memory=False)
    for df in [train, val, test]:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df.sort_values(["city", "datetime"], inplace=True)
    logger.info(f"   ✓ train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")
    return train, val, test


def get_feature_columns(df):
    exclude = {"datetime","city","country","continent","climate_zone","heatwave_threshold",
               "target_rain","target_heatwave","target_storm","target_disaster","target_temperature_next"}
    return [c for c in df.columns if c not in exclude]


def hyperparameter_search(X_train, y_train, X_val, y_val, spw):
    logger.info(f"\n🔍 Hyperparameter tuning ({len(HP_SEARCH_SPACE)} configs)...")
    best_f1, best_params = -1, None
    results = []
    for i, (ne, md, lr, ss, cs, mcw, gm) in enumerate(HP_SEARCH_SPACE, 1):
        t0 = time.time()
        m = XGBClassifier(n_estimators=ne, max_depth=md, learning_rate=lr, subsample=ss,
                          colsample_bytree=cs, min_child_weight=mcw, gamma=gm,
                          scale_pos_weight=spw, objective="binary:logistic",
                          eval_metric="logloss", tree_method="hist",
                          random_state=RANDOM_SEED, n_jobs=-1, early_stopping_rounds=50)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        yp = m.predict(X_val)
        yprob = m.predict_proba(X_val)[:, 1]
        f1 = f1_score(y_val, yp)
        auc = roc_auc_score(y_val, yprob) if y_val.sum() > 0 else 0.0
        elapsed = time.time() - t0
        results.append({"config":i, "n_est":ne, "depth":md, "lr":lr, "f1":round(f1,4), "auc":round(auc,4), "time":round(elapsed,1)})
        logger.info(f"   [{i}/{len(HP_SEARCH_SPACE)}] n={ne} d={md} lr={lr} → F1={f1:.3f}, AUC={auc:.3f}, time={elapsed:.0f}s")
        if f1 > best_f1:
            best_f1 = f1
            best_params = dict(n_estimators=ne, max_depth=md, learning_rate=lr, subsample=ss,
                               colsample_bytree=cs, min_child_weight=mcw, gamma=gm)
    pd.DataFrame(results).to_csv(REPORTS_DIR/"pipeline_C_hp_search_results.csv", index=False)
    logger.info(f"\n   🏆 Best F1: {best_f1:.4f}")
    return best_params


def cross_validate(X, y, params, spw):
    logger.info(f"\n🔁 Time-series CV ({CV_FOLDS} folds)...")
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    cv_f1, cv_auc = [], []
    for fold, (ti, vi) in enumerate(tscv.split(X), 1):
        m = XGBClassifier(**params, scale_pos_weight=spw, objective="binary:logistic",
                          eval_metric="logloss", tree_method="hist", random_state=RANDOM_SEED, n_jobs=-1)
        m.fit(X.iloc[ti], y.iloc[ti], verbose=False)
        p = m.predict(X.iloc[vi])
        pr = m.predict_proba(X.iloc[vi])[:, 1]
        cv_f1.append(f1_score(y.iloc[vi], p))
        cv_auc.append(roc_auc_score(y.iloc[vi], pr) if y.iloc[vi].sum() > 0 else 0.0)
        logger.info(f"   Fold {fold}: F1={cv_f1[-1]:.3f}, AUC={cv_auc[-1]:.3f}")
    logger.info(f"\n   CV F1 = {np.mean(cv_f1):.3f} ± {np.std(cv_f1):.3f}")
    return {"cv_f1_mean": float(np.mean(cv_f1)), "cv_f1_std": float(np.std(cv_f1)), "cv_auc_mean": float(np.mean(cv_auc))}


def train_final(X_tr, y_tr, X_vl, y_vl, params, spw):
    logger.info("\n🏋️ Training final heatwave classifier...")
    t0 = time.time()
    m = XGBClassifier(**params, scale_pos_weight=spw, objective="binary:logistic",
                      eval_metric="logloss", tree_method="hist", random_state=RANDOM_SEED,
                      n_jobs=-1, early_stopping_rounds=80)
    m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
    logger.info(f"   ✓ Trained in {time.time()-t0:.0f}s, best iter: {m.best_iteration}")
    return m


def evaluate(y_true, y_pred, y_prob, name="model"):
    return {
        "name": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if y_true.sum() > 0 else 0.0,
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


def per_city_eval(test_df, y_true, y_pred, y_prob):
    logger.info("\n🌍 Per-city evaluation...")
    results = []
    for city in sorted(test_df["city"].unique()):
        mask = test_df["city"].values == city
        if mask.sum() < 10: continue
        yt, yp, ypr = y_true[mask], y_pred[mask], y_prob[mask]
        hw_pct = float(yt.mean() * 100)
        results.append({
            "city": city, "test_rows": int(mask.sum()),
            "heatwave_pct": round(hw_pct, 2),
            "f1": round(float(f1_score(yt, yp, zero_division=0)), 3),
            "recall": round(float(recall_score(yt, yp, zero_division=0)), 3),
            "auc": round(float(roc_auc_score(yt, ypr)), 3) if yt.sum() > 0 else "N/A",
        })
    city_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    city_df.to_csv(REPORTS_DIR / "pipeline_C_per_city_performance.csv", index=False)
    has_hw = city_df[city_df["heatwave_pct"] > 0]
    no_hw = city_df[city_df["heatwave_pct"] == 0]
    if not has_hw.empty:
        logger.info(f"   ✓ Cities with heatwaves: {len(has_hw)}")
        logger.info(f"   ✓ Best (with events): {has_hw.iloc[0]['city']} (F1={has_hw.iloc[0]['f1']})")
    if not no_hw.empty:
        logger.info(f"   ✓ Cities without heatwaves: {len(no_hw)} (e.g. {no_hw.iloc[0]['city']})")
    return city_df


def run_shap(model, X_sample, feature_names):
    logger.info(f"\n🔬 SHAP analysis (sample={len(X_sample)})...")
    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample)
    logger.info(f"   ✓ SHAP computed in {time.time()-t0:.0f}s")

    # Beeswarm
    plt.figure(figsize=(11, 8))
    shap.summary_plot(sv, X_sample, feature_names=feature_names, show=False, max_display=20)
    plt.title("Pipeline C — SHAP Feature Impact (Heatwave)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout(); plt.savefig(FIGURES_DIR/"pipeline_C_shap_beeswarm.png"); plt.close()
    logger.info("   ✓ Saved: pipeline_C_shap_beeswarm.png")

    # Bar
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_sample, feature_names=feature_names, plot_type="bar", show=False, max_display=20)
    plt.title("Pipeline C — Top Features (Heatwave)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout(); plt.savefig(FIGURES_DIR/"pipeline_C_shap_bar.png"); plt.close()
    logger.info("   ✓ Saved: pipeline_C_shap_bar.png")

    # Waterfall
    plt.figure(figsize=(10, 8))
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, sv[0], X_sample.iloc[0],
                                            feature_names=feature_names, max_display=15, show=False)
    plt.title("Pipeline C — SHAP Waterfall (Heatwave)", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(FIGURES_DIR/"pipeline_C_shap_waterfall.png"); plt.close()
    logger.info("   ✓ Saved: pipeline_C_shap_waterfall.png")

    # Top features
    mean_abs = np.abs(sv).mean(axis=0)
    top = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False).head(15).reset_index(drop=True)
    top["mean_abs_shap"] = top["mean_abs_shap"].round(4)
    top.to_csv(REPORTS_DIR/"pipeline_C_top_features.csv", index=False)
    logger.info(f"   ✓ Top 5: {top['feature'].head(5).tolist()}")
    return top


def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Oranges", xticklabels=["No Heatwave","Heatwave"], yticklabels=["No Heatwave","Heatwave"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("Pipeline C — Confusion Matrix (Heatwave)", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(FIGURES_DIR/"pipeline_C_confusion_matrix.png"); plt.close()
    logger.info("   ✓ Saved: pipeline_C_confusion_matrix.png")


def plot_roc(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, "r-", lw=2, label=f"XGBoost (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],"k--",lw=1, label="Random")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("Pipeline C — ROC Curve (Heatwave)", fontsize=13, fontweight="bold")
    plt.legend(); plt.tight_layout(); plt.savefig(FIGURES_DIR/"pipeline_C_roc_curve.png"); plt.close()
    logger.info("   ✓ Saved: pipeline_C_roc_curve.png")


def plot_calibration(y_true, y_prob):
    frac, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
    brier = brier_score_loss(y_true, y_prob)
    plt.figure(figsize=(8, 8))
    plt.plot(mean_pred, frac, "rs-", lw=2, label=f"XGBoost (Brier={brier:.4f})")
    plt.plot([0,1],[0,1],"k--",lw=1, label="Perfect")
    plt.xlabel("Mean Predicted Probability"); plt.ylabel("Fraction of Positives")
    plt.title("Pipeline C — Calibration Curve (Heatwave)", fontsize=13, fontweight="bold")
    plt.legend(); plt.tight_layout(); plt.savefig(FIGURES_DIR/"pipeline_C_calibration_curve.png"); plt.close()
    logger.info("   ✓ Saved: pipeline_C_calibration_curve.png")


def plot_heatwave_by_city(test_df, y_true):
    hw_counts = test_df.loc[y_true == 1, "city"].value_counts().sort_values(ascending=True)
    if len(hw_counts) == 0: return
    plt.figure(figsize=(10, 8))
    hw_counts.plot(kind="barh", color="#FF5722", edgecolor="black", alpha=0.85)
    plt.xlabel("Number of Heatwave Hours (Test Set)")
    plt.title("Pipeline C — Heatwave Events by City (2019–2020)", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(FIGURES_DIR/"pipeline_C_heatwave_by_city.png"); plt.close()
    logger.info("   ✓ Saved: pipeline_C_heatwave_by_city.png")


def main():
    print(r"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║   PIPELINE C — HEATWAVE DETECTION                              ║
    ║   Model: XGBoost Binary Classifier                             ║
    ║   Uncertainty: Probability Calibration                         ║
    ║   Explainability: SHAP beeswarm + waterfall                    ║
    ║   Mode: ULTRA-THOROUGH (publication quality)                   ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    total_start = time.time()

    train, val, test = load_splits()
    features = get_feature_columns(train)

    X_tr, y_tr = train[features], train[TARGET]
    X_vl, y_vl = val[features], val[TARGET]
    X_te, y_te = test[features], test[TARGET]

    n_neg, n_pos = (y_tr==0).sum(), (y_tr==1).sum()
    spw = n_neg / max(n_pos, 1)
    logger.info(f"\n⚖️ Class balance: {n_neg:,} neg / {n_pos:,} pos (ratio {n_neg/max(n_pos,1):.1f}:1)")
    logger.info(f"   scale_pos_weight = {spw:.2f}")

    best_params = hyperparameter_search(X_tr, y_tr, X_vl, y_vl, spw)
    cv = cross_validate(pd.concat([X_tr, X_vl]), pd.concat([y_tr, y_vl]), best_params, spw)
    model = train_final(X_tr, y_tr, X_vl, y_vl, best_params, spw)

    logger.info("\n📊 Evaluating on test set...")
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    metrics = evaluate(y_te.values, y_pred, y_prob, "XGBoost (test)")
    logger.info(f"   ✓ Acc={metrics['accuracy']:.3f}, F1={metrics['f1_score']:.3f}, AUC={metrics['roc_auc']:.3f}")

    # Baseline
    logger.info("\n📏 Majority baseline...")
    bl_pred = np.zeros(len(y_te))
    bl_acc = accuracy_score(y_te, bl_pred)
    bl_f1 = f1_score(y_te, bl_pred, zero_division=0)
    logger.info(f"   ✓ Baseline: Acc={bl_acc:.3f}, F1={bl_f1:.3f}")

    city_perf = per_city_eval(test, y_te.values, y_pred, y_prob)
    X_shap = X_te.sample(min(SHAP_SAMPLE_SIZE, len(X_te)), random_state=RANDOM_SEED)
    top_features = run_shap(model, X_shap, features)

    logger.info("\n🎨 Generating figures...")
    plot_confusion(y_te.values, y_pred)
    plot_roc(y_te.values, y_prob)
    plot_calibration(y_te.values, y_prob)
    plot_heatwave_by_city(test, y_te.values)

    logger.info("\n💾 Saving model...")
    joblib.dump(model, MODELS_DIR/"pipeline_C_heatwave_xgb.pkl")

    final = {
        "pipeline": "C", "task": "heatwave_detection", "target": TARGET,
        "class_balance": {"neg": int(n_neg), "pos": int(n_pos), "spw": round(spw, 2)},
        "best_params": best_params, "cv": cv, "test_metrics": metrics,
        "baseline": {"accuracy": bl_acc, "f1": bl_f1},
        "top_5_features": top_features["feature"].head(5).tolist(),
    }
    with open(REPORTS_DIR/"pipeline_C_metrics.json", "w") as f:
        json.dump(final, f, indent=2)

    elapsed = (time.time() - total_start) / 60
    print(f"\n{'='*70}")
    print("  ✅ PIPELINE C — HEATWAVE DETECTION — COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time:        {elapsed:.1f} minutes")
    print(f"  Test Accuracy:     {metrics['accuracy']:.3f}")
    print(f"  Test Precision:    {metrics['precision']:.3f}")
    print(f"  Test Recall:       {metrics['recall']:.3f}")
    print(f"  Test F1:           {metrics['f1_score']:.3f}")
    print(f"  Test ROC-AUC:      {metrics['roc_auc']:.3f}")
    print(f"  Brier Score:       {metrics['brier_score']:.4f}")
    print(f"  Top feature:       {top_features.iloc[0]['feature']}")
    print(f"{'='*70}\n")

    return model, final

if __name__ == "__main__":
    model, metrics = main()
