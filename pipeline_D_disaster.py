#!/usr/bin/env python3
"""
==============================================================================
FILE 6: pipeline_D_disaster.py
==============================================================================
Project : AI-Driven Extreme Weather Prediction — A Global Perspective
Pipeline: D — Storm Risk / Disaster Classification (Multi-Class)
Author  : [Your Name]

Purpose:
    Multi-class disaster classification:
      Class 0 = Normal
      Class 1 = Heatwave
      Class 2 = Heavy Rain
      Class 3 = Storm
    Using XGBoost Multi-Class Classifier. Includes:
      • Hyperparameter tuning
      • Evaluation: Confusion Matrix, Weighted F1, per-class ROC
      • Uncertainty: Entropy-based confidence scoring
      • Explainability: SHAP multi-class force plots
      • Per-city and seasonal analysis

Outputs:
    models/pipeline_D_disaster_xgb.pkl
    reports/pipeline_D_metrics.json
    reports/figures/pipeline_D_*.png
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
    accuracy_score, f1_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
)
from sklearn.preprocessing import label_binarize
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

TARGET = "target_disaster"
CLASS_NAMES = {0: "Normal", 1: "Heatwave", 2: "Heavy Rain", 3: "Storm"}
RANDOM_SEED = 42
SHAP_SAMPLE_SIZE = 3000  # smaller for multi-class (4x memory)

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


def compute_sample_weights(y):
    """Compute per-sample weights to handle multi-class imbalance."""
    class_counts = y.value_counts()
    total = len(y)
    n_classes = len(class_counts)
    weights = {cls: total / (n_classes * count) for cls, count in class_counts.items()}
    logger.info("   Class weights:")
    for cls, w in sorted(weights.items()):
        logger.info(f"      {cls} ({CLASS_NAMES.get(cls, '?')}): {w:.2f} (n={class_counts[cls]:,})")
    return y.map(weights).values, weights


def hyperparameter_search(X_tr, y_tr, X_vl, y_vl, sw_train, sw_val):
    logger.info(f"\n🔍 Hyperparameter tuning ({len(HP_SEARCH_SPACE)} configs)...")
    best_f1, best_params = -1, None
    results = []

    for i, (ne, md, lr, ss, cs, mcw, gm) in enumerate(HP_SEARCH_SPACE, 1):
        t0 = time.time()
        m = XGBClassifier(
            n_estimators=ne, max_depth=md, learning_rate=lr, subsample=ss,
            colsample_bytree=cs, min_child_weight=mcw, gamma=gm,
            objective="multi:softprob", num_class=4, eval_metric="mlogloss",
            tree_method="hist", random_state=RANDOM_SEED, n_jobs=-1,
            early_stopping_rounds=50,
        )
        m.fit(X_tr, y_tr, sample_weight=sw_train,
              eval_set=[(X_vl, y_vl)], sample_weight_eval_set=[sw_val],
              verbose=False)

        yp = m.predict(X_vl)
        wf1 = f1_score(y_vl, yp, average="weighted")
        elapsed = time.time() - t0
        results.append({"config":i, "n_est":ne, "depth":md, "lr":lr,
                        "weighted_f1":round(wf1,4), "time":round(elapsed,1)})
        logger.info(f"   [{i}/{len(HP_SEARCH_SPACE)}] n={ne} d={md} lr={lr} → wF1={wf1:.3f}, time={elapsed:.0f}s")
        if wf1 > best_f1:
            best_f1 = wf1
            best_params = dict(n_estimators=ne, max_depth=md, learning_rate=lr,
                               subsample=ss, colsample_bytree=cs,
                               min_child_weight=mcw, gamma=gm)

    pd.DataFrame(results).to_csv(REPORTS_DIR/"pipeline_D_hp_search_results.csv", index=False)
    logger.info(f"\n   🏆 Best weighted F1: {best_f1:.4f}")
    return best_params


def cross_validate(X, y, params):
    logger.info(f"\n🔁 Time-series CV ({CV_FOLDS} folds)...")
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    cv_f1 = []
    for fold, (ti, vi) in enumerate(tscv.split(X), 1):
        sw, _ = compute_sample_weights(y.iloc[ti]) if fold == 1 else (y.iloc[ti].map(
            {cls: len(y.iloc[ti])/(4*cnt) for cls, cnt in y.iloc[ti].value_counts().items()}).values, None)
        m = XGBClassifier(**params, objective="multi:softprob", num_class=4,
                          eval_metric="mlogloss", tree_method="hist",
                          random_state=RANDOM_SEED, n_jobs=-1)
        m.fit(X.iloc[ti], y.iloc[ti], sample_weight=sw, verbose=False)
        p = m.predict(X.iloc[vi])
        cv_f1.append(f1_score(y.iloc[vi], p, average="weighted"))
        logger.info(f"   Fold {fold}: wF1={cv_f1[-1]:.3f}")
    logger.info(f"\n   CV wF1 = {np.mean(cv_f1):.3f} ± {np.std(cv_f1):.3f}")
    return {"cv_wf1_mean": float(np.mean(cv_f1)), "cv_wf1_std": float(np.std(cv_f1))}


def train_final(X_tr, y_tr, X_vl, y_vl, params, sw_tr, sw_vl):
    logger.info("\n🏋️ Training final disaster classifier...")
    t0 = time.time()
    m = XGBClassifier(**params, objective="multi:softprob", num_class=4,
                      eval_metric="mlogloss", tree_method="hist",
                      random_state=RANDOM_SEED, n_jobs=-1, early_stopping_rounds=80)
    m.fit(X_tr, y_tr, sample_weight=sw_tr,
          eval_set=[(X_vl, y_vl)], sample_weight_eval_set=[sw_vl], verbose=False)
    logger.info(f"   ✓ Trained in {time.time()-t0:.0f}s, best iter: {m.best_iteration}")
    return m


# ============================================================
# ENTROPY-BASED UNCERTAINTY
# ============================================================
def compute_entropy_confidence(probs):
    """
    Compute Shannon entropy of predicted class probabilities.
    Low entropy = high confidence, High entropy = uncertainty.
    Max entropy for 4 classes = log2(4) = 2.0 bits.
    """
    eps = 1e-10
    entropy = -np.sum(probs * np.log2(probs + eps), axis=1)
    max_entropy = np.log2(probs.shape[1])
    confidence = 1.0 - (entropy / max_entropy)  # 0=no confidence, 1=certain
    return entropy, confidence


# ============================================================
# EVALUATION
# ============================================================
def evaluate_multiclass(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    wf1 = f1_score(y_true, y_pred, average="weighted")
    mf1 = f1_score(y_true, y_pred, average="macro")

    # Per-class ROC-AUC (one-vs-rest)
    y_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
    per_class_auc = {}
    for cls in range(4):
        if y_bin[:, cls].sum() > 0:
            per_class_auc[CLASS_NAMES[cls]] = float(roc_auc_score(y_bin[:, cls], y_prob[:, cls]))
        else:
            per_class_auc[CLASS_NAMES[cls]] = "N/A"

    return {
        "accuracy": float(acc),
        "weighted_f1": float(wf1),
        "macro_f1": float(mf1),
        "per_class_auc": per_class_auc,
    }


def per_city_eval(test_df, y_true, y_pred):
    logger.info("\n🌍 Per-city evaluation...")
    results = []
    for city in sorted(test_df["city"].unique()):
        mask = test_df["city"].values == city
        if mask.sum() < 10: continue
        results.append({
            "city": city,
            "rows": int(mask.sum()),
            "weighted_f1": round(float(f1_score(y_true[mask], y_pred[mask], average="weighted")), 3),
            "accuracy": round(float(accuracy_score(y_true[mask], y_pred[mask])), 3),
        })
    df = pd.DataFrame(results).sort_values("weighted_f1", ascending=False)
    df.to_csv(REPORTS_DIR/"pipeline_D_per_city_performance.csv", index=False)
    logger.info(f"   ✓ Best:  {df.iloc[0]['city']} (wF1={df.iloc[0]['weighted_f1']})")
    logger.info(f"   ✓ Worst: {df.iloc[-1]['city']} (wF1={df.iloc[-1]['weighted_f1']})")
    return df


# ============================================================
# SHAP (multi-class)
# ============================================================
def run_shap(model, X_sample, feature_names):
    logger.info(f"\n🔬 SHAP analysis (sample={len(X_sample)}, multi-class)...")
    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    logger.info(f"   ✓ SHAP computed in {time.time()-t0:.0f}s")

    # shap_values is list of 4 arrays (one per class)
    if isinstance(shap_values, list) and len(shap_values) == 4:
        # Summary for each class
        for cls_id, cls_name in CLASS_NAMES.items():
            plt.figure(figsize=(11, 8))
            shap.summary_plot(shap_values[cls_id], X_sample, feature_names=feature_names,
                              show=False, max_display=15)
            plt.title(f"Pipeline D — SHAP for Class {cls_id} ({cls_name})", fontsize=13, fontweight="bold", pad=15)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR/f"pipeline_D_shap_class_{cls_id}_{cls_name.lower().replace(' ','_')}.png")
            plt.close()
            logger.info(f"   ✓ Saved: pipeline_D_shap_class_{cls_id}_{cls_name}.png")

        # Overall bar (mean across classes)
        mean_all = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        # Single array fallback
        mean_all = np.abs(shap_values).mean(axis=0) if shap_values.ndim == 2 else np.abs(shap_values).mean(axis=(0, 2))

    top = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_all}).sort_values("mean_abs_shap", ascending=False).head(15).reset_index(drop=True)
    top["mean_abs_shap"] = top["mean_abs_shap"].round(4)
    top.to_csv(REPORTS_DIR/"pipeline_D_top_features.csv", index=False)
    logger.info(f"   ✓ Top 5: {top['feature'].head(5).tolist()}")

    # Overall bar chart
    plt.figure(figsize=(10, 8))
    top.sort_values("mean_abs_shap").plot(kind="barh", x="feature", y="mean_abs_shap",
                                          legend=False, color="#4CAF50", edgecolor="black", ax=plt.gca())
    plt.xlabel("Mean |SHAP| (averaged across classes)")
    plt.title("Pipeline D — Top Features (Disaster Classification)", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(FIGURES_DIR/"pipeline_D_shap_bar_overall.png"); plt.close()
    logger.info("   ✓ Saved: pipeline_D_shap_bar_overall.png")

    return top


# ============================================================
# PLOTS
# ============================================================
def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
    labels = [CLASS_NAMES[i] for i in range(4)]
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="YlOrRd",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("Pipeline D — Confusion Matrix (Disaster)", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(FIGURES_DIR/"pipeline_D_confusion_matrix.png"); plt.close()
    logger.info("   ✓ Saved: pipeline_D_confusion_matrix.png")


def plot_per_class_roc(y_true, y_prob):
    y_bin = label_binarize(y_true, classes=[0,1,2,3])
    plt.figure(figsize=(9, 8))
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    for cls in range(4):
        if y_bin[:, cls].sum() == 0: continue
        fpr, tpr, _ = roc_curve(y_bin[:, cls], y_prob[:, cls])
        auc = roc_auc_score(y_bin[:, cls], y_prob[:, cls])
        plt.plot(fpr, tpr, color=colors[cls], lw=2,
                 label=f"{CLASS_NAMES[cls]} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],"k--",lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Pipeline D — Per-Class ROC Curves", fontsize=13, fontweight="bold")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIGURES_DIR/"pipeline_D_roc_per_class.png"); plt.close()
    logger.info("   ✓ Saved: pipeline_D_roc_per_class.png")


def plot_entropy_distribution(entropy, confidence, y_true, y_pred):
    correct = y_true == y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(entropy[correct], bins=50, alpha=0.6, label="Correct", color="#4CAF50", density=True)
    axes[0].hist(entropy[~correct], bins=50, alpha=0.6, label="Incorrect", color="#F44336", density=True)
    axes[0].set_xlabel("Shannon Entropy (bits)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Entropy: Correct vs Incorrect")
    axes[0].legend()

    axes[1].hist(confidence[correct], bins=50, alpha=0.6, label="Correct", color="#4CAF50", density=True)
    axes[1].hist(confidence[~correct], bins=50, alpha=0.6, label="Incorrect", color="#F44336", density=True)
    axes[1].set_xlabel("Confidence Score (0–1)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Confidence: Correct vs Incorrect")
    axes[1].legend()

    plt.suptitle("Pipeline D — Entropy-Based Uncertainty Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(FIGURES_DIR/"pipeline_D_entropy_uncertainty.png"); plt.close()
    logger.info("   ✓ Saved: pipeline_D_entropy_uncertainty.png")


def plot_class_distribution(y_train, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, y, title in [(axes[0], y_train, "Training"), (axes[1], y_test, "Test")]:
        counts = y.value_counts().sort_index()
        labels = [f"{CLASS_NAMES[i]}\n(n={counts.get(i,0):,})" for i in range(4)]
        colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
        ax.bar(labels, [counts.get(i,0) for i in range(4)], color=colors, edgecolor="black", alpha=0.85)
        ax.set_ylabel("Count"); ax.set_title(f"{title} Set Class Distribution")
    plt.suptitle("Pipeline D — Class Distribution", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(FIGURES_DIR/"pipeline_D_class_distribution.png"); plt.close()
    logger.info("   ✓ Saved: pipeline_D_class_distribution.png")


# ============================================================
# MAIN
# ============================================================
def main():
    print(r"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║   PIPELINE D — DISASTER CLASSIFICATION (MULTI-CLASS)           ║
    ║   Classes: Normal / Heatwave / Heavy Rain / Storm              ║
    ║   Model: XGBoost Multi-Class (softprob)                        ║
    ║   Uncertainty: Entropy-Based Confidence Scoring                ║
    ║   Explainability: SHAP multi-class analysis                    ║
    ║   Mode: ULTRA-THOROUGH (publication quality)                   ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    total_start = time.time()

    train, val, test = load_splits()
    features = get_feature_columns(train)

    X_tr, y_tr = train[features], train[TARGET]
    X_vl, y_vl = val[features], val[TARGET]
    X_te, y_te = test[features], test[TARGET]

    logger.info(f"\n📊 Target distribution (train):")
    for cls, name in CLASS_NAMES.items():
        n = (y_tr == cls).sum()
        logger.info(f"   {cls} ({name}): {n:,} ({n/len(y_tr)*100:.2f}%)")

    sw_train, class_weights = compute_sample_weights(y_tr)
    sw_val, _ = compute_sample_weights(y_vl)

    best_params = hyperparameter_search(X_tr, y_tr, X_vl, y_vl, sw_train, sw_val)
    cv = cross_validate(pd.concat([X_tr, X_vl]), pd.concat([y_tr, y_vl]), best_params)
    model = train_final(X_tr, y_tr, X_vl, y_vl, best_params, sw_train, sw_val)

    logger.info("\n📊 Evaluating on test set...")
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)
    metrics = evaluate_multiclass(y_te.values, y_pred, y_prob)
    logger.info(f"   ✓ Acc={metrics['accuracy']:.3f}, wF1={metrics['weighted_f1']:.3f}, mF1={metrics['macro_f1']:.3f}")
    logger.info(f"   ✓ Per-class AUC: {metrics['per_class_auc']}")

    # Classification report
    report = classification_report(y_te, y_pred, target_names=[CLASS_NAMES[i] for i in range(4)])
    logger.info(f"\n{report}")

    # Baseline
    logger.info("📏 Majority baseline...")
    bl_pred = np.zeros(len(y_te), dtype=int)
    bl_acc = accuracy_score(y_te, bl_pred)
    bl_wf1 = f1_score(y_te, bl_pred, average="weighted")
    logger.info(f"   ✓ Baseline: Acc={bl_acc:.3f}, wF1={bl_wf1:.3f}")

    # Entropy uncertainty
    logger.info("\n🎲 Computing entropy-based uncertainty...")
    entropy, confidence = compute_entropy_confidence(y_prob)
    logger.info(f"   ✓ Mean entropy: {entropy.mean():.3f} bits (max possible: 2.0)")
    logger.info(f"   ✓ Mean confidence: {confidence.mean():.3f}")
    correct_mask = y_te.values == y_pred
    logger.info(f"   ✓ Confidence when correct: {confidence[correct_mask].mean():.3f}")
    logger.info(f"   ✓ Confidence when wrong:   {confidence[~correct_mask].mean():.3f}")

    city_perf = per_city_eval(test, y_te.values, y_pred)

    X_shap = X_te.sample(min(SHAP_SAMPLE_SIZE, len(X_te)), random_state=RANDOM_SEED)
    top_features = run_shap(model, X_shap, features)

    logger.info("\n🎨 Generating figures...")
    plot_confusion(y_te.values, y_pred)
    plot_per_class_roc(y_te.values, y_prob)
    plot_entropy_distribution(entropy, confidence, y_te.values, y_pred)
    plot_class_distribution(y_tr, y_te)

    logger.info("\n💾 Saving model...")
    joblib.dump(model, MODELS_DIR/"pipeline_D_disaster_xgb.pkl")

    final = {
        "pipeline": "D", "task": "disaster_classification", "target": TARGET,
        "classes": CLASS_NAMES,
        "class_weights": {str(k): round(v, 2) for k, v in class_weights.items()},
        "best_params": best_params, "cv": cv, "test_metrics": metrics,
        "baseline": {"accuracy": bl_acc, "weighted_f1": bl_wf1},
        "entropy_uncertainty": {
            "mean_entropy": float(entropy.mean()),
            "mean_confidence": float(confidence.mean()),
            "confidence_when_correct": float(confidence[correct_mask].mean()),
            "confidence_when_wrong": float(confidence[~correct_mask].mean()) if (~correct_mask).sum() > 0 else "N/A",
        },
        "top_5_features": top_features["feature"].head(5).tolist(),
    }
    with open(REPORTS_DIR/"pipeline_D_metrics.json", "w") as f:
        json.dump(final, f, indent=2)

    elapsed = (time.time() - total_start) / 60
    print(f"\n{'='*70}")
    print("  ✅ PIPELINE D — DISASTER CLASSIFICATION — COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time:         {elapsed:.1f} minutes")
    print(f"  Test Accuracy:      {metrics['accuracy']:.3f}")
    print(f"  Weighted F1:        {metrics['weighted_f1']:.3f}")
    print(f"  Macro F1:           {metrics['macro_f1']:.3f}")
    print(f"  Mean Confidence:    {confidence.mean():.3f}")
    print(f"  Baseline wF1:       {bl_wf1:.3f}")
    print(f"  Top feature:        {top_features.iloc[0]['feature']}")
    print(f"  Outputs:")
    print(f"    · Model:   {MODELS_DIR}")
    print(f"    · Reports: {REPORTS_DIR}")
    print(f"    · Figures: {FIGURES_DIR}")
    print(f"{'='*70}\n")

    return model, final

if __name__ == "__main__":
    model, metrics = main()
