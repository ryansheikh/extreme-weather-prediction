#!/usr/bin/env python3
"""
==============================================================================
FILE 2: preprocessing_feature_engineering.py
==============================================================================
Project : AI-Driven Extreme Weather Prediction — A Global Perspective
Author  : [Your Name]
Date    : 2025

Purpose:
    Load master weather dataset, clean it, engineer time-series features,
    create four target variables using meteorological rules, and split data
    using a time-series-aware strategy for ML pipelines.

Pipeline Stages:
    1. Load & validate master CSV
    2. Data cleaning (duplicates, missing values, sorting)
    3. Feature engineering:
       - Lag features (24h, 48h, 72h)
       - Rolling features (24h, 7-day)
       - Time features (hour, month, season, year, weekend)
       - Coastal binary feature
    4. Target variable creation (REGIONAL HEATWAVE THRESHOLDS):
       - Rain:     precipitation > 1mm
       - Heatwave: temp >= 35°C (global) OR 40°C (South Asia)
       - Storm:    windspeed >= 40 km/h AND cloudcover >= 70%
       - Disaster: 0=Normal, 1=Heatwave, 2=Heavy Rain, 3=Storm
    5. Time-series-aware train/val/test split:
       - Training:   2009–2017 (9 years)
       - Validation: 2018 (1 year)
       - Testing:    2019–2020 (2 years)
       - Extended:   2021–2023 reserved for future real-world simulation

Output:
    - data/processed/train.csv
    - data/processed/val.csv
    - data/processed/test.csv
    - data/processed/feature_info.json
    - data/reports/target_distribution_report.csv
==============================================================================
"""

# ============================================================================
# 1. IMPORTS
# ============================================================================
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# ============================================================================
# 2. LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# 3. AUTO-DETECT DATA LOCATION
# ============================================================================
# Try multiple possible locations so the script works regardless of notebook CWD
possible_locations = [
    Path.cwd() / "data",
    Path.home() / "data",
    Path.home() / "Desktop" / "FYP 2026" / "data",
]

DATA_DIR = None
for loc in possible_locations:
    if (loc / "raw" / "master_weather_data.csv").exists():
        DATA_DIR = loc
        break

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find master_weather_data.csv. "
        "Expected in one of: " + ", ".join(str(p) for p in possible_locations)
    )

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORT_DIR = DATA_DIR / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"📁 Data directory:       {DATA_DIR}")
logger.info(f"📁 Processed output dir: {PROCESSED_DIR}")

# ============================================================================
# 4. CONFIGURATION
# ============================================================================

# Weather features used as model inputs
WEATHER_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "windspeed_10m",
    "surface_pressure",
    "cloudcover",
    "shortwave_radiation",
]

# Features that get lag/rolling transformations
LAG_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "windspeed_10m",
]

# Lag windows (hours)
LAG_WINDOWS = [24, 48, 72]

# Rolling window sizes (hours)
ROLLING_WINDOWS = {
    "24h": 24,
    "7d": 24 * 7,
}

# Target thresholds (meteorological rules)
RAIN_THRESHOLD_MM = 1.0              # precipitation > 1mm → rain
HEATWAVE_GLOBAL_THRESHOLD = 35.0     # international standard
HEATWAVE_SOUTH_ASIA_THRESHOLD = 40.0 # Pakistan/India/Bangladesh hot climate
STORM_WIND_THRESHOLD = 40.0          # km/h
STORM_CLOUD_THRESHOLD = 70.0         # %

# South Asian cities that use 40°C threshold (adapted to hot climate baseline)
SOUTH_ASIA_CITIES = {
    "Karachi",   # Pakistan
    "Delhi",     # India
    "Mumbai",    # India
    "Dhaka",     # Bangladesh
}

# Time-series split boundaries (YEARS)
TRAIN_START_YEAR = 2009
TRAIN_END_YEAR = 2017
VAL_YEAR = 2018
TEST_START_YEAR = 2019
TEST_END_YEAR = 2020
# 2021–2023 reserved for future real-world simulation

# ============================================================================
# 5. DATA LOADING
# ============================================================================

def load_master_data() -> pd.DataFrame:
    """Load the master weather CSV produced by File 1."""
    master_path = RAW_DIR / "master_weather_data.csv"
    logger.info(f"\n📥 Loading master dataset from {master_path}...")

    df = pd.read_csv(master_path, low_memory=False)

    # Parse datetime
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    # Drop any leftover columns we don't need
    if "visibility" in df.columns:
        df.drop(columns=["visibility"], inplace=True)

    logger.info(f"   ✓ Loaded {len(df):,} rows × {len(df.columns)} columns")
    logger.info(f"   ✓ Cities: {df['city'].nunique()}")
    logger.info(
        f"   ✓ Date range: {df['datetime'].min()} → {df['datetime'].max()}"
    )

    return df


# ============================================================================
# 6. DATA CLEANING
# ============================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset:
    - Remove duplicates
    - Sort by city + datetime
    - Forward-fill any remaining missing weather values
    - Reset index
    """
    logger.info("\n🧹 Cleaning data...")
    initial_rows = len(df)

    # Remove duplicates per (city, datetime)
    df = df.drop_duplicates(subset=["city", "datetime"], keep="first")

    # Sort chronologically within each city (critical for time-series)
    df = df.sort_values(["city", "datetime"]).reset_index(drop=True)

    # Forward-fill missing weather values within each city (rare edge cases)
    missing_before = df[WEATHER_FEATURES].isna().sum().sum()
    df[WEATHER_FEATURES] = (
        df.groupby("city")[WEATHER_FEATURES]
        .transform(lambda g: g.ffill().bfill())
    )
    missing_after = df[WEATHER_FEATURES].isna().sum().sum()

    logger.info(f"   ✓ Removed {initial_rows - len(df):,} duplicate rows")
    logger.info(f"   ✓ Filled {missing_before - missing_after:,} missing values")
    logger.info(f"   ✓ Final rows: {len(df):,}")

    return df


# ============================================================================
# 7. FEATURE ENGINEERING
# ============================================================================

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features from datetime column."""
    logger.info("\n⏰ Creating time features...")

    # Convert to local-equivalent naive datetime for feature extraction
    # (datetime is in UTC, so these are UTC-based features)
    dt = df["datetime"]

    df["hour"] = dt.dt.hour
    df["day"] = dt.dt.day
    df["day_of_week"] = dt.dt.dayofweek         # 0=Mon, 6=Sun
    df["month"] = dt.dt.month
    df["year"] = dt.dt.year
    df["day_of_year"] = dt.dt.dayofyear
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    # Meteorological season (hemisphere-adjusted)
    # Northern hemisphere default; we'll flip for southern hemisphere cities
    def northern_season(month):
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Autumn

    df["season"] = df["month"].apply(northern_season)

    # Flip season for Southern Hemisphere cities (latitude < 0)
    south_mask = df["latitude"] < 0
    df.loc[south_mask, "season"] = (df.loc[south_mask, "season"] + 2) % 4

    logger.info(
        f"   ✓ Added: hour, day, day_of_week, month, year, "
        f"day_of_year, is_weekend, season"
    )

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features — previous hour values of key weather variables.
    Lags must be computed per-city to prevent data leakage across cities.
    """
    logger.info("\n⏪ Creating lag features...")

    for var in LAG_FEATURES:
        for lag in LAG_WINDOWS:
            col_name = f"{var}_lag_{lag}h"
            df[col_name] = df.groupby("city")[var].shift(lag)

    n_lag_cols = len(LAG_FEATURES) * len(LAG_WINDOWS)
    logger.info(
        f"   ✓ Added {n_lag_cols} lag features "
        f"({LAG_WINDOWS} hours × {len(LAG_FEATURES)} variables)"
    )

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling mean and std features over 24h and 7-day windows.
    Rolling windows must be computed per-city.
    """
    logger.info("\n📉 Creating rolling window features...")

    for var in LAG_FEATURES:
        for window_name, window_hours in ROLLING_WINDOWS.items():
            grouped = df.groupby("city")[var]
            df[f"{var}_roll_{window_name}_mean"] = (
                grouped.transform(
                    lambda s: s.rolling(window_hours, min_periods=1).mean()
                )
            )
            df[f"{var}_roll_{window_name}_std"] = (
                grouped.transform(
                    lambda s: s.rolling(window_hours, min_periods=1).std()
                )
            )

    n_roll_cols = len(LAG_FEATURES) * len(ROLLING_WINDOWS) * 2
    logger.info(
        f"   ✓ Added {n_roll_cols} rolling features "
        f"(mean + std for {list(ROLLING_WINDOWS.keys())})"
    )

    return df


# ============================================================================
# 8. TARGET VARIABLE CREATION (REGIONAL HEATWAVE LOGIC)
# ============================================================================

def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create four target variables using meteorological rules:

    - target_rain      (binary):    precipitation > 1mm
    - target_heatwave  (binary):    regional temperature threshold
    - target_storm     (binary):    windspeed >= 40 km/h AND cloudcover >= 70%
    - target_disaster  (multi-class):
          0 = Normal
          1 = Heatwave
          2 = Heavy Rain
          3 = Storm

    Priority for multi-class (when multiple conditions apply):
        Storm > Heavy Rain > Heatwave > Normal
        (Storms are more severe; hierarchy prevents overlap bias.)
    """
    logger.info("\n🎯 Creating target variables...")

    # --- Target 1: Rain ---
    df["target_rain"] = (df["precipitation"] > RAIN_THRESHOLD_MM).astype(int)

    # --- Target 2: Heatwave (REGIONAL) ---
    df["heatwave_threshold"] = HEATWAVE_GLOBAL_THRESHOLD
    south_asia_mask = df["city"].isin(SOUTH_ASIA_CITIES)
    df.loc[south_asia_mask, "heatwave_threshold"] = HEATWAVE_SOUTH_ASIA_THRESHOLD

    df["target_heatwave"] = (
        df["temperature_2m"] >= df["heatwave_threshold"]
    ).astype(int)

    # --- Target 3: Storm ---
    df["target_storm"] = (
        (df["windspeed_10m"] >= STORM_WIND_THRESHOLD)
        & (df["cloudcover"] >= STORM_CLOUD_THRESHOLD)
    ).astype(int)

    # --- Target 4: Multi-class Disaster ---
    # Priority: Storm (3) > Heavy Rain (2) > Heatwave (1) > Normal (0)
    df["target_disaster"] = 0
    df.loc[df["target_heatwave"] == 1, "target_disaster"] = 1
    df.loc[df["target_rain"] == 1, "target_disaster"] = 2
    df.loc[df["target_storm"] == 1, "target_disaster"] = 3

    # Summary statistics
    logger.info("   ✓ Target variables created:")
    logger.info(
        f"      · Rain      : {df['target_rain'].sum():>10,} positive "
        f"({df['target_rain'].mean()*100:5.2f}%)"
    )
    logger.info(
        f"      · Heatwave  : {df['target_heatwave'].sum():>10,} positive "
        f"({df['target_heatwave'].mean()*100:5.2f}%)"
    )
    logger.info(
        f"      · Storm     : {df['target_storm'].sum():>10,} positive "
        f"({df['target_storm'].mean()*100:5.2f}%)"
    )

    disaster_dist = df["target_disaster"].value_counts().sort_index()
    class_labels = {0: "Normal", 1: "Heatwave", 2: "Heavy Rain", 3: "Storm"}
    logger.info("      · Disaster (multi-class):")
    for cls, count in disaster_dist.items():
        pct = count / len(df) * 100
        logger.info(
            f"           {cls} ({class_labels[cls]:>10s}): {count:>10,} "
            f"({pct:5.2f}%)"
        )

    return df


def save_target_distribution_report(df: pd.DataFrame) -> None:
    """Save per-city target statistics for review."""
    logger.info("\n📊 Saving target distribution report...")

    report_rows = []
    for city in sorted(df["city"].unique()):
        city_df = df[df["city"] == city]
        report_rows.append({
            "city": city,
            "country": city_df["country"].iloc[0],
            "continent": city_df["continent"].iloc[0],
            "coastal": bool(city_df["coastal"].iloc[0]),
            "heatwave_threshold_C": float(city_df["heatwave_threshold"].iloc[0]),
            "total_hours": len(city_df),
            "rain_hours": int(city_df["target_rain"].sum()),
            "rain_pct": round(city_df["target_rain"].mean() * 100, 2),
            "heatwave_hours": int(city_df["target_heatwave"].sum()),
            "heatwave_pct": round(city_df["target_heatwave"].mean() * 100, 2),
            "storm_hours": int(city_df["target_storm"].sum()),
            "storm_pct": round(city_df["target_storm"].mean() * 100, 2),
            "disaster_normal_pct": round(
                (city_df["target_disaster"] == 0).mean() * 100, 2
            ),
            "disaster_heatwave_pct": round(
                (city_df["target_disaster"] == 1).mean() * 100, 2
            ),
            "disaster_rain_pct": round(
                (city_df["target_disaster"] == 2).mean() * 100, 2
            ),
            "disaster_storm_pct": round(
                (city_df["target_disaster"] == 3).mean() * 100, 2
            ),
        })

    report_df = pd.DataFrame(report_rows)
    report_path = REPORT_DIR / "target_distribution_report.csv"
    report_df.to_csv(report_path, index=False)
    logger.info(f"   ✓ Report saved: {report_path}")


# ============================================================================
# 9. ENCODING
# ============================================================================

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical columns to numeric codes for ML models.
    Creates a city ID mapping (small but important for cross-city modeling).
    """
    logger.info("\n🔢 Encoding categorical features...")

    # City as integer ID (keeps original name for analysis)
    df["city_id"] = df["city"].astype("category").cat.codes

    # Continent as integer ID
    df["continent_id"] = df["continent"].astype("category").cat.codes

    # Climate zone as integer ID
    df["climate_zone_id"] = df["climate_zone"].astype("category").cat.codes

    logger.info("   ✓ Added: city_id, continent_id, climate_zone_id")
    return df


# ============================================================================
# 10. TIME-SERIES SPLIT
# ============================================================================

def time_series_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform time-series-aware train/val/test split.
    NO random shuffling — splits are based on year boundaries.

    Training   : 2009–2017 (9 years)
    Validation : 2018 (1 year)
    Testing    : 2019–2020 (2 years)

    2021–2023 is excluded from standard splits and reserved for extended
    out-of-sample evaluation / real-world simulation in Streamlit dashboard.
    """
    logger.info("\n📅 Performing time-series-aware split...")

    year = df["datetime"].dt.year

    train_df = df[year.between(TRAIN_START_YEAR, TRAIN_END_YEAR)].copy()
    val_df = df[year == VAL_YEAR].copy()
    test_df = df[year.between(TEST_START_YEAR, TEST_END_YEAR)].copy()

    logger.info(
        f"   ✓ Training  : {len(train_df):>10,} rows "
        f"({TRAIN_START_YEAR}–{TRAIN_END_YEAR})"
    )
    logger.info(
        f"   ✓ Validation: {len(val_df):>10,} rows ({VAL_YEAR})"
    )
    logger.info(
        f"   ✓ Testing   : {len(test_df):>10,} rows "
        f"({TEST_START_YEAR}–{TEST_END_YEAR})"
    )

    # Also report retained extended range
    extended_df = df[year >= 2021]
    logger.info(
        f"   ℹ Extended  : {len(extended_df):>10,} rows (2021–2023, reserved)"
    )

    return train_df, val_df, test_df


# ============================================================================
# 11. SAVE OUTPUTS
# ============================================================================

def save_splits(train_df, val_df, test_df) -> None:
    """Save train/val/test splits as compressed CSVs for fast loading."""
    logger.info("\n💾 Saving processed datasets...")

    for name, df_split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = PROCESSED_DIR / f"{name}.csv"
        df_split.to_csv(path, index=False)
        size_mb = path.stat().st_size / (1024**2)
        logger.info(
            f"   ✓ {name:>5s}.csv: {len(df_split):>10,} rows, {size_mb:6.1f} MB"
        )


def save_feature_info(df: pd.DataFrame) -> None:
    """Save metadata about features for downstream pipelines to read."""
    logger.info("\n📋 Saving feature metadata...")

    # Identify feature types
    target_cols = [c for c in df.columns if c.startswith("target_")]
    weather_cols = WEATHER_FEATURES
    lag_cols = [c for c in df.columns if "_lag_" in c]
    roll_cols = [c for c in df.columns if "_roll_" in c]
    time_cols = [
        "hour", "day", "day_of_week", "month", "year",
        "day_of_year", "is_weekend", "season",
    ]
    encoded_cols = ["city_id", "continent_id", "climate_zone_id"]
    geo_cols = ["latitude", "longitude", "coastal"]

    # Features used as model INPUTS (everything except targets + raw metadata)
    excluded = set(target_cols + [
        "datetime", "city", "country", "continent", "climate_zone",
        "heatwave_threshold",
    ])
    input_features = [c for c in df.columns if c not in excluded]

    info = {
        "total_rows": len(df),
        "total_features": len(df.columns),
        "feature_groups": {
            "weather_raw": weather_cols,
            "lag_features": lag_cols,
            "rolling_features": roll_cols,
            "time_features": time_cols,
            "encoded_features": encoded_cols,
            "geographic_features": geo_cols,
        },
        "targets": {
            "regression": ["temperature_2m"],
            "binary_classification": [
                "target_rain", "target_heatwave", "target_storm",
            ],
            "multiclass_classification": {
                "column": "target_disaster",
                "classes": {
                    "0": "Normal",
                    "1": "Heatwave",
                    "2": "Heavy Rain",
                    "3": "Storm",
                },
            },
        },
        "model_input_features": input_features,
        "thresholds": {
            "rain_mm": RAIN_THRESHOLD_MM,
            "heatwave_global_C": HEATWAVE_GLOBAL_THRESHOLD,
            "heatwave_south_asia_C": HEATWAVE_SOUTH_ASIA_THRESHOLD,
            "south_asia_cities": sorted(SOUTH_ASIA_CITIES),
            "storm_wind_kmh": STORM_WIND_THRESHOLD,
            "storm_cloud_pct": STORM_CLOUD_THRESHOLD,
        },
        "split_config": {
            "train_years": f"{TRAIN_START_YEAR}-{TRAIN_END_YEAR}",
            "validation_year": VAL_YEAR,
            "test_years": f"{TEST_START_YEAR}-{TEST_END_YEAR}",
            "extended_reserved": "2021-2023",
        },
    }

    info_path = PROCESSED_DIR / "feature_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    logger.info(f"   ✓ Saved: {info_path}")


# ============================================================================
# 12. MAIN PIPELINE
# ============================================================================

def main():
    print(r"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║   FILE 2 — Preprocessing & Feature Engineering                 ║
    ║   20 Global Cities · Regional Heatwave Thresholds              ║
    ║   Target: train/val/test sets ready for ML modeling            ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # ── Step 1: Load ─────────────────────────────────────────────────
    df = load_master_data()

    # ── Step 2: Clean ────────────────────────────────────────────────
    df = clean_data(df)

    # ── Step 3: Feature engineering ──────────────────────────────────
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    # ── Step 4: Create targets ───────────────────────────────────────
    df = create_targets(df)
    save_target_distribution_report(df)

    # ── Step 5: Encode categoricals ──────────────────────────────────
    df = encode_categorical_features(df)

    # ── Step 6: Drop rows that lack lag history ──────────────────────
    # First 72 hours per city have NaN lag values; drop them
    before = len(df)
    df = df.dropna(subset=[f"{v}_lag_72h" for v in LAG_FEATURES]).reset_index(drop=True)
    logger.info(
        f"\n🗑  Dropped {before - len(df):,} rows with incomplete lag history"
    )

    # ── Step 7: Time-series split ────────────────────────────────────
    train_df, val_df, test_df = time_series_split(df)

    # ── Step 8: Save outputs ─────────────────────────────────────────
    save_splits(train_df, val_df, test_df)
    save_feature_info(df)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  ✅ PREPROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"  Total rows processed:      {len(df):,}")
    print(f"  Total features created:    {len(df.columns)}")
    print(f"  Training rows:             {len(train_df):,}")
    print(f"  Validation rows:           {len(val_df):,}")
    print(f"  Test rows:                 {len(test_df):,}")
    print(f"  Files saved to:            {PROCESSED_DIR}")
    print(f"{'='*70}\n")

    return df, train_df, val_df, test_df


# ============================================================================
# 13. ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    df, train_df, val_df, test_df = main()
