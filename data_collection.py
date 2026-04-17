#!/usr/bin/env python3
"""
==============================================================================
FILE 1: data_collection.py
==============================================================================
Project : AI-Driven Extreme Weather Prediction — A Global Perspective
Author  : [Your Name]
Date    : 2025

Purpose:
    Fetch hourly historical weather data from the Open-Meteo Archive API
    for 20 global cities (10 coastal + 10 non-coastal) across 6 continents.
    Data covers 2009-01-01 to 2023-12-31 (~15 years, hourly resolution).

Data Source:
    Open-Meteo Historical Weather API (free, no API key required)
    https://open-meteo.com/en/docs/historical-weather-api

Features Collected:
    - temperature_2m         (°C)
    - relative_humidity_2m   (%)
    - precipitation          (mm)
    - windspeed_10m          (km/h)
    - surface_pressure       (hPa)
    - cloudcover             (%)
    - shortwave_radiation    (W/m²)  — UV proxy
    - visibility             (m)     — available via hourly params

Output:
    - Individual CSV per city  → data/raw/{city_name}.csv
    - Merged master CSV        → data/raw/master_weather_data.csv
    - Data quality report      → data/reports/data_quality_report.csv
==============================================================================
"""

# ============================================================================
# 1. IMPORTS
# ============================================================================
import os
import time
import json
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd
import numpy as np

# ============================================================================
# 2. LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# 3. PROJECT DIRECTORY SETUP
# ============================================================================
# Create output directories
RAW_DIR = Path("data/raw")
REPORT_DIR = Path("data/reports")
CACHE_DIR = Path("data/cache")

for d in [RAW_DIR, REPORT_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directory ready: {d}")

# ============================================================================
# 4. CITY METADATA REGISTRY
# ============================================================================
# Each city includes: name, country, continent, lat, lon, coastal flag,
# and a brief note on why it was selected (climate diversity).

CITIES = {
    # ── COASTAL CITIES (10) ──────────────────────────────────────────────
    "Mumbai": {
        "country": "India",
        "continent": "Asia",
        "latitude": 19.0760,
        "longitude": 72.8777,
        "coastal": True,
        "climate_zone": "Tropical monsoon",
        "note": "Severe monsoon flooding, cyclone exposure",
    },
    "Miami": {
        "country": "USA",
        "continent": "North America",
        "latitude": 25.7617,
        "longitude": -80.1918,
        "coastal": True,
        "climate_zone": "Tropical monsoon",
        "note": "Hurricane corridor, sea-level rise hotspot",
    },
    "Lagos": {
        "country": "Nigeria",
        "continent": "Africa",
        "latitude": 6.5244,
        "longitude": 3.3792,
        "coastal": True,
        "climate_zone": "Tropical savanna",
        "note": "Rapid urbanisation, coastal flooding",
    },
    "Jakarta": {
        "country": "Indonesia",
        "continent": "Asia",
        "latitude": -6.2088,
        "longitude": 106.8456,
        "coastal": True,
        "climate_zone": "Tropical rainforest",
        "note": "Land subsidence, extreme rainfall",
    },
    "Sydney": {
        "country": "Australia",
        "continent": "Oceania",
        "latitude": -33.8688,
        "longitude": 151.2093,
        "coastal": True,
        "climate_zone": "Humid subtropical",
        "note": "Bushfire risk, drought cycles",
    },
    "Rotterdam": {
        "country": "Netherlands",
        "continent": "Europe",
        "latitude": 51.9244,
        "longitude": 4.4777,
        "coastal": True,
        "climate_zone": "Oceanic",
        "note": "Below sea level, advanced flood infrastructure",
    },
    "Cape_Town": {
        "country": "South Africa",
        "continent": "Africa",
        "latitude": -33.9249,
        "longitude": 18.4241,
        "coastal": True,
        "climate_zone": "Mediterranean",
        "note": "Water scarcity, Day Zero crisis (2018)",
    },
    "Dhaka": {
        "country": "Bangladesh",
        "continent": "Asia",
        "latitude": 23.8103,
        "longitude": 90.4125,
        "coastal": True,
        "climate_zone": "Tropical monsoon",
        "note": "Cyclone-prone, extreme flood risk",
    },
    "Tokyo": {
        "country": "Japan",
        "continent": "Asia",
        "latitude": 35.6762,
        "longitude": 139.6503,
        "coastal": True,
        "climate_zone": "Humid subtropical",
        "note": "Typhoon corridor, urban heat island",
    },
    "Karachi": {
        "country": "Pakistan",
        "continent": "Asia",
        "latitude": 24.8607,
        "longitude": 67.0011,
        "coastal": True,
        "climate_zone": "Arid / hot desert",
        "note": "Extreme heatwaves, monsoon flooding",
    },
    # ── NON-COASTAL CITIES (10) ──────────────────────────────────────────
    "Delhi": {
        "country": "India",
        "continent": "Asia",
        "latitude": 28.7041,
        "longitude": 77.1025,
        "coastal": False,
        "climate_zone": "Semi-arid / hot",
        "note": "Severe air pollution, extreme heat",
    },
    "Riyadh": {
        "country": "Saudi Arabia",
        "continent": "Asia",
        "latitude": 24.7136,
        "longitude": 46.6753,
        "coastal": False,
        "climate_zone": "Hot desert",
        "note": "Extreme heat >50°C, sandstorms",
    },
    "Nairobi": {
        "country": "Kenya",
        "continent": "Africa",
        "latitude": -1.2921,
        "longitude": 36.8219,
        "coastal": False,
        "climate_zone": "Subtropical highland",
        "note": "Drought–flood oscillation, altitude climate",
    },
    "Chicago": {
        "country": "USA",
        "continent": "North America",
        "latitude": 41.8781,
        "longitude": -87.6298,
        "coastal": False,
        "climate_zone": "Humid continental",
        "note": "Polar vortex, severe winter storms",
    },
    "Moscow": {
        "country": "Russia",
        "continent": "Europe",
        "latitude": 55.7558,
        "longitude": 37.6173,
        "coastal": False,
        "climate_zone": "Humid continental",
        "note": "Extreme cold, heatwave events (2010)",
    },
    "Ulaanbaatar": {
        "country": "Mongolia",
        "continent": "Asia",
        "latitude": 47.8864,
        "longitude": 106.9057,
        "coastal": False,
        "climate_zone": "Subarctic / cold steppe",
        "note": "Coldest capital, dzud (extreme winter)",
    },
    "Phoenix": {
        "country": "USA",
        "continent": "North America",
        "latitude": 33.4484,
        "longitude": -112.0740,
        "coastal": False,
        "climate_zone": "Hot desert",
        "note": "Record-breaking heatwaves, urban heat island",
    },
    "Sao_Paulo": {
        "country": "Brazil",
        "continent": "South America",
        "latitude": -23.5505,
        "longitude": -46.6333,
        "coastal": False,
        "climate_zone": "Humid subtropical",
        "note": "Urban flooding, water crisis (2014-15)",
    },
    "Cairo": {
        "country": "Egypt",
        "continent": "Africa",
        "latitude": 30.0444,
        "longitude": 31.2357,
        "coastal": False,
        "climate_zone": "Hot desert",
        "note": "Extreme heat, dust storms",
    },
    "Madrid": {
        "country": "Spain",
        "continent": "Europe",
        "latitude": 40.4168,
        "longitude": -3.7038,
        "coastal": False,
        "climate_zone": "Hot-summer Mediterranean",
        "note": "European heatwaves, drought risk",
    },
}

logger.info(f"Registered {len(CITIES)} cities: "
            f"{sum(1 for c in CITIES.values() if c['coastal'])} coastal, "
            f"{sum(1 for c in CITIES.values() if not c['coastal'])} non-coastal")

# ============================================================================
# 5. API CONFIGURATION
# ============================================================================
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Hourly weather variables to fetch
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "windspeed_10m",
    "surface_pressure",
    "cloudcover",
    "shortwave_radiation",
    "visibility",
]

# Date range
START_DATE = "2009-01-01"
END_DATE = "2023-12-31"

# Chunking: Open-Meteo handles large ranges but we chunk by year
# to avoid timeouts and enable progress tracking + caching
CHUNK_BY_YEAR = True

# Rate limiting: Open-Meteo allows ~10,000 requests/day for free tier
# We add a polite delay between requests to avoid hitting limits
REQUEST_DELAY_SECONDS = 1.5   # seconds between API calls
MAX_RETRIES = 5               # retry on failure
RETRY_BACKOFF_FACTOR = 2.0    # exponential backoff multiplier

# ============================================================================
# 6. HELPER FUNCTIONS
# ============================================================================

def generate_cache_key(city_name: str, start: str, end: str) -> str:
    """Generate a deterministic cache key for a city + date range."""
    raw = f"{city_name}_{start}_{end}_{'_'.join(HOURLY_VARIABLES)}"
    return hashlib.md5(raw.encode()).hexdigest()


def load_from_cache(cache_key: str) -> pd.DataFrame | None:
    """Load cached data if it exists and is valid."""
    cache_file = CACHE_DIR / f"{cache_key}.parquet"
    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            logger.info(f"  ✓ Loaded from cache ({len(df)} rows)")
            return df
        except Exception as e:
            logger.warning(f"  ✗ Cache corrupted, re-downloading: {e}")
            cache_file.unlink()
    return None


def save_to_cache(df: pd.DataFrame, cache_key: str) -> None:
    """Save dataframe to parquet cache."""
    cache_file = CACHE_DIR / f"{cache_key}.parquet"
    df.to_parquet(cache_file, index=False)


def fetch_chunk(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    retries: int = MAX_RETRIES,
) -> dict | None:
    """
    Fetch one chunk of data from Open-Meteo Archive API.
    Implements exponential backoff on failure.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
    }

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(BASE_URL, params=params, timeout=120)

            if response.status_code == 200:
                return response.json()

            elif response.status_code == 429:
                # Rate limited — wait and retry
                wait = REQUEST_DELAY_SECONDS * RETRY_BACKOFF_FACTOR ** attempt
                logger.warning(
                    f"  ⚠ Rate limited (429). Waiting {wait:.0f}s "
                    f"(attempt {attempt}/{retries})"
                )
                time.sleep(wait)

            elif response.status_code == 400:
                logger.error(
                    f"  ✗ Bad request (400): {response.text[:200]}"
                )
                return None

            else:
                logger.warning(
                    f"  ⚠ HTTP {response.status_code} — attempt "
                    f"{attempt}/{retries}"
                )
                time.sleep(REQUEST_DELAY_SECONDS * attempt)

        except requests.exceptions.Timeout:
            wait = REQUEST_DELAY_SECONDS * RETRY_BACKOFF_FACTOR ** attempt
            logger.warning(
                f"  ⚠ Timeout — waiting {wait:.0f}s "
                f"(attempt {attempt}/{retries})"
            )
            time.sleep(wait)

        except requests.exceptions.ConnectionError as e:
            wait = REQUEST_DELAY_SECONDS * RETRY_BACKOFF_FACTOR ** attempt
            logger.warning(
                f"  ⚠ Connection error — waiting {wait:.0f}s "
                f"(attempt {attempt}/{retries}): {e}"
            )
            time.sleep(wait)

    logger.error("  ✗ All retry attempts exhausted.")
    return None


def parse_api_response(data: dict) -> pd.DataFrame:
    """
    Parse Open-Meteo JSON response into a clean DataFrame.

    The API returns:
    {
        "hourly": {
            "time": ["2009-01-01T00:00", ...],
            "temperature_2m": [5.2, ...],
            ...
        }
    }
    """
    hourly = data.get("hourly", {})
    if not hourly or "time" not in hourly:
        return pd.DataFrame()

    df = pd.DataFrame(hourly)
    df.rename(columns={"time": "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    return df


# ============================================================================
# 7. MAIN DATA COLLECTION FUNCTION
# ============================================================================

def collect_city_data(city_name: str, city_info: dict) -> pd.DataFrame:
    """
    Collect all hourly weather data for a single city.
    Downloads year-by-year chunks with caching and rate limiting.

    Parameters
    ----------
    city_name : str
        Name of the city (used for logging and file naming).
    city_info : dict
        Dictionary with latitude, longitude, and metadata.

    Returns
    -------
    pd.DataFrame
        Complete hourly weather data for the city (2009–2023).
    """
    lat = city_info["latitude"]
    lon = city_info["longitude"]

    logger.info(
        f"\n{'='*60}\n"
        f"  Collecting: {city_name} ({city_info['country']})\n"
        f"  Coordinates: ({lat}, {lon})\n"
        f"  Coastal: {city_info['coastal']}\n"
        f"  Climate Zone: {city_info['climate_zone']}\n"
        f"{'='*60}"
    )

    yearly_frames = []
    start_year = int(START_DATE[:4])
    end_year = int(END_DATE[:4])

    for year in range(start_year, end_year + 1):
        chunk_start = f"{year}-01-01"
        chunk_end = f"{year}-12-31"

        logger.info(f"  📅 Year {year} ({chunk_start} → {chunk_end})")

        # Check cache first
        cache_key = generate_cache_key(city_name, chunk_start, chunk_end)
        cached_df = load_from_cache(cache_key)

        if cached_df is not None:
            yearly_frames.append(cached_df)
            continue

        # Fetch from API
        raw_data = fetch_chunk(lat, lon, chunk_start, chunk_end)

        if raw_data is None:
            logger.error(
                f"  ✗ FAILED to fetch {city_name} for {year}. "
                f"Skipping this year."
            )
            continue

        df_chunk = parse_api_response(raw_data)

        if df_chunk.empty:
            logger.warning(f"  ⚠ No data returned for {year}")
            continue

        # Cache for future runs
        save_to_cache(df_chunk, cache_key)
        yearly_frames.append(df_chunk)

        logger.info(
            f"  ✓ Fetched {len(df_chunk):,} rows for {year}"
        )

        # Polite delay between API calls
        time.sleep(REQUEST_DELAY_SECONDS)

    if not yearly_frames:
        logger.error(f"  ✗ No data collected for {city_name}!")
        return pd.DataFrame()

    # Concatenate all years
    df_city = pd.concat(yearly_frames, ignore_index=True)

    # Add city metadata columns
    df_city["city"] = city_name
    df_city["country"] = city_info["country"]
    df_city["continent"] = city_info["continent"]
    df_city["latitude"] = lat
    df_city["longitude"] = lon
    df_city["coastal"] = int(city_info["coastal"])  # Binary: 1 or 0
    df_city["climate_zone"] = city_info["climate_zone"]

    # Sort by time and remove duplicates
    df_city.sort_values("datetime", inplace=True)
    df_city.drop_duplicates(subset=["datetime"], keep="first", inplace=True)
    df_city.reset_index(drop=True, inplace=True)

    logger.info(
        f"  ✅ {city_name} complete: {len(df_city):,} total rows "
        f"({df_city['datetime'].min()} → {df_city['datetime'].max()})"
    )

    return df_city


# ============================================================================
# 8. DATA QUALITY ASSESSMENT
# ============================================================================

def generate_quality_report(
    city_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Generate a comprehensive data quality report for all cities.

    Checks:
    - Total rows (expected ~131,400 per city for 15 years hourly)
    - Missing values per feature
    - Outlier counts (IQR method)
    - Date coverage completeness
    - Basic statistics

    Returns
    -------
    pd.DataFrame
        Quality report with one row per city.
    """
    logger.info("\n📊 Generating Data Quality Report...")

    expected_hourly_rows = 15 * 365.25 * 24  # ~131,490

    report_rows = []

    for city_name, df in city_frames.items():
        if df.empty:
            report_rows.append({
                "city": city_name,
                "status": "FAILED",
                "total_rows": 0,
            })
            continue

        row = {
            "city": city_name,
            "country": df["country"].iloc[0],
            "continent": df["continent"].iloc[0],
            "coastal": bool(df["coastal"].iloc[0]),
            "total_rows": len(df),
            "expected_rows": int(expected_hourly_rows),
            "coverage_pct": round(len(df) / expected_hourly_rows * 100, 2),
            "date_start": str(df["datetime"].min()),
            "date_end": str(df["datetime"].max()),
        }

        # Missing values per weather feature
        for var in HOURLY_VARIABLES:
            if var in df.columns:
                n_missing = df[var].isna().sum()
                pct_missing = round(n_missing / len(df) * 100, 4)
                row[f"missing_{var}"] = n_missing
                row[f"missing_{var}_pct"] = pct_missing

        # Outlier detection (IQR method) for numeric weather features
        for var in HOURLY_VARIABLES:
            if var in df.columns and df[var].dtype in ["float64", "int64"]:
                q1 = df[var].quantile(0.25)
                q3 = df[var].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                n_outliers = int(
                    ((df[var] < lower) | (df[var] > upper)).sum()
                )
                row[f"outliers_{var}"] = n_outliers

        # Basic statistics for temperature (key variable)
        if "temperature_2m" in df.columns:
            row["temp_mean"] = round(df["temperature_2m"].mean(), 2)
            row["temp_min"] = round(df["temperature_2m"].min(), 2)
            row["temp_max"] = round(df["temperature_2m"].max(), 2)
            row["temp_std"] = round(df["temperature_2m"].std(), 2)

        row["status"] = "OK"
        report_rows.append(row)

    report_df = pd.DataFrame(report_rows)
    return report_df


def print_quality_summary(report: pd.DataFrame) -> None:
    """Print a human-readable summary of the quality report."""
    print("\n" + "=" * 70)
    print("   DATA QUALITY SUMMARY")
    print("=" * 70)

    total_cities = len(report)
    ok_cities = len(report[report["status"] == "OK"])
    failed = len(report[report["status"] == "FAILED"])

    print(f"\n  Cities processed:  {total_cities}")
    print(f"  Successful:        {ok_cities}")
    print(f"  Failed:            {failed}")

    if "total_rows" in report.columns:
        total_rows = report["total_rows"].sum()
        print(f"\n  Total data points: {total_rows:,.0f}")

    if "coverage_pct" in report.columns:
        ok_report = report[report["status"] == "OK"]
        if not ok_report.empty:
            avg_coverage = ok_report["coverage_pct"].mean()
            min_coverage = ok_report["coverage_pct"].min()
            print(f"  Avg coverage:      {avg_coverage:.1f}%")
            print(f"  Min coverage:      {min_coverage:.1f}%")

    # Show missing values summary
    missing_cols = [c for c in report.columns if c.startswith("missing_") and c.endswith("_pct")]
    if missing_cols and ok_cities > 0:
        print("\n  Missing Values (avg % across cities):")
        for col in missing_cols:
            var_name = col.replace("missing_", "").replace("_pct", "")
            avg_pct = report.loc[report["status"] == "OK", col].mean()
            print(f"    {var_name:.<30s} {avg_pct:.3f}%")

    # Temperature ranges
    if "temp_mean" in report.columns and ok_cities > 0:
        ok_r = report[report["status"] == "OK"]
        print("\n  Temperature Ranges (°C):")
        coldest = ok_r.loc[ok_r["temp_mean"].idxmin()]
        hottest = ok_r.loc[ok_r["temp_mean"].idxmax()]
        print(f"    Coldest avg:  {coldest['city']} ({coldest['temp_mean']}°C)")
        print(f"    Hottest avg:  {hottest['city']} ({hottest['temp_mean']}°C)")
        if "temp_min" in report.columns:
            overall_min_row = ok_r.loc[ok_r["temp_min"].idxmin()]
            overall_max_row = ok_r.loc[ok_r["temp_max"].idxmax()]
            print(f"    Abs min:      {overall_min_row['city']} "
                  f"({overall_min_row['temp_min']}°C)")
            print(f"    Abs max:      {overall_max_row['city']} "
                  f"({overall_max_row['temp_max']}°C)")

    print("\n" + "=" * 70)


# ============================================================================
# 9. MASTER MERGE FUNCTION
# ============================================================================

def merge_and_save(city_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all city DataFrames into one master dataset and save.

    Saves:
    - Individual city CSVs → data/raw/{city_name}.csv
    - Merged master CSV     → data/raw/master_weather_data.csv
    """
    logger.info("\n📁 Saving individual city CSVs...")

    valid_frames = []
    for city_name, df in city_frames.items():
        if df.empty:
            logger.warning(f"  ⚠ Skipping {city_name} (empty)")
            continue

        city_path = RAW_DIR / f"{city_name}.csv"
        df.to_csv(city_path, index=False)
        logger.info(f"  ✓ Saved {city_name}: {city_path} ({len(df):,} rows)")
        valid_frames.append(df)

    if not valid_frames:
        logger.error("  ✗ No valid data to merge!")
        return pd.DataFrame()

    # Merge all cities
    logger.info("\n🔗 Merging all cities into master dataset...")
    master_df = pd.concat(valid_frames, ignore_index=True)
    master_df.sort_values(["city", "datetime"], inplace=True)
    master_df.reset_index(drop=True, inplace=True)

    master_path = RAW_DIR / "master_weather_data.csv"
    master_df.to_csv(master_path, index=False)

    logger.info(
        f"  ✅ Master dataset saved: {master_path}\n"
        f"     Total rows:   {len(master_df):,}\n"
        f"     Total cities: {master_df['city'].nunique()}\n"
        f"     Columns:      {list(master_df.columns)}\n"
        f"     Date range:   {master_df['datetime'].min()} → "
        f"{master_df['datetime'].max()}\n"
        f"     File size:    {master_path.stat().st_size / (1024**2):.1f} MB"
    )

    return master_df


# ============================================================================
# 10. MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution flow:
    1. Iterate over all 20 cities
    2. Fetch hourly data year-by-year (with caching + rate limiting)
    3. Save individual city CSVs
    4. Merge into master CSV
    5. Generate and save data quality report
    """
    print(r"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║   AI-Driven Extreme Weather Prediction — Data Collection       ║
    ║   20 Global Cities · Hourly Data · 2009–2023                   ║
    ║   Source: Open-Meteo Historical Weather API                    ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    start_time = time.time()

    # ── Step 1: Collect data for each city ──────────────────────────────
    city_frames: dict[str, pd.DataFrame] = {}
    total = len(CITIES)

    for idx, (city_name, city_info) in enumerate(CITIES.items(), 1):
        logger.info(f"\n[{idx}/{total}] Processing {city_name}...")
        df = collect_city_data(city_name, city_info)
        city_frames[city_name] = df

    # ── Step 2: Save individual + master CSV ────────────────────────────
    master_df = merge_and_save(city_frames)

    # ── Step 3: Generate quality report ─────────────────────────────────
    report_df = generate_quality_report(city_frames)
    report_path = REPORT_DIR / "data_quality_report.csv"
    report_df.to_csv(report_path, index=False)
    logger.info(f"\n📊 Quality report saved: {report_path}")

    # Print human-readable summary
    print_quality_summary(report_df)

    # ── Step 4: Save city metadata as JSON ──────────────────────────────
    metadata_path = RAW_DIR / "city_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(CITIES, f, indent=2)
    logger.info(f"📍 City metadata saved: {metadata_path}")

    # ── Completion summary ──────────────────────────────────────────────
    elapsed = time.time() - start_time
    elapsed_min = elapsed / 60

    print(f"\n{'='*60}")
    print(f"  ✅ DATA COLLECTION COMPLETE")
    print(f"  Time elapsed: {elapsed_min:.1f} minutes")
    print(f"  Files created:")
    print(f"    · {len(city_frames)} city CSVs in {RAW_DIR}/")
    print(f"    · Master CSV: {RAW_DIR}/master_weather_data.csv")
    print(f"    · Quality report: {report_path}")
    print(f"    · City metadata: {metadata_path}")
    print(f"{'='*60}\n")

    return master_df, report_df


# ============================================================================
# 11. ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    master_df, report_df = main()
