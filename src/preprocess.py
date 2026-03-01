from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression

from .config import PATHS, STUDY_AREA
from .utils import ensure_dirs


TARGET_COL = "uhi_c"
TILE_DEG = 0.05
TILE_FINE = 0.02


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df[df[TARGET_COL].notna()]

    if "lon" in df.columns and "lat" in df.columns:
        west, south, east, north = STUDY_AREA.bbox
        df = df[df["lon"].between(west, east) & df["lat"].between(south, north)]

    if "month" not in df.columns or "year" not in df.columns:
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"])
            df["month"] = dates.dt.month
            df["year"] = dates.dt.year
    if "date" not in df.columns and {"year", "month"} <= set(df.columns):
        df["date"] = pd.to_datetime(df["year"].astype(int).astype(str) + "-" + df["month"].astype(int).astype(str) + "-01")
    else:
        df["date"] = pd.to_datetime(df["date"])

    if "month" in df.columns:
        radians = 2 * np.pi * df["month"] / 12.0
        df["month_sin"] = np.sin(radians)
        df["month_cos"] = np.cos(radians)
        df["month_cat"] = df["month"].astype(int)
    if "year" in df.columns:
        df["year_c"] = df["year"] - df["year"].mean()

    if "lat" in df.columns and "lon" in df.columns:
        df["lat2"] = df["lat"] ** 2
        df["lon2"] = df["lon"] ** 2
        df["lat_lon"] = df["lat"] * df["lon"]
        df["tile_lat_bin"] = np.floor(df["lat"] / TILE_DEG).astype(int)
        df["tile_lon_bin"] = np.floor(df["lon"] / TILE_DEG).astype(int)
        df["tile_id"] = df["tile_lat_bin"].astype(str) + "_" + df["tile_lon_bin"].astype(str)
        df["tile_lat_bin_f"] = np.floor(df["lat"] / TILE_FINE).astype(int)
        df["tile_lon_bin_f"] = np.floor(df["lon"] / TILE_FINE).astype(int)
        df["tile_id_f"] = df["tile_lat_bin_f"].astype(str) + "_" + df["tile_lon_bin_f"].astype(str)

    def mask_outliers(col: str, low: float, high: float) -> None:
        if col not in df.columns:
            return
        valid = df[col].between(low, high, inclusive="both") | df[col].isna()
        df.loc[~valid, col] = np.nan

    mask_outliers("lst_day_c", -10, 70)
    mask_outliers("lst_night_c", -20, 50)
    mask_outliers("ndvi", -1.0, 1.0)
    mask_outliers("evi", -1.0, 1.0)

    if TARGET_COL in df.columns:
        lower, upper = df[TARGET_COL].quantile([0.01, 0.99])
        df[TARGET_COL] = df[TARGET_COL].clip(lower, upper)

    if "tile_id" in df.columns and TARGET_COL in df.columns:
        df = df.sort_values("date")
        for lag in [1, 3, 6, 12]:
            df[f"uhi_lag{lag}"] = (
                df.groupby("tile_id")[TARGET_COL]
                .shift(lag)
            )
        for window in [3, 6, 12]:
            df[f"uhi_roll{window}"] = (
                df.groupby("tile_id")[TARGET_COL]
                .apply(lambda s: s.shift(1).rolling(window).mean())
                .reset_index(level=0, drop=True)
            )
        for lag in [1, 3, 6]:
            df[f"uhi_lag{lag}_f"] = (
                df.groupby("tile_id_f")[TARGET_COL]
                .shift(lag)
            )

    if {"t2m_c", "impervious"} <= set(df.columns):
        df["t2m_x_impervious"] = df["t2m_c"] * df["impervious"]
    if {"night_lights", "is_urban"} <= set(df.columns):
        df["nl_x_urban"] = df["night_lights"] * df["is_urban"]
    if {"tree_cover", "albedo_wsa_sw"} <= set(df.columns):
        df["tree_x_albedo"] = df["tree_cover"] * df["albedo_wsa_sw"]
    if {"lat", "month_sin"} <= set(df.columns):
        df["lat_sin_m"] = df["lat"] * df["month_sin"]
    if {"lat", "month_cos"} <= set(df.columns):
        df["lat_cos_m"] = df["lat"] * df["month_cos"]

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    if "landcover" in df.columns:
        df["landcover"] = df["landcover"].fillna(-1).astype(int)

    if "impervious" in df.columns:
        df["is_urban"] = (df["impervious"] >= 20).astype(int)

    if "tile_id" in df.columns:
        drop_cols = [c for c in ["tile_id", "tile_id_f"] if c in df.columns]
        df = df.drop(columns=drop_cols)

    return df


def feature_selection(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].values
    y = df[TARGET_COL].values

    mi = mutual_info_regression(X, y, random_state=42)
    ranking = pd.DataFrame({"feature": feature_cols, "mutual_info": mi})
    ranking = ranking.sort_values("mutual_info", ascending=False)
    ensure_dirs([PATHS.tables_dir])
    ranking.to_csv(f"{PATHS.tables_dir}/feature_selection_mi.csv", index=False)
    return ranking


def save_cleaned(df: pd.DataFrame) -> None:
    ensure_dirs([PATHS.data_dir])
    df.to_parquet(PATHS.cleaned_samples, index=False)
