from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path

from .gee_pipeline import load_or_fetch_samples
from .preprocess import clean_dataframe, feature_selection, save_cleaned
from .eda import plot_target_distribution, plot_correlation, plot_spatial_uhi, plot_time_series
from .models import train_models, get_feature_cols, LEAKAGE_FEATURES
from .arima import fit_arima_forecast
from .config import PATHS
from .utils import ts_print


def log(message: str) -> None:
    ts_print(message)


def _format_rows(n: int) -> str:
    return f"{n:,}"


def _read_csv_if_exists(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def summarize_report(df: pd.DataFrame, raw_count: int) -> None:
    metrics_path = f"{PATHS.tables_dir}/model_metrics.csv"
    arima_path = f"{PATHS.tables_dir}/arima_metrics.csv"
    feature_path = f"{PATHS.tables_dir}/feature_selection_mi.csv"

    log("\n=== Pipeline Summary ===")
    log(f"Samples (raw): {_format_rows(raw_count)}")
    log(f"Samples (cleaned): {_format_rows(len(df))}")

    metrics = _read_csv_if_exists(metrics_path)
    if metrics is not None and not metrics.empty:
        ml_metrics = metrics[~metrics["model"].isin(["SeasonalityOnly"])]
        if not ml_metrics.empty:
            best = ml_metrics.sort_values("RMSE").iloc[0]
            log(f"Best ML model: {best['model']} (RMSE={best['RMSE']:.4f}, MAE={best['MAE']:.4f}, R2={best['R2']:.4f})")
        else:
            log("Best ML model: N/A (no ML models found)")
        baseline = metrics[metrics["model"] == "SeasonalityOnly"]
        if not baseline.empty:
            row = baseline.iloc[0]
            log(f"Seasonality-only baseline: RMSE={row['RMSE']:.4f}, MAE={row['MAE']:.4f}, R2={row['R2']:.4f}")

        log("All ML model metrics:")
        for row in metrics.itertuples(index=False):
            log(f"- {row.model}: RMSE={row.RMSE:.4f}, MAE={row.MAE:.4f}, R2={row.R2:.4f}")
    else:
        log("Best ML model: N/A (metrics file missing)")

    arima = _read_csv_if_exists(arima_path)
    if arima is not None and not arima.empty:
        log("ARIMA metrics:")
        for row in arima.itertuples(index=False):
            log(f"- {row.model} (RMSE={row.RMSE:.4f})")
    else:
        log("ARIMA: N/A (metrics file missing)")

    feats = _read_csv_if_exists(feature_path)
    if feats is not None and not feats.empty:
        top = feats.sort_values("mutual_info", ascending=False).head(5)
        top_list = ", ".join(f"{r.feature} ({r.mutual_info:.3f})" for r in top.itertuples())
        log(f"Top features (MI): {top_list}")
    else:
        log("Top features (MI): N/A (feature selection file missing)")

    numeric_cols, categorical_cols = get_feature_cols(df)
    leakage_in_use = sorted(set(numeric_cols + categorical_cols) & LEAKAGE_FEATURES)
    if leakage_in_use:
        log(f"Leakage warning: model inputs include {', '.join(leakage_in_use)}.")
    else:
        log("Leakage check: OK (no known leakage features in model inputs).")

    log("Outputs:")
    log(f"- Tables: {PATHS.tables_dir}")
    log(f"- Figures: {PATHS.figures_dir}")
    log(f"- Models: {PATHS.models_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="UHI modeling pipeline for DFW.")
    parser.add_argument("--skip-fetch", action="store_true", help="Use existing local data if available.")
    parser.add_argument("--skip-dnn", action="store_true", help="Skip training the Keras DNN model.")
    args = parser.parse_args()

    log("Starting pipeline.")
    if args.skip_fetch:
        log("Skip fetch enabled. Loading local samples if available.")
        try:
            log(f"Reading samples from {PATHS.raw_samples}.")
            df = pd.read_parquet(PATHS.raw_samples)
            log(f"Loaded samples: {len(df):,} rows.")
        except Exception:
            log("Local samples not found or unreadable. Fetching from GEE.")
            df = load_or_fetch_samples()
    else:
        log("Fetching samples from GEE (or loading cached).")
        df = load_or_fetch_samples()
    log("Sample load complete.")
    raw_count = len(df)

    log("Cleaning dataframe.")
    df = clean_dataframe(df)
    log("Saving cleaned dataframe.")
    save_cleaned(df)

    log("Generating EDA plots.")
    plot_target_distribution(df)
    plot_spatial_uhi(df)
    plot_time_series(df)

    numeric_cols, _ = get_feature_cols(df)
    corr_cols = numeric_cols + (["uhi_c"] if "uhi_c" in df.columns else [])
    log("Generating correlation plot.")
    plot_correlation(df, corr_cols)

    log("Running feature selection.")
    feature_selection(df, numeric_cols)

    log("Training ML models.")
    train_models(df, skip_dnn=args.skip_dnn)

    log("Running ARIMA forecast.")
    fit_arima_forecast(df)
    log("Pipeline complete.")
    summarize_report(df, raw_count=raw_count)


if __name__ == "__main__":
    main()
