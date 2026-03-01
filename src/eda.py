from __future__ import annotations

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .config import PATHS
from .utils import ensure_dirs


sns.set_theme(style="whitegrid")


def plot_target_distribution(df: pd.DataFrame) -> None:
    ensure_dirs([PATHS.figures_dir])
    plt.figure(figsize=(8, 4))
    sns.histplot(df["uhi_c"], bins=40, kde=True, color="#1f77b4")
    plt.title("Distribution of UHI Severity (C)")
    plt.xlabel("UHI Severity (C)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{PATHS.figures_dir}/fig1_uhi_distribution.png", dpi=200)
    plt.close()


def plot_correlation(df: pd.DataFrame, cols: list[str]) -> None:
    ensure_dirs([PATHS.figures_dir])
    corr = df[cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{PATHS.figures_dir}/fig2_correlation_heatmap.png", dpi=200)
    plt.close()


def plot_spatial_uhi(df: pd.DataFrame) -> None:
    ensure_dirs([PATHS.figures_dir])
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(df["lon"], df["lat"], c=df["uhi_c"], s=6, cmap="inferno", alpha=0.7)
    plt.colorbar(sc, label="UHI Severity (C)")
    plt.title("Spatial Distribution of UHI Severity (Sample)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(f"{PATHS.figures_dir}/fig3_spatial_uhi.png", dpi=200)
    plt.close()


def plot_time_series(df: pd.DataFrame) -> None:
    ensure_dirs([PATHS.figures_dir])
    monthly = (
        df.groupby(["year", "month"], as_index=False)["uhi_c"].mean()
        .assign(date=lambda d: pd.to_datetime(d["year"].astype(str) + "-" + d["month"].astype(str) + "-01"))
        .sort_values("date")
    )

    plt.figure(figsize=(10, 4))
    plt.plot(monthly["date"], monthly["uhi_c"], color="#2ca02c")
    plt.title("Monthly Mean UHI Severity")
    plt.xlabel("Date")
    plt.ylabel("UHI Severity (C)")
    plt.tight_layout()
    plt.savefig(f"{PATHS.figures_dir}/fig4_uhi_timeseries.png", dpi=200)
    plt.close()
