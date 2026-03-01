from __future__ import annotations

import itertools
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from .config import PATHS
from .utils import ensure_dirs, ts_print


def prepare_monthly_series(df: pd.DataFrame) -> pd.Series:
    monthly = (
        df.groupby(["year", "month"], as_index=False)["uhi_c"].mean()
        .assign(date=lambda d: pd.to_datetime(d["year"].astype(str) + "-" + d["month"].astype(str) + "-01"))
        .sort_values("date")
    )
    series = monthly.set_index("date")["uhi_c"].asfreq("MS")
    if series.isna().any():
        series = series.interpolate("time").ffill().bfill()
    return series


def select_arima_order(series: pd.Series, p_range=(0, 2), d_range=(0, 1), q_range=(0, 2)):
    best_aic = np.inf
    best_order = None
    best_any_aic = np.inf
    best_any_order = (1, 0, 1)

    for p, d, q in itertools.product(range(p_range[0], p_range[1] + 1), range(d_range[0], d_range[1] + 1), range(q_range[0], q_range[1] + 1)):
        try:
            model = SARIMAX(series, order=(p, d, q), seasonal_order=(1, 0, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                result = model.fit(disp=False, maxiter=200)

            if result.aic < best_any_aic:
                best_any_aic = result.aic
                best_any_order = (p, d, q)

            converged = result.mle_retvals.get("converged", True)
            if converged and result.aic < best_aic:
                best_aic = result.aic
                best_order = (p, d, q)
        except Exception:
            continue

    return best_order or best_any_order


def fit_arima_forecast(df: pd.DataFrame, forecast_steps: int = 12):
    ensure_dirs([PATHS.figures_dir, PATHS.tables_dir])

    ts_print("Preparing ARIMA series...")
    series = prepare_monthly_series(df)
    train = series.iloc[:-forecast_steps]
    test = series.iloc[-forecast_steps:]

    baseline_rmse = None
    if len(series) >= forecast_steps + 12:
        seasonal_naive = series.shift(12).iloc[-forecast_steps:]
        if seasonal_naive.isna().any():
            seasonal_naive = seasonal_naive.ffill().bfill()
        baseline_rmse = float(np.sqrt(((seasonal_naive - test) ** 2).mean()))

    ts_print("Selecting ARIMA order...")
    order = select_arima_order(train)
    ts_print(f"Fitting ARIMA model with order={order}.")
    model = SARIMAX(train, order=order, seasonal_order=(1, 0, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit(disp=False, maxiter=200)
    if not result.mle_retvals.get("converged", True):
        ts_print("ARIMA fit did not fully converge; results may be less reliable.")

    ts_print("Forecasting ARIMA...")
    forecast = result.get_forecast(steps=forecast_steps)
    pred = forecast.predicted_mean
    conf_int = forecast.conf_int()

    rmse = np.sqrt(((pred - test) ** 2).mean())
    metrics = [{"model": f"ARIMA{order}", "RMSE": float(rmse)}]
    if baseline_rmse is not None:
        metrics.append({"model": "SeasonalNaive(t-12)", "RMSE": float(baseline_rmse)})
    pd.DataFrame(metrics).to_csv(f"{PATHS.tables_dir}/arima_metrics.csv", index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(train.index, train.values, label="Train")
    plt.plot(test.index, test.values, label="Test", color="#2ca02c")
    plt.plot(pred.index, pred.values, label="Forecast", color="#d62728")
    plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color="#d62728", alpha=0.2)
    plt.title("ARIMA Forecast of Monthly UHI Severity")
    plt.xlabel("Date")
    plt.ylabel("UHI Severity (C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PATHS.figures_dir}/fig7_arima_forecast.png", dpi=200)
    plt.close()

    ts_print("ARIMA forecast complete.")
    return result
