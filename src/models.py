from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, StackingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import RidgeCV

from xgboost import XGBRegressor

import matplotlib.pyplot as plt
import seaborn as sns

from .config import PATHS
from .utils import ensure_dirs, ts_print
from .deep_models import train_keras_dnn, predict_keras_dnn


sns.set_theme(style="whitegrid")


TARGET_COL = "uhi_c"
LEAKAGE_FEATURES = {"lst_day_c", "rural_mean_c"}


def add_date_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str) + "-01")
    else:
        df["date"] = pd.to_datetime(df["date"])
    return df


def time_split(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = add_date_column(df)
    df = df.sort_values("date")
    unique_months = df["date"].drop_duplicates().sort_values()
    cutoff_idx = int(len(unique_months) * (1 - test_size))
    cutoff_date = unique_months.iloc[cutoff_idx]
    train = df[df["date"] < cutoff_date]
    test = df[df["date"] >= cutoff_date]
    return train, test


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def get_feature_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = [
        "year_c",
        "month_cos",
        "month_sin",
        "is_urban",
        "t2m_c",
        "impervious",
        "night_lights",
        "lat",
        "lon",
        "lat2",
        "lon2",
        "lat_lon",
        "dist_to_water_m",
        "lst_night_c",
        "tree_cover",
        "albedo_wsa_sw",
        "t2m_x_impervious",
        "nl_x_urban",
        "tree_x_albedo",
        "lat_sin_m",
        "lat_cos_m",
        "tile_lat_bin",
        "tile_lon_bin",
        "tile_lat_bin_f",
        "tile_lon_bin_f",
        "uhi_lag1",
        "uhi_lag3",
        "uhi_lag6",
        "uhi_lag9",
        "uhi_lag12",
        "uhi_roll3",
        "uhi_roll6",
        "uhi_roll12",
        "uhi_lag1_f",
        "uhi_lag3_f",
        "uhi_lag6_f",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns and c not in LEAKAGE_FEATURES]
    categorical_cols = [c for c in ["landcover", "month_cat"] if c in df.columns]
    return numeric_cols, categorical_cols


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(mean_squared_error(y_true, y_pred, squared=False))
    except TypeError:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": _rmse(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_baselines(X_train, y_train):
    ts_print("Training baseline models...")
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=16,
            min_samples_split=3,
            random_state=42,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=200,
            max_depth=16,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
        ),
        "MLPRegressor": MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            alpha=1e-4,
            max_iter=300,
            random_state=42,
        ),
        "HistGB": HistGradientBoostingRegressor(
            max_depth=10,
            learning_rate=0.06,
            max_iter=400,
            random_state=42,
        ),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    ts_print("Baseline models trained.")
    return models


def tune_xgboost(X, y, n_splits: int = 4, n_iter: int = 12, random_state: int = 42):
    ts_print("Tuning XGBoost with time-series CV...")
    tscv = TimeSeriesSplit(n_splits=n_splits)

    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        tree_method="hist",
        n_estimators=1400,
        n_jobs=-1,
    )

    param_distributions = {
        "max_depth": [4, 6, 8, 10, 12],
        "learning_rate": [0.015, 0.03, 0.05, 0.08],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "min_child_weight": [1, 3, 5, 10],
        "gamma": [0, 0.05, 0.1],
        "reg_alpha": [0, 0.01, 0.1, 0.3],
        "reg_lambda": [0.8, 1, 1.5, 2],
    }

    def rmse_scorer(estimator, X_val, y_val):
        pred = estimator.predict(X_val)
        return -float(np.sqrt(mean_squared_error(y_val, pred)))

    search = RandomizedSearchCV(
        xgb,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=tscv,
        scoring=rmse_scorer,
        random_state=random_state,
        n_jobs=1,
        verbose=0,
    )
    search.fit(X, y)
    best_rmse = -search.best_score_
    ts_print(f"Best XGBoost CV RMSE: {best_rmse:.4f}")
    return search.best_estimator_


def train_models(df: pd.DataFrame, skip_dnn: bool = False) -> tuple[pd.DataFrame, dict]:
    ensure_dirs([PATHS.figures_dir, PATHS.models_dir, PATHS.tables_dir])

    ts_print("Preparing train/test splits...")
    train_df, test_df = time_split(df)
    train_df = train_df.sort_values("date")
    split_idx = int(len(train_df) * 0.9)
    train_split = train_df.iloc[:split_idx]
    val_split = train_df.iloc[split_idx:]

    numeric_cols, categorical_cols = get_feature_cols(df)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    ts_print("Fitting preprocessors...")
    preprocessor.fit(train_df)

    ts_print("Transforming datasets...")
    X_train = preprocessor.transform(train_split)
    X_val = preprocessor.transform(val_split)
    X_train_full = preprocessor.transform(train_df)
    X_test = preprocessor.transform(test_df)

    train_cap = min(len(train_df), 80000)
    y_train = train_split[TARGET_COL].values
    y_val = val_split[TARGET_COL].values
    y_train_full = train_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    overall_mean = float(np.mean(y_train_full))
    seasonal_means = train_df.groupby("month")[TARGET_COL].mean().to_dict()

    def seasonal_mean_for(df_slice: pd.DataFrame) -> np.ndarray:
        return df_slice["month"].map(lambda m: seasonal_means.get(m, overall_mean)).values

    train_season_full = seasonal_mean_for(train_df)
    train_season = seasonal_mean_for(train_split)
    val_season = seasonal_mean_for(val_split)
    test_season = seasonal_mean_for(test_df)

    y_train_resid_full = y_train_full - train_season_full
    y_train_resid = y_train - train_season
    y_val_resid = y_val - val_season

    ts_print("Training baselines on subset for speed...")
    base_cap = min(len(train_df), 60000)
    base_X = X_train_full[:base_cap]
    base_y = y_train_resid_full[:base_cap]
    models = train_baselines(base_X, base_y)

    tune_limit = min(len(train_df), 50000)
    X_tune = X_train_full[:tune_limit]
    y_tune = y_train_resid_full[:tune_limit]
    tuned_xgb = tune_xgboost(X_tune, y_tune)
    tuned_xgb.fit(X_train_full, y_train_resid_full)
    models["XGBoostTuned"] = tuned_xgb

    metrics = []
    predictions = {"y_test": y_test}
    season_pred = test_season
    predictions["SeasonalityOnly"] = season_pred
    season_scores = evaluate_model(y_test, season_pred)
    season_scores["model"] = "SeasonalityOnly"
    metrics.append(season_scores)

    ts_print("Scoring baseline models...")
    for name, model in models.items():
        y_pred_resid = model.predict(X_test)
        y_pred = y_pred_resid + test_season
        predictions[name] = y_pred
        scores = evaluate_model(y_test, y_pred)
        scores["model"] = name
        metrics.append(scores)

    ts_print("Training stacking ensemble...")
    base_estimators = []
    for n in ["RandomForest", "ExtraTrees", "HistGB", "XGBoostTuned"]:
        if n in models:
            base_estimators.append((n, models[n]))
    ts_print(f"Stacking base models: {[name for name, _ in base_estimators]}")
    if base_estimators:
        meta = RidgeCV(alphas=[0.1, 1.0, 10.0])
        stack = StackingRegressor(
            estimators=base_estimators,
            final_estimator=meta,
            cv=2,
            passthrough=True,
            n_jobs=-1,
        )
        stack_cap = min(len(X_train_full), 30000)
        stack.fit(X_train_full[:stack_cap], y_train_resid_full[:stack_cap])
        y_pred_resid = stack.predict(X_test)
        y_pred = y_pred_resid + test_season
        predictions["Stacked"] = y_pred
        scores = evaluate_model(y_test, y_pred)
        scores["model"] = "Stacked"
        metrics.append(scores)

    metrics_df = pd.DataFrame(metrics).sort_values("RMSE")
    metrics_df.to_csv(f"{PATHS.tables_dir}/model_metrics.csv", index=False)

    if skip_dnn:
        ts_print("Skipping Keras DNN training.")
    else:
        ts_print("Training Keras DNN...")
        dnn = train_keras_dnn(X_train, y_train_resid, X_val, y_val_resid)
        dnn_pred = predict_keras_dnn(dnn, X_test)
        dnn_pred = dnn_pred + test_season
        predictions["KerasDNN"] = dnn_pred
        dnn_scores = evaluate_model(y_test, dnn_pred)
        dnn_scores["model"] = "KerasDNN"
        metrics_df = pd.concat([metrics_df, pd.DataFrame([dnn_scores])], ignore_index=True)
        metrics_df = metrics_df.sort_values("RMSE")
        metrics_df.to_csv(f"{PATHS.tables_dir}/model_metrics.csv", index=False)

    ts_print("Generating prediction and importance plots...")
    plot_predictions(predictions)
    plot_feature_importance(models, preprocessor, numeric_cols, categorical_cols)
    ts_print("Model training complete.")

    return metrics_df, predictions


def plot_predictions(predictions: dict) -> None:
    ensure_dirs([PATHS.figures_dir])

    y_true = predictions["y_test"]
    model_names = [k for k in predictions.keys() if k != "y_test"]

    n = len(model_names)
    cols = 2
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(10, 4 * rows))
    for i, name in enumerate(model_names, start=1):
        plt.subplot(rows, cols, i)
        sns.scatterplot(x=y_true, y=predictions[name], s=10, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color="red", linestyle="--")
        plt.title(f"Actual vs Predicted: {name}")
        plt.xlabel("Actual UHI (C)")
        plt.ylabel("Predicted UHI (C)")

    plt.tight_layout()
    plt.savefig(f"{PATHS.figures_dir}/fig5_model_predictions.png", dpi=200)
    plt.close()


def plot_feature_importance(models: dict, preprocessor, numeric_cols: list[str], categorical_cols: list[str]) -> None:
    ensure_dirs([PATHS.figures_dir])

    feature_names = list(numeric_cols)
    if categorical_cols:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = ohe.get_feature_names_out(categorical_cols).tolist()
        feature_names.extend(cat_names)

    for name in ["RandomForest", "XGBoostTuned"]:
        model = models.get(name)
        if model is None or not hasattr(model, "feature_importances_"):
            continue
        importances = model.feature_importances_
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False).head(20)

        plt.figure(figsize=(8, 6))
        sns.barplot(data=imp_df, y="feature", x="importance", color="#1f77b4")
        plt.title(f"Top Feature Importances: {name}")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(f"{PATHS.figures_dir}/fig6_feature_importance_{name}.png", dpi=200)
        plt.close()
