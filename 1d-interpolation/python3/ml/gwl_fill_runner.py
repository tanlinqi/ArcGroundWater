#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Python 3 backend for ArcWater groundwater time-series imputation tools."""

from __future__ import print_function

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
import warnings
import webbrowser

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


METHOD_LABELS = {
    "linear": "Linear Regression",
    "ridge": "Ridge Regression",
    "rf": "Random Forest",
    "knn": "KNN Regression",
    "svr": "SVR",
    "gradient": "Gradient Boosting",
    "decisiontree": "DecisionTree",
    "extratrees": "ExtraTrees",
    "adaboost": "AdaBoost",
    "arima": "ARIMA",
}

FILE_SUFFIXES = {
    "linear": "_lr_filled.csv",
    "ridge": "_ridge_filled.csv",
    "rf": "_rf_filled.csv",
    "knn": "_knn_filled.csv",
    "svr": "_svr_filled.csv",
    "gradient": "_gb_filled.csv",
    "decisiontree": "_decisiontree_filled.csv",
    "extratrees": "_extratrees_filled.csv",
    "adaboost": "_adaboost_filled.csv",
    "arima": "_arima_filled.csv",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run ArcWater ML imputation")
    parser.add_argument("--params-file", required=True)
    parser.add_argument("--run-log", default="")
    return parser.parse_args(argv)


def log(message, run_log=""):
    text = str(message)
    print(text)
    if run_log:
        try:
            with open(run_log, "a", encoding="utf-8") as stream:
                stream.write(text + "\n")
        except Exception:
            pass


def read_payload(path):
    with open(path, "r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def text(params, index, default=""):
    if index >= len(params) or params[index] is None:
        return default
    value = str(params[index]).strip()
    if value == "#":
        return default
    return value if value else default


def parse_bool(value, default=True):
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() not in ("false", "0", "no", "n")


def parse_int(value, default):
    try:
        return int(str(value).strip())
    except Exception:
        return default


def parse_float(value, default):
    try:
        return float(str(value).strip())
    except Exception:
        return default


def parse_csv_paths(value):
    return [item.strip("\\'\" ") for item in str(value or "").split(";") if item.strip("\\'\" ")]


def parse_float_list(value, default):
    result = []
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            result.append(float(item))
        except ValueError:
            pass
    return result or list(default)


def parse_int_list(value, default):
    result = []
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            result.append(int(item))
        except ValueError:
            pass
    return result or list(default)


def parse_text_list(value, default, lower=True):
    result = []
    for item in str(value or "").split(","):
        item = item.strip()
        if item:
            result.append(item.lower() if lower else item)
    return result or list(default)


def parse_depth_list(value, default):
    result = []
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() == "none":
            result.append(None)
        else:
            try:
                result.append(int(item))
            except ValueError:
                pass
    return result or list(default)


def parse_gamma_list(value):
    result = []
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() in ("scale", "auto"):
            result.append(item.lower())
            continue
        try:
            result.append(float(item))
        except ValueError:
            pass
    return result or ["scale", "auto"]


def parse_optional_float(value):
    value = str(value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_mixed_list(value, default):
    result = []
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        lower = item.lower()
        if lower == "none":
            result.append(None)
        elif lower in ("sqrt", "log2", "auto"):
            result.append(lower)
        else:
            try:
                number = float(item)
                result.append(int(number) if number.is_integer() else number)
            except ValueError:
                pass
    return result or list(default)


def first_csv_item(value, default=""):
    text_value = str(value or "").strip()
    if not text_value or text_value == "#":
        return default
    return text_value.split(",", 1)[0].strip()


def single_float_param(value, default):
    try:
        return [float(first_csv_item(value, default))]
    except Exception:
        return [float(default)]


def single_int_param(value, default):
    try:
        return [int(float(first_csv_item(value, default)))]
    except Exception:
        return [int(default)]


def single_depth_param(value, default=None):
    item = first_csv_item(value, "None" if default is None else str(default))
    if item.lower() == "none":
        return [None]
    try:
        return [int(float(item))]
    except Exception:
        return [default]


def single_text_param(value, default, lower=True):
    item = first_csv_item(value, default)
    return [item.lower() if lower else item]


def single_gamma_param(value, default="scale"):
    item = first_csv_item(value, default).lower()
    if item in ("scale", "auto"):
        return [item]
    try:
        return [float(item)]
    except Exception:
        return [default]


def single_mixed_param(value, default):
    item = first_csv_item(value, "None" if default is None else str(default))
    lower = item.lower()
    if lower == "none":
        return [None]
    if lower in ("sqrt", "log2", "auto"):
        return [lower]
    try:
        number = float(item)
        return [int(number) if number.is_integer() else number]
    except Exception:
        return [default]

def parse_train_test_ratio(value, default=0.2):
    value = str(value or "").strip()
    if not value:
        return default
    try:
        if ":" in value:
            left, right = value.split(":", 1)
            left, right = float(left), float(right)
            return right / (left + right) if left + right > 0 else default
        return float(value)
    except Exception:
        return default


def load_html_template(project_root):
    path = os.path.join(project_root, "python2", "ml", "html_template.py")
    spec = importlib.util.spec_from_file_location("arcwater_html_template", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_json(value):
    return json.dumps(value, ensure_ascii=False)


def read_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls", ".xlsm"):
        return pd.read_excel(path)
    return pd.read_csv(path, parse_dates=[0])


def csv_header(path):
    return list(read_table(path).columns)


def detect_new_layout(method, params, csv_paths):
    if len(params) < 3 or not csv_paths:
        return False
    try:
        header = set(csv_header(csv_paths[0]))
    except Exception:
        return False
    return text(params, 1) in header and text(params, 2) in header


def rmse(y_true, y_pred):
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def y_axis_bounds(values):
    valid = [float(v) for v in values if pd.notna(v)]
    if not valid:
        return None, None, 0.0
    low, high = min(valid), max(valid)
    spread = high - low
    if spread == 0:
        padding = max(abs(high) * 0.002, 0.01) if high != 0 else 1.0
    else:
        padding = max(spread * 0.03, 0.01)
    return round(low - padding, 2), round(high + padding, 2), round(float(np.mean(valid)), 2)


def prepare_frame(df, common):
    time_field = common.get("time_field", "")
    if time_field:
        if time_field not in df.columns:
            raise ValueError("Time field '{}' not found.".format(time_field))
        df = df.copy()
        df["_arcwater_time"] = pd.to_datetime(df[time_field], errors="coerce")
        if df["_arcwater_time"].isna().all():
            df["_arcwater_time"] = df[time_field].astype(str)
        if common.get("sort_by_time", True):
            df = df.sort_values("_arcwater_time").reset_index(drop=True)
        duplicated = df.duplicated("_arcwater_time", keep=False)
        if duplicated.any():
            how = common.get("duplicate_handling", "error")
            if how == "error":
                raise ValueError("Duplicate time values were found in '{}'.".format(time_field))
            if how in ("first", "last"):
                df = df.drop_duplicates("_arcwater_time", keep=how).reset_index(drop=True)
            elif how == "mean":
                target = common["y_field"]
                agg = {col: "first" for col in df.columns}
                agg[target] = "mean"
                df = df.groupby("_arcwater_time", as_index=False).agg(agg).reset_index(drop=True)
            else:
                raise ValueError("Unsupported duplicate handling: {}".format(how))
    return df


def x_labels_for(df, common):
    time_field = common.get("time_field", "")
    source = df["_arcwater_time"] if "_arcwater_time" in df.columns else df.iloc[:, 0]
    dates = pd.to_datetime(source, errors="coerce")
    if dates.notna().all():
        return dates.dt.strftime("%Y-%m-%d").tolist()
    return source.astype(str).tolist()


def feature_seed_series(series, method):
    # Strict no-future rule: lag features may only be seeded from previous
    # observations.  Other historical options were removed because they can
    # use later values and inflate validation scores.
    return series.ffill()


def add_time_features(df, common, feature_cols):
    if common.get("include_trend", False):
        df["_trend_index"] = np.arange(len(df), dtype=float)
        feature_cols.append("_trend_index")
    if common.get("include_seasonal", False) and "_arcwater_time" in df.columns:
        dates = pd.to_datetime(df["_arcwater_time"], errors="coerce")
        if dates.notna().all():
            month_angle = 2.0 * np.pi * dates.dt.month.astype(float) / 12.0
            df["_month_sin"] = np.sin(month_angle)
            df["_month_cos"] = np.cos(month_angle)
            feature_cols.extend(["_month_sin", "_month_cos"])


def split_train_test(X, y, common):
    ratio = common["test_ratio"]
    split = int(len(y) * (1.0 - ratio))
    split = max(1, min(split, len(y) - 1))
    return X[:split], X[split:], y[:split], y[split:]


def cv_strategy(y_train, common):
    folds = max(2, min(common["cv_folds"], len(y_train) - 1))
    return TimeSeriesSplit(n_splits=folds)


def clamp_values(values, common):
    result = np.asarray(values, dtype=float).copy()
    if common.get("min_value") is not None:
        result = np.maximum(result, common["min_value"])
    if common.get("max_value") is not None:
        result = np.minimum(result, common["max_value"])
    return result


def finite_metric_values(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def rounded_or_none(value, digits=2):
    try:
        value = float(value)
    except Exception:
        return None
    if not np.isfinite(value):
        return None
    return round(value, digits)


def metric_block(indices, y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_true_metric, y_pred_metric = finite_metric_values(y_true, y_pred)
    residuals = y_pred - y_true
    block = {
        "indices": list(indices),
        "y_true": [rounded_or_none(v) for v in y_true],
        "y_pred": [rounded_or_none(v) for v in y_pred],
        "residual": [rounded_or_none(v) for v in residuals],
        "r2": round(float(r2_score(y_true_metric, y_pred_metric)), 2) if len(y_true_metric) > 1 else None,
        "rmse": round(rmse(y_true_metric, y_pred_metric), 2) if len(y_true_metric) else None,
        "mae": round(float(mean_absolute_error(y_true_metric, y_pred_metric)), 2) if len(y_true_metric) else None,
        "bias": round(float(np.mean(y_pred_metric - y_true_metric)), 2) if len(y_true_metric) else None,
    }
    residual_rows = []
    for idx, true_value, pred_value, residual in zip(block["indices"], block["y_true"], block["y_pred"], block["residual"]):
        if residual is None:
            continue
        residual_rows.append({
            "index": idx,
            "y_true": true_value,
            "y_pred": pred_value,
            "residual": residual,
            "abs_residual": rounded_or_none(abs(residual)),
        })
    residual_rows.sort(key=lambda item: item["abs_residual"] or 0, reverse=True)
    block["top_residuals"] = residual_rows[:10]
    return block


def missing_segments(missing_mask):
    segments = []
    start = None
    seg_id = 0
    for index, is_missing in enumerate(list(missing_mask) + [False]):
        if is_missing and start is None:
            start = index
        elif (not is_missing) and start is not None:
            end = index - 1
            seg_id += 1
            segments.append({
                "id": seg_id,
                "start": int(start),
                "end": int(end),
                "length": int(end - start + 1),
            })
            start = None
    return segments


def imputation_report_data(df, y_field, filled_field, missing_mask, x_labels, common):
    raw_values = df[y_field].to_numpy(dtype=float)
    filled_values = df[filled_field].to_numpy(dtype=float)
    mask = np.asarray(missing_mask, dtype=bool)
    segments = missing_segments(mask)
    segment_by_index = {}
    for segment in segments:
        for index in range(segment["start"], segment["end"] + 1):
            segment_by_index[index] = segment

    details = []
    for index, is_missing in enumerate(mask):
        if not is_missing:
            continue
        prev_value = None
        for prev_index in range(index - 1, -1, -1):
            if np.isfinite(raw_values[prev_index]):
                prev_value = raw_values[prev_index]
                break
        next_value = None
        for next_index in range(index + 1, len(raw_values)):
            if np.isfinite(raw_values[next_index]):
                next_value = raw_values[next_index]
                break
        segment = segment_by_index.get(index, {"id": None, "length": None})
        filled_value = filled_values[index]
        details.append({
            "row": int(len(details) + 1),
            "index": int(index),
            "time": x_labels[index] if index < len(x_labels) else str(index),
            "imputed_value": rounded_or_none(filled_value),
            "status": "filled" if np.isfinite(filled_value) else "unfilled",
            "previous_observed": rounded_or_none(prev_value),
            "next_observed_reference": rounded_or_none(next_value),
            "segment_id": segment["id"],
            "segment_length": segment["length"],
        })

    total = int(len(mask))
    missing_count = int(mask.sum())
    filled_count = int(sum(1 for item in details if item["status"] == "filled"))
    longest = max([item["length"] for item in segments], default=0)
    leading = 0
    for item in mask:
        if item:
            leading += 1
        else:
            break
    trailing = 0
    for item in mask[::-1]:
        if item:
            trailing += 1
        else:
            break
    risk_flags = []
    missing_rate = float(missing_count) / total if total else 0.0
    if missing_rate >= 0.3:
        risk_flags.append("High missing ratio: {:.1f}%".format(missing_rate * 100.0))
    if longest > max(3, common.get("lag_steps", 0)):
        risk_flags.append("Long continuous missing segment: {} points".format(longest))
    if leading:
        risk_flags.append("Leading missing values: {} points; strict no-future mode may leave early values unfilled.".format(leading))
    if filled_count < missing_count:
        risk_flags.append("Unfilled missing values: {} points due to insufficient historical features.".format(missing_count - filled_count))

    return {
        "imputationDetails": details,
        "missingSegments": segments,
        "qualitySummary": {
            "total_count": total,
            "missing_count": missing_count,
            "filled_count": filled_count,
            "unfilled_count": missing_count - filled_count,
            "missing_rate": round(missing_rate, 2),
            "longest_missing_segment": int(longest),
            "leading_missing_count": int(leading),
            "trailing_missing_count": int(trailing),
            "risk_flags": risk_flags,
        },
    }


def fit_arima_model(series, order, seasonal_order, trend, enforce_stationarity, enforce_invertibility):
    from statsmodels.tsa.arima.model import ARIMA

    clean = pd.Series(series, dtype=float).dropna()
    if len(clean) < max(8, order[0] + order[2] + 3):
        raise ValueError("Not enough historical observations for ARIMA.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(
            clean,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
        ).fit()


def rolling_arima_predictions(values, order, seasonal_order, trend, enforce_stationarity, enforce_invertibility, common):
    predictions = np.full((len(values),), np.nan, dtype=float)
    history = []
    fallback = np.nan
    for index, value in enumerate(values):
        if len([item for item in history if np.isfinite(item)]) >= max(8, order[0] + order[2] + 3):
            try:
                fitted = fit_arima_model(history, order, seasonal_order, trend, enforce_stationarity, enforce_invertibility)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pred = float(fitted.forecast(1).iloc[0])
                pred = float(clamp_values([pred], common)[0])
            except Exception:
                pred = fallback
        else:
            pred = fallback
        predictions[index] = pred
        if np.isfinite(value):
            history.append(float(value))
            fallback = float(value)
        elif np.isfinite(pred):
            history.append(float(pred))
        else:
            history.append(np.nan)
    return predictions


def _value(params, common, new_index, old_index, default=""):
    if common["new_layout"]:
        return text(params, new_index, default)
    return text(params, old_index, default)


def method_config(method, params, common):
    seed = common["random_seed"]
    if method == "linear":
        fit = parse_bool(_value(params, common, 20, 8, "true"), True)
        positive = parse_bool(_value(params, common, 21, 99, "false"), False)
        return Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]), {
            "model__fit_intercept": [fit],
            "model__positive": [positive],
        }
    if method == "ridge":
        return Pipeline([("scaler", StandardScaler()), ("model", Ridge(random_state=seed))]), {
            "model__alpha": single_float_param(_value(params, common, 15, 8), 1.0),
            "model__fit_intercept": [parse_bool(_value(params, common, 16, 99, "true"), True)],
            "model__solver": single_text_param(_value(params, common, 17, 99, "auto"), "auto"),
            "model__tol": single_float_param(_value(params, common, 18, 99, "0.001"), 0.001),
            "model__positive": [parse_bool(_value(params, common, 19, 99, "false"), False)],
        }
    if method == "rf":
        return Pipeline([("scaler", StandardScaler()), ("model", RandomForestRegressor(random_state=seed))]), {
            "model__n_estimators": single_int_param(_value(params, common, 16, 8), 100),
            "model__max_depth": single_depth_param(_value(params, common, 17, 9), None),
            "model__min_samples_leaf": single_int_param(_value(params, common, 18, 99), 1),
            "model__min_samples_split": single_int_param(_value(params, common, 19, 99), 2),
            "model__max_features": single_mixed_param(_value(params, common, 20, 99), "sqrt"),
            "model__bootstrap": [parse_bool(_value(params, common, 21, 99, "true"), True)],
        }
    if method == "knn":
        return Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor(n_jobs=1))]), {
            "model__n_neighbors": single_int_param(_value(params, common, 15, 8), 5),
            "model__weights": single_text_param(_value(params, common, 16, 9), "uniform"),
            "model__metric": single_text_param(_value(params, common, 17, 99, "minkowski"), "minkowski"),
            "model__p": single_float_param(_value(params, common, 18, 99, "2"), 2.0),
            "model__algorithm": single_text_param(_value(params, common, 19, 99, "auto"), "auto"),
        }
    if method == "svr":
        max_iter = parse_int(_value(params, common, 21, 13), -1)
        return Pipeline([("scaler", StandardScaler()), ("model", SVR(max_iter=max_iter))]), {
            "model__kernel": single_text_param(_value(params, common, 15, 8), "rbf"),
            "model__C": single_float_param(_value(params, common, 16, 9), 1.0),
            "model__gamma": single_gamma_param(_value(params, common, 17, 10), "scale"),
            "model__epsilon": single_float_param(_value(params, common, 18, 11), 0.1),
            "model__degree": single_int_param(_value(params, common, 19, 99), 3),
            "model__coef0": single_float_param(_value(params, common, 20, 99), 0.0),
            "model__tol": single_float_param(_value(params, common, 22, 99, "0.001"), 0.001),
        }
    if method == "gradient":
        return Pipeline([("scaler", StandardScaler()), ("model", GradientBoostingRegressor(random_state=seed))]), {
            "model__learning_rate": single_float_param(_value(params, common, 16, 8), 0.1),
            "model__n_estimators": single_int_param(_value(params, common, 17, 9), 100),
            "model__max_depth": single_int_param(_value(params, common, 18, 99), 3),
            "model__min_samples_leaf": single_int_param(_value(params, common, 19, 99), 1),
            "model__subsample": single_float_param(_value(params, common, 20, 99), 1.0),
            "model__loss": single_text_param(_value(params, common, 21, 99, "squared_error"), "squared_error"),
        }
    if method == "decisiontree":
        return Pipeline([("scaler", StandardScaler()), ("model", DecisionTreeRegressor(random_state=seed))]), {
            "model__max_depth": single_depth_param(_value(params, common, 16, 8), None),
            "model__min_samples_leaf": single_int_param(_value(params, common, 17, 9), 1),
            "model__min_samples_split": single_int_param(_value(params, common, 18, 99), 2),
            "model__max_features": single_mixed_param(_value(params, common, 19, 99), None),
            "model__criterion": single_text_param(_value(params, common, 20, 10), "squared_error"),
            "model__ccp_alpha": single_float_param(_value(params, common, 21, 99), 0.0),
        }
    if method == "extratrees":
        return Pipeline([("scaler", StandardScaler()), ("model", ExtraTreesRegressor(random_state=seed))]), {
            "model__n_estimators": single_int_param(_value(params, common, 16, 8), 100),
            "model__max_depth": single_depth_param(_value(params, common, 17, 9), None),
            "model__min_samples_leaf": single_int_param(_value(params, common, 18, 99), 1),
            "model__min_samples_split": single_int_param(_value(params, common, 19, 10), 2),
            "model__max_features": single_mixed_param(_value(params, common, 20, 99), "sqrt"),
            "model__bootstrap": [parse_bool(_value(params, common, 21, 99, "false"), False)],
        }
    if method == "adaboost":
        depth = single_depth_param(_value(params, common, 19, 99), 3)[0]
        leaf = single_int_param(_value(params, common, 20, 99), 1)[0]
        base_estimator = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf, random_state=seed)
        return Pipeline([("scaler", StandardScaler()), ("model", AdaBoostRegressor(random_state=seed))]), {
            "model__n_estimators": single_int_param(_value(params, common, 16, 8), 100),
            "model__learning_rate": single_float_param(_value(params, common, 17, 9), 0.1),
            "model__loss": single_text_param(_value(params, common, 18, 10), "linear"),
            "model__estimator": [base_estimator],
        }
    raise ValueError("Unsupported method: {}".format(method))

def process_ml_file(csv_path, method, params, common, run_log):
    df = prepare_frame(read_table(csv_path), common)
    y_field = common["y_field"]
    if y_field not in df.columns:
        raise ValueError("Target field '{}' not found in {}".format(y_field, os.path.basename(csv_path)))

    source = pd.to_numeric(df[y_field], errors="coerce")
    df["is_missing"] = pd.isna(source) | (source == common["missing_value"])
    df[y_field] = source.replace(common["missing_value"], np.nan)

    temp_series = feature_seed_series(df[y_field], common["feature_fill_method"])
    feature_cols = []
    for index in range(1, common["lag_steps"] + 1):
        name = "lag_{}".format(index)
        df[name] = temp_series.shift(index)
        feature_cols.append(name)
    add_time_features(df, common, feature_cols)

    known_mask = ~df["is_missing"].to_numpy()
    missing_mask = df["is_missing"].to_numpy()
    if int(known_mask.sum()) < common["lag_steps"] + 5:
        raise ValueError("Not enough valid points in {}".format(os.path.basename(csv_path)))

    X_known_all = df.loc[known_mask, feature_cols].to_numpy(dtype=float)
    y_known_all = df.loc[known_mask, y_field].to_numpy(dtype=float)
    valid_known = np.isfinite(X_known_all).all(axis=1)
    X_known = X_known_all[valid_known]
    y_known = y_known_all[valid_known]
    if len(y_known) < max(5, common["cv_folds"] + 2):
        raise ValueError("Not enough valid training rows after lag construction in {}".format(os.path.basename(csv_path)))
    X_predict = df.loc[missing_mask, feature_cols].to_numpy(dtype=float)

    pipeline, param_grid = method_config(method, params, common)
    X_train, X_test, y_train, y_test = split_train_test(X_known, y_known, common)
    grid = GridSearchCV(pipeline, param_grid, cv=cv_strategy(y_train, common), scoring="neg_mean_squared_error", n_jobs=1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred_test = best_model.predict(X_test)
    y_pred_train = best_model.predict(X_train)
    predicted = np.full((len(X_predict),), np.nan, dtype=float)
    if len(X_predict):
        valid_predict = np.isfinite(X_predict).all(axis=1)
        if valid_predict.any():
            predicted[valid_predict] = clamp_values(best_model.predict(X_predict[valid_predict]), common)

    filled_field = y_field + "_filled"
    df[filled_field] = df[y_field].copy()
    df.loc[missing_mask, filled_field] = predicted

    file_label = os.path.splitext(os.path.basename(csv_path))[0]
    output_csv = os.path.join(common["output_folder"], file_label + FILE_SUFFIXES[method])
    df.drop(feature_cols, axis=1).to_csv(output_csv, index=False, encoding="utf-8-sig")

    x_labels = x_labels_for(df, common)

    values = [round(float(v), 2) if pd.notna(v) else None for v in df[filled_field].tolist()]
    scatter = [[int(i), values[i]] for i, item in enumerate(missing_mask.tolist()) if item and values[i] is not None]
    y_min, y_max, mean_value = y_axis_bounds(df[filled_field].tolist())

    train_data = metric_block(list(range(len(y_train))), y_train, y_pred_train)
    test_data = metric_block(list(range(len(y_train), len(y_train) + len(y_test))), y_test, y_pred_test)
    report_data = imputation_report_data(df, y_field, filled_field, missing_mask, x_labels, common)

    parameter_text = (
        "Method: {}<br/>Lag Steps: {}<br/>CV Folds: {}<br/>Train/Test Split: {}:{}"
        .format(METHOD_LABELS[method], common["lag_steps"], common["cv_folds"], round(1 - common["test_ratio"], 2), round(common["test_ratio"], 2))
    )
    log("Saved CSV: {}".format(output_csv), run_log)
    result = {
        "dates": x_labels,
        "lineValues": values,
        "scatterPts": scatter,
        "meanLine": mean_value,
        "yAxisMin": y_min,
        "yAxisMax": y_max,
        "parameterText": parameter_text,
        "bestParamsText": str(grid.best_params_),
        "trainData": train_data,
        "testData": test_data,
    }
    result.update(report_data)
    return file_label, output_csv, result


def process_arima_file(csv_path, params, common, run_log):
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except Exception as exc:
        raise RuntimeError("statsmodels is required for ARIMA in the selected Python 3 environment: {}".format(exc))

    df = prepare_frame(read_table(csv_path), common)
    y_field = common["y_field"]
    if y_field not in df.columns:
        raise ValueError("Target field '{}' not found in {}".format(y_field, os.path.basename(csv_path)))
    source = pd.to_numeric(df[y_field], errors="coerce")
    df["is_missing"] = pd.isna(source) | (source == common["missing_value"])
    df[y_field] = source.replace(common["missing_value"], np.nan)
    known_mask = ~df["is_missing"].to_numpy()
    missing_mask = df["is_missing"].to_numpy()
    y_known = df.loc[known_mask, y_field].to_numpy(dtype=float)
    split = int(len(y_known) * (1 - common["test_ratio"]))
    split = max(1, min(split, len(y_known) - 1))
    train_for_selection = y_known[:split]

    if common["new_layout"]:
        p_list = [parse_int(text(params, 12, "1"), 1)]
        d_list = [parse_int(text(params, 13, "0"), 0)]
        q_list = [parse_int(text(params, 14, "1"), 1)]
        seasonal_enabled = parse_bool(text(params, 15, "false"), False)
        seasonal_p = [parse_int(text(params, 16, "0"), 0)]
        seasonal_d = [parse_int(text(params, 17, "0"), 0)]
        seasonal_q = [parse_int(text(params, 18, "0"), 0)]
        seasonal_period = parse_int(text(params, 19), 12)
        trend = text(params, 20, "n")
        information_criterion = text(params, 21, "aic").lower()
        enforce_stationarity = parse_bool(text(params, 22, "true"), True)
        enforce_invertibility = parse_bool(text(params, 23, "true"), True)
    else:
        p_list = parse_int_list(text(params, 6), [0, 1, 2])
        d_list = parse_int_list(text(params, 7), [0, 1])
        q_list = parse_int_list(text(params, 8), [0, 1, 2])
        seasonal_enabled = False
        seasonal_p, seasonal_d, seasonal_q = [0], [0], [0]
        seasonal_period = 0
        trend = "n"
        information_criterion = "aic"
        enforce_stationarity = True
        enforce_invertibility = True
    best_aic = float("inf")
    best_order = (1, 0, 0)
    best_seasonal_order = (0, 0, 0, 0)
    for p in p_list:
        for d in d_list:
            for q in q_list:
                seasonal_orders = [(0, 0, 0, 0)]
                if seasonal_enabled:
                    seasonal_orders = [
                        (sp, sd, sq, seasonal_period)
                        for sp in seasonal_p for sd in seasonal_d for sq in seasonal_q
                    ]
                for seasonal_order in seasonal_orders:
                    try:
                        fitted = fit_arima_model(
                            train_for_selection,
                            (p, d, q),
                            seasonal_order,
                            trend,
                            enforce_stationarity,
                            enforce_invertibility,
                        )
                        score = fitted.bic if information_criterion == "bic" else fitted.aic
                        if score < best_aic:
                            best_aic = score
                            best_order = (p, d, q)
                            best_seasonal_order = seasonal_order
                    except Exception:
                        continue
    if not np.isfinite(best_aic):
        raise RuntimeError("No ARIMA parameter combination could be fitted.")
    all_preds = rolling_arima_predictions(
        df[y_field].to_numpy(dtype=float),
        best_order,
        best_seasonal_order,
        trend,
        enforce_stationarity,
        enforce_invertibility,
        common,
    )

    preds_known = all_preds[known_mask]
    y_train, y_test = y_known[:split], y_known[split:]
    y_pred_train, y_pred_test = preds_known[:split], preds_known[split:]

    filled_field = y_field + "_filled"
    df[filled_field] = df[y_field].copy()
    df.loc[missing_mask, filled_field] = all_preds[missing_mask]

    file_label = os.path.splitext(os.path.basename(csv_path))[0]
    output_csv = os.path.join(common["output_folder"], file_label + FILE_SUFFIXES["arima"])
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    x_labels = x_labels_for(df, common)
    values = [round(float(v), 2) if pd.notna(v) else None for v in df[filled_field].tolist()]
    scatter = [[int(i), values[i]] for i, item in enumerate(missing_mask.tolist()) if item and values[i] is not None]
    y_min, y_max, mean_value = y_axis_bounds(df[filled_field].tolist())
    train_data = metric_block(list(range(len(y_train))), y_train, y_pred_train)
    test_data = metric_block(list(range(len(y_train), len(y_train) + len(y_test))), y_test, y_pred_test)
    report_data = imputation_report_data(df, y_field, filled_field, missing_mask, x_labels, common)
    log("Saved CSV: {}".format(output_csv), run_log)
    result = {
        "dates": x_labels,
        "lineValues": values,
        "scatterPts": scatter,
        "meanLine": mean_value,
        "yAxisMin": y_min,
        "yAxisMax": y_max,
        "parameterText": "Method: ARIMA<br/>Train/Test Split: {}:{}".format(round(1 - common["test_ratio"], 2), round(common["test_ratio"], 2)),
        "bestParamsText": str({
            "p": best_order[0], "d": best_order[1], "q": best_order[2],
            "seasonal_order": best_seasonal_order,
            information_criterion.upper(): round(float(best_aic), 2),
        }),
        "trainData": train_data,
        "testData": test_data,
    }
    result.update(report_data)
    return file_label, output_csv, result


def common_params(method, params):
    initial_csv_paths = parse_csv_paths(text(params, 0))
    new_layout = detect_new_layout(method, params, initial_csv_paths)
    if method == "arima":
        csv_text = text(params, 0)
        if new_layout:
            time_field = text(params, 1)
            y_field = text(params, 2, "WaterLevel")
            time_frequency = "Auto"
            missing_value = parse_float(text(params, 3, "-99999"), -99999.0)
            sort_by_time = parse_bool(text(params, 4, "true"), True)
            duplicate_handling = "error"
            output_html = text(params, 5)
            output_folder = text(params, 6)
            open_html = parse_bool(text(params, 7, "true"), True)
            validation_strategy = "time_split"
            test_ratio = parse_train_test_ratio(text(params, 9), 0.2)
            random_seed = 42
            min_value = parse_optional_float(text(params, 11))
            max_value = parse_optional_float(text(params, 12))
        else:
            time_field = ""
            y_field = text(params, 1, "WaterLevel")
            missing_value = parse_float(text(params, 2, "-99999"), -99999.0)
            time_frequency = "Auto"
            sort_by_time = True
            duplicate_handling = "error"
            output_html = text(params, 3)
            output_folder = text(params, 4)
            open_html = parse_bool(text(params, 5, "true"), True)
            validation_strategy = "time_split"
            test_ratio = parse_train_test_ratio(text(params, 9), 0.2)
            random_seed = 42
            min_value = None
            max_value = None
        lag_steps = 0
        cv_folds = 3
        feature_fill_method = "past_only"
        include_trend = False
        include_seasonal = False
        ml_param_offset = 0
    else:
        csv_text = text(params, 0)
        if new_layout:
            time_field = text(params, 1)
            y_field = text(params, 2, "WaterLevel")
            time_frequency = "Auto"
            missing_value = parse_float(text(params, 3, "-99999"), -99999.0)
            sort_by_time = parse_bool(text(params, 4, "true"), True)
            duplicate_handling = "error"
            output_html = text(params, 5)
            output_folder = text(params, 6)
            open_html = parse_bool(text(params, 7, "true"), True)
            lag_steps = parse_int(text(params, 8, "3"), 3)
            ml_param_offset = 0
            feature_fill_method = "past_only"
            validation_strategy = "time_split"
            test_ratio = parse_train_test_ratio(text(params, 9), 0.2)
            cv_folds = parse_int(text(params, 10, "3"), 3)
            min_value = parse_optional_float(text(params, 11))
            max_value = parse_optional_float(text(params, 12))
            include_trend = parse_bool(text(params, 13, "false"), False)
            include_seasonal = parse_bool(text(params, 14, "false"), False)
            random_seed = parse_int(text(params, 15), 42) if method in ("rf", "gradient", "decisiontree", "extratrees", "adaboost") else 42
        else:
            time_field = ""
            y_field = text(params, 1, "WaterLevel")
            missing_value = parse_float(text(params, 2, "-99999"), -99999.0)
            time_frequency = "Auto"
            sort_by_time = True
            duplicate_handling = "error"
            lag_steps = parse_int(text(params, 3, "3"), 3)
            cv_folds = parse_int(text(params, 4, "3"), 3)
            output_html = text(params, 5)
            output_folder = text(params, 6)
            open_html = parse_bool(text(params, 7, "true"), True)
            feature_fill_method = "past_only"
            validation_strategy = "time_split"
            test_ratio = parse_train_test_ratio(text(params, 12 if method == "svr" else 11), 0.2)
            random_seed = 42
            min_value = None
            max_value = None
            include_trend = False
            include_seasonal = False
            ml_param_offset = 0

    csv_paths = parse_csv_paths(csv_text)
    if not csv_paths:
        raise ValueError("Input files not provided.")
    if not output_folder:
        output_folder = os.path.dirname(csv_paths[0])
    if not output_folder:
        output_folder = os.getcwd()
    os.makedirs(output_folder, exist_ok=True)
    if not output_html:
        output_html = os.path.join(output_folder, "Multiple_Interpolation_Result.html")
    if not os.path.splitext(output_html)[1]:
        output_html += ".html"
    if not os.path.dirname(output_html):
        output_html = os.path.join(output_folder, output_html)

    if min_value is not None and max_value is not None and max_value <= min_value:
        max_value = None

    return {
        "csv_paths": csv_paths,
        "new_layout": new_layout,
        "ml_param_offset": ml_param_offset if method != "arima" else 0,
        "time_field": time_field,
        "y_field": y_field,
        "missing_value": missing_value,
        "time_frequency": time_frequency,
        "sort_by_time": sort_by_time,
        "duplicate_handling": duplicate_handling,
        "lag_steps": lag_steps,
        "feature_fill_method": feature_fill_method,
        "validation_strategy": validation_strategy,
        "test_ratio": test_ratio,
        "cv_folds": cv_folds,
        "random_seed": random_seed,
        "min_value": min_value,
        "max_value": max_value,
        "include_trend": include_trend,
        "include_seasonal": include_seasonal,
        "output_html": output_html,
        "output_folder": output_folder,
        "open_html": open_html,
    }


def copy_echarts(project_root, output_html):
    source = os.path.join(project_root, "echarts.min.js")
    target = os.path.join(os.path.dirname(output_html), "echarts.min.js")
    if os.path.isfile(source) and os.path.abspath(source) != os.path.abspath(target):
        shutil.copy2(source, target)


def main(argv=None):
    args = parse_args(argv)
    payload = read_payload(args.params_file)
    method = str(payload["method"]).lower()
    params = payload["params"]
    project_root = payload["project_root"]
    if method not in METHOD_LABELS:
        raise ValueError("Unsupported method: {}".format(method))

    common = common_params(method, params)
    log("Interpolation method: {}".format(METHOD_LABELS[method]), args.run_log)
    log("Output folder: {}".format(common["output_folder"]), args.run_log)

    chart_data = {}
    outputs = []
    errors = []
    for csv_path in common["csv_paths"]:
        try:
            if not os.path.isfile(csv_path):
                raise ValueError("File not found: {}".format(csv_path))
            log("Processing: {}".format(os.path.basename(csv_path)), args.run_log)
            if method == "arima":
                label, output_csv, data = process_arima_file(csv_path, params, common, args.run_log)
            else:
                label, output_csv, data = process_ml_file(csv_path, method, params, common, args.run_log)
            chart_data[label] = data
            outputs.append(output_csv)
        except Exception as exc:
            message = "{}: {}".format(os.path.basename(csv_path), exc)
            errors.append(message)
            log("WARNING: " + message, args.run_log)

    if not chart_data:
        raise RuntimeError("No files were successfully processed. " + "; ".join(errors))

    template = load_html_template(project_root)
    html = template.generate_report_html(
        safe_json(METHOD_LABELS[method] + " Imputation")[1:-1],
        safe_json(chart_data),
        safe_json(common["y_field"]),
    )
    with open(common["output_html"], "w", encoding="utf-8") as stream:
        stream.write(html)
    copy_echarts(project_root, common["output_html"])
    log("HTML generated: {}".format(common["output_html"]), args.run_log)
    if errors:
        log("Completed with warnings: {}".format("; ".join(errors)), args.run_log)
    if common["open_html"]:
        webbrowser.open_new_tab(common["output_html"])

    manifest = {
        "success": True,
        "method": METHOD_LABELS[method],
        "output_html": common["output_html"],
        "output_csvs": outputs,
        "warnings": errors,
    }
    with open(os.path.join(common["output_folder"], "arcwater_ml_manifest.json"), "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()



