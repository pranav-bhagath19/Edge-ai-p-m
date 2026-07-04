"""
utils.py
========
Shared helper functions used across training / edge-optimization / dashboard
scripts.
"""

import os
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")


def add_rolling_features(df, sensor_cols, group_col, window=5):
    """
    Adds rolling mean & rolling std features per group (e.g. per unit_id)
    for the given sensor columns. This is standard feature engineering for
    time-series degradation modelling (smooths noise, exposes trend).
    """
    df = df.sort_values([group_col, "cycle"]).copy()
    for col in sensor_cols:
        df[f"{col}_roll_mean"] = (
            df.groupby(group_col)[col]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        df[f"{col}_roll_std"] = (
            df.groupby(group_col)[col]
            .transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        )
    return df


def benchmark_inference(model, X_sample, n_runs=200):
    """
    Benchmarks average single-sample inference latency (ms) for a fitted
    model -- used to demonstrate "edge-readiness" (fast, lightweight
    inference suitable for on-device deployment).
    """
    # warm-up
    for _ in range(5):
        model.predict(X_sample[:1])

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(X_sample[:1])
        times.append((time.perf_counter() - start) * 1000.0)
    return {
        "mean_ms": float(np.mean(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }


def model_size_kb(path):
    return os.path.getsize(path) / 1024.0
