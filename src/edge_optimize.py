"""
edge_optimize.py
================
Demonstrates "Edge AI readiness" for both trained models:
    1. Compresses the trained models (joblib compression -- analogous to
       quantization/pruning workflows used for TFLite/ONNX edge export).
    2. Benchmarks single-sample inference latency, which is the key metric
       for real-time, on-device decision-making without cloud dependency.
    3. Reports model size before/after compression.

This produces the numbers used in the "Edge AI" story of the pitch: model
footprint (KB) and inference latency (ms) -- proving the models are light
and fast enough to run directly on-device (e.g. Raspberry Pi / Jetson Nano
class edge hardware) rather than depending on a cloud round-trip.

Run:
    python src/edge_optimize.py
"""

import os
import sys
import json
import joblib
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_inference, model_size_kb, MODEL_DIR, DATA_DIR  # noqa: E402


def optimize_rul_model():
    bundle_path = os.path.join(MODEL_DIR, "rul_model.joblib")
    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    # size before re-saving at max compression
    uncompressed_path = os.path.join(MODEL_DIR, "_tmp_rul_uncompressed.joblib")
    joblib.dump(bundle, uncompressed_path, compress=0)
    size_before = model_size_kb(uncompressed_path)

    optimized_path = os.path.join(MODEL_DIR, "rul_model_edge.joblib")
    joblib.dump(bundle, optimized_path, compress=9)
    size_after = model_size_kb(optimized_path)

    os.remove(uncompressed_path)

    # benchmark using a real feature row from the test predictions data
    test_df = pd.read_csv(os.path.join(DATA_DIR, "aerospace_test.csv"))
    from utils import add_rolling_features
    test_df = add_rolling_features(test_df, [f"sensor_{i}" for i in range(1, 9)], "unit_id", 5)
    X_sample = test_df[feature_cols].values

    latency = benchmark_inference(model, X_sample)

    return {
        "model_name": "RUL Prediction (RandomForestRegressor)",
        "size_before_kb": round(size_before, 1),
        "size_after_kb": round(size_after, 1),
        "compression_ratio": round(size_before / size_after, 2) if size_after else None,
        "inference_latency_ms": {k: round(v, 3) for k, v in latency.items()},
    }


def optimize_anomaly_model():
    bundle_path = os.path.join(MODEL_DIR, "anomaly_model.joblib")
    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]

    uncompressed_path = os.path.join(MODEL_DIR, "_tmp_anomaly_uncompressed.joblib")
    joblib.dump(bundle, uncompressed_path, compress=0)
    size_before = model_size_kb(uncompressed_path)

    optimized_path = os.path.join(MODEL_DIR, "anomaly_model_edge.joblib")
    joblib.dump(bundle, optimized_path, compress=9)
    size_after = model_size_kb(optimized_path)

    os.remove(uncompressed_path)

    test_df = pd.read_csv(os.path.join(DATA_DIR, "ihm_bearing.csv"))
    X_sample = scaler.transform(test_df[feature_cols].values)

    latency = benchmark_inference(model, X_sample)

    return {
        "model_name": "Anomaly Detection (IsolationForest)",
        "size_before_kb": round(size_before, 1),
        "size_after_kb": round(size_after, 1),
        "compression_ratio": round(size_before / size_after, 2) if size_after else None,
        "inference_latency_ms": {k: round(v, 3) for k, v in latency.items()},
    }


def main():
    print("Running Edge AI optimization + benchmarking ...\n")
    rul_report = optimize_rul_model()
    anomaly_report = optimize_anomaly_model()

    report = {"rul_model": rul_report, "anomaly_model": anomaly_report}

    print(json.dumps(report, indent=2))

    out_path = os.path.join(MODEL_DIR, "edge_benchmark_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved edge benchmark report -> {out_path}")


if __name__ == "__main__":
    main()
