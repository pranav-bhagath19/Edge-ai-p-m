# Edge AI-based Predictive Maintenance and Anomaly Detection System

**For Aerospace / Industrial Heavy Machinery Components Using Sensor Time-Series Data**

*InnoVent-27 — AI at the Edge | Track: Industrial Heavy Machinery (IHM) —
Edge AI for Autonomous & Intelligent Heavy Machines*

> See **HOW_TO_RUN.md** for setup instructions.

---

## Problem Statement

Unplanned equipment failure in Aerospace and Industrial Heavy Machinery
leads to costly downtime, safety risk, and reactive (rather than proactive)
maintenance. This project addresses that problem with an Edge AI-based
Predictive Maintenance and Anomaly Detection system that analyzes sensor
time-series data (vibration, temperature, pressure) to detect early signs
of degradation and predict Remaining Useful Life (RUL) — enabling
maintenance teams to act *before* failure occurs.

## What This Prototype Does

1. **Remaining Useful Life (RUL) Prediction — Aerospace track**
   A Random Forest regression model trained on turbofan engine sensor data
   predicts how many operating cycles remain before a component needs
   maintenance.

2. **Anomaly Detection — Industrial Heavy Machinery track**
   An Isolation Forest model trained only on healthy vibration data flags
   deviations (early fault / severe fault) in real time, the same way real
   condition-monitoring systems are built in practice.

3. **Edge Deployment Readiness**
   Both models are compressed and benchmarked for single-sample inference
   latency — proving they are light and fast enough for real-time, on-device
   decision-making without depending on constant cloud connectivity.

4. **Interactive Dashboard**
   A Streamlit web app lets you pick a simulated engine/machine, scrub
   through its operating life, and watch live predictions, health status,
   and anomaly alerts — exactly what you'd demo to judges.

## Why Synthetic Data (and why that's the right call)

The system is developed and validated using datasets **generated to match
the structure, feature set, and degradation behavior** of two established
public benchmarks:

- **NASA CMAPSS** turbofan engine degradation dataset (Aerospace)
- **CWRU Bearing Fault** dataset (Industrial Heavy Machinery)

This makes the prototype fully reproducible offline, with no dataset
licensing/download friction and no dependency on physical hardware or
factory access — appropriate for a **virtual Proof of Concept** at Stage 2
of the competition. The modeling approach (feature engineering, RUL
regression, unsupervised anomaly detection, edge-latency benchmarking) is
identical to what would be used on the real CMAPSS/CWRU datasets; swapping
in the real files is a drop-in replacement (see `data/generate_data.py` for
the exact schema each script expects).

## Results (from the included pre-trained models)

| Model | Metric | Value |
|---|---|---|
| RUL Prediction (Aerospace) | RMSE | ~7 cycles |
| RUL Prediction (Aerospace) | R² | ~0.97 |
| Anomaly Detection (IHM) | ROC-AUC | ~0.998 |
| Both models | Avg inference latency | < 10 ms/sample |
| Both models | Compressed size | Aerospace ~3.8 MB, IHM ~0.5 MB |

*(Exact numbers will vary slightly each time you regenerate data/retrain —
run `python src/edge_optimize.py` to refresh, or check the Model Performance
tab in the dashboard.)*

## Tech Stack

- **Python, scikit-learn** — model training (Random Forest, Isolation Forest)
- **pandas / numpy** — data processing & feature engineering
- **joblib** — model serialization & compression (edge-deployment proxy)
- **Streamlit + Plotly** — interactive dashboard

## Roadmap / Future Work (good talking points for judges)

- Swap synthetic data for real CMAPSS/CWRU files once available for
  fine-tuning/validation
- Export models to ONNX/TensorFlow Lite for deployment on actual edge
  hardware (Raspberry Pi, NVIDIA Jetson Nano, industrial PLC gateways)
- Add MQTT/OPC-UA ingestion for real sensor streaming
- Extend RUL model with LSTM/Transformer architectures for longer-horizon
  degradation trends
- Add automated maintenance-ticket generation on anomaly detection

---

*Built for the InnoVent-27 "AI at the Edge" hackathon — Industrial Heavy
Machinery track.*
