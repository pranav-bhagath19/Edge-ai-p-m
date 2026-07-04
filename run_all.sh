#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_all.sh
# One-click setup + run script for Linux / macOS.
# Regenerates data, retrains both models, benchmarks edge readiness, and
# launches the dashboard.
# ---------------------------------------------------------------------------
set -e

echo "==============================================="
echo " Edge AI Predictive Maintenance -- Setup & Run "
echo "==============================================="

echo ""
echo "[1/5] Installing dependencies ..."
pip install -r requirements.txt --quiet

echo ""
echo "[2/5] Generating synthetic datasets ..."
python3 data/generate_data.py

echo ""
echo "[3/5] Training RUL prediction model (Aerospace) ..."
python3 src/train_rul_model.py

echo ""
echo "[4/5] Training anomaly detection model (IHM) ..."
python3 src/train_anomaly_model.py

echo ""
echo "[5/5] Running edge optimization & benchmarking ..."
python3 src/edge_optimize.py

echo ""
echo "==============================================="
echo " Setup complete! Launching dashboard ...        "
echo "==============================================="
streamlit run dashboard/app.py
