# How to Run — Edge AI Predictive Maintenance & Anomaly Detection System

This project is fully self-contained and works **completely offline** — it
uses realistic synthetic datasets modeled after the NASA CMAPSS (Aerospace)
and CWRU Bearing (IHM) benchmarks, so there is nothing to download and
nothing that requires physical hardware.

The project **already includes pre-generated data and pre-trained models**
in the `data/` and `models/` folders, so you can jump straight to Step 3 if
you just want to see the dashboard. Steps 1–2 let you regenerate everything
from scratch (useful if you want to show "live training" during your demo).

---

## Prerequisites

- **Python 3.9 or newer** installed
  (Check with: `python3 --version` or `python --version`)
- Internet connection only needed once, to install Python packages via pip
  (no dataset downloads required — everything else runs offline)

---

## Option A — One-Click Run (Recommended)

### On Windows:
1. Unzip the project folder.
2. Open the folder, double-click **`run_all.bat`**
   (or open Command Prompt inside the folder and run `run_all.bat`)
3. Wait for it to install dependencies, regenerate data, retrain both
   models, run the edge benchmark, and automatically launch the dashboard
   in your browser.

### On macOS / Linux:
1. Unzip the project folder.
2. Open a terminal inside the folder and run:
   ```bash
   chmod +x run_all.sh
   ./run_all.sh
   ```
3. This installs dependencies, regenerates data, retrains both models, runs
   the edge benchmark, and launches the dashboard automatically.

The dashboard will open in your browser at **http://localhost:8501**

---

## Option B — Manual Step-by-Step Run

If you prefer to run each step yourself (or want to demo the pipeline step
by step for judges):

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the synthetic datasets
```bash
python data/generate_data.py
```
This creates:
- `data/aerospace_train.csv`, `data/aerospace_test.csv` (CMAPSS-style)
- `data/ihm_bearing.csv` (CWRU-style)

### 3. Train the Aerospace RUL prediction model
```bash
python src/train_rul_model.py
```
Trains a Random Forest model to predict Remaining Useful Life (RUL) of
turbofan engine units. Prints RMSE / MAE / R² and saves the model to
`models/rul_model.joblib`.

### 4. Train the IHM anomaly detection model
```bash
python src/train_anomaly_model.py
```
Trains an Isolation Forest anomaly detector on healthy bearing vibration
data. Prints ROC-AUC and a classification report, and saves the model to
`models/anomaly_model.joblib`.

### 5. Run the Edge AI optimization & benchmark
```bash
python src/edge_optimize.py
```
Compresses both models and benchmarks single-sample inference latency —
this produces the "Edge Readiness" numbers (model size in KB, latency in
milliseconds) shown in the dashboard and usable in your pitch deck.

### 6. Launch the dashboard
```bash
streamlit run dashboard/app.py
```
Opens the interactive dashboard in your browser at **http://localhost:8501**

---

## What You'll See in the Dashboard

- **✈️ Aerospace tab** — pick a simulated engine unit, scrub through its
  operating cycles, and watch the model predict Remaining Useful Life in
  real time alongside actual sensor trends and a health status indicator.
- **⚙️ IHM tab** — pick a simulated machine, scrub through a streaming
  vibration feed, and watch the anomaly detector flag degradation before
  failure.
- **📊 Model Performance tab** — model accuracy metrics (RMSE/MAE/R²,
  ROC-AUC) plus the Edge Readiness benchmark table (model size before/after
  compression, inference latency) — proof the models are lightweight and
  fast enough for real on-device deployment.

---

## Troubleshooting

**"streamlit: command not found"**
Run `pip install streamlit` (or `pip install -r requirements.txt`) again,
then try `python -m streamlit run dashboard/app.py` instead.

**"Trained models not found" error in the dashboard**
Run steps 2–5 above (or `run_all.sh` / `run_all.bat`) before launching the
dashboard — the dashboard needs the generated data and trained models to
exist first.

**Port 8501 already in use**
Run: `streamlit run dashboard/app.py --server.port 8502` and open
`http://localhost:8502` instead.

**Using `python` vs `python3`**
On Windows, use `python`. On macOS/Linux, use `python3` if `python` is not
recognized.

---

## Project Structure

```
edge_ai_predictive_maintenance/
├── data/
│   ├── generate_data.py        # synthetic dataset generator
│   ├── aerospace_train.csv     # (pre-generated) CMAPSS-style training data
│   ├── aerospace_test.csv      # (pre-generated) CMAPSS-style test data
│   └── ihm_bearing.csv         # (pre-generated) CWRU-style vibration data
├── src/
│   ├── utils.py                 # shared feature engineering / benchmarking helpers
│   ├── train_rul_model.py       # trains the Aerospace RUL prediction model
│   ├── train_anomaly_model.py   # trains the IHM anomaly detection model
│   └── edge_optimize.py         # compresses models & benchmarks edge inference
├── models/                       # (pre-generated) trained models & metrics
├── dashboard/
│   └── app.py                    # Streamlit interactive dashboard
├── requirements.txt
├── run_all.sh                    # one-click setup+run (Linux/Mac)
├── run_all.bat                   # one-click setup+run (Windows)
├── README.md                     # project overview & pitch summary
└── HOW_TO_RUN.md                 # this file
```
