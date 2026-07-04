"""
app.py
======
Edge AI Predictive Maintenance & Anomaly Detection -- Interactive Dashboard

Two tabs:
  1. Aerospace (RUL Prediction)      -- CMAPSS-style turbofan engine data
  2. Industrial Heavy Machinery (Anomaly Detection) -- CWRU-style bearing data

Also shows an "Edge Readiness" panel with model size & inference latency,
proving the models are lightweight enough for real-time, on-device use.

Run:
    streamlit run dashboard/app.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from utils import add_rolling_features  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(
    page_title="Edge AI Predictive Maintenance",
    page_icon="🛠️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_rul_bundle():
    return joblib.load(os.path.join(MODEL_DIR, "rul_model.joblib"))


@st.cache_resource
def load_anomaly_bundle():
    return joblib.load(os.path.join(MODEL_DIR, "anomaly_model.joblib"))


@st.cache_data
def load_aerospace_test():
    df = pd.read_csv(os.path.join(DATA_DIR, "aerospace_test.csv"))
    return df


@st.cache_data
def load_ihm_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "ihm_bearing.csv"))
    return df


@st.cache_data
def load_edge_report():
    path = os.path.join(MODEL_DIR, "edge_benchmark_report.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


MODELS_READY = os.path.exists(os.path.join(MODEL_DIR, "rul_model.joblib")) and os.path.exists(
    os.path.join(MODEL_DIR, "anomaly_model.joblib")
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🛠️ Edge AI PdM System")
st.sidebar.markdown(
    "**Edge AI-based Predictive Maintenance & Anomaly Detection**\n\n"
    "Aerospace / Industrial Heavy Machinery"
)
st.sidebar.markdown("---")

edge_report = load_edge_report()
st.sidebar.subheader("⚡ Edge Readiness")
if edge_report:
    for key, label in [("rul_model", "RUL Model"), ("anomaly_model", "Anomaly Model")]:
        r = edge_report[key]
        st.sidebar.markdown(f"**{label}**")
        st.sidebar.metric(
            "Compressed size",
            f"{r['size_after_kb']:.0f} KB",
            delta=f"-{100 - 100*r['size_after_kb']/r['size_before_kb']:.0f}% vs uncompressed",
        )
        st.sidebar.caption(
            f"Avg inference latency: **{r['inference_latency_ms']['mean_ms']:.2f} ms**/sample "
            f"(p95: {r['inference_latency_ms']['p95_ms']:.2f} ms)"
        )
        st.sidebar.markdown("")
else:
    st.sidebar.info("Run `python src/edge_optimize.py` to generate edge benchmark stats.")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Models trained on synthetic datasets modeled after NASA CMAPSS "
    "(Aerospace) and CWRU Bearing (IHM) benchmarks -- no internet or "
    "physical hardware required."
)

if not MODELS_READY:
    st.error(
        "Trained models not found. Please run the setup steps first:\n\n"
        "```\npython data/generate_data.py\npython src/train_rul_model.py\n"
        "python src/train_anomaly_model.py\npython src/edge_optimize.py\n```"
    )
    st.stop()

rul_bundle = load_rul_bundle()
anomaly_bundle = load_anomaly_bundle()

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
st.title("Edge AI-based Predictive Maintenance & Anomaly Detection")
st.caption(
    "AI at the Edge Solutions for Industrial Heavy Machinery — "
    "Edge AI for Autonomous & Intelligent Heavy Machines"
)

tab1, tab2, tab3 = st.tabs(
    ["✈️ Aerospace — RUL Prediction", "⚙️ IHM — Anomaly Detection", "📊 Model Performance"]
)

# ---------------------------------------------------------------------------
# TAB 1 -- Aerospace RUL Prediction
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Remaining Useful Life (RUL) Prediction — Turbofan Engine Units")
    st.markdown(
        "Simulates a fleet of aircraft engines (CMAPSS-style). Select a unit "
        "to see its sensor trends and the model's predicted Remaining Useful "
        "Life at each operating cycle."
    )

    test_df = load_aerospace_test()
    unit_ids = sorted(test_df["unit_id"].unique())
    selected_unit = st.selectbox("Select engine unit", unit_ids)

    unit_df = test_df[test_df["unit_id"] == selected_unit].copy()
    unit_df = add_rolling_features(unit_df, [f"sensor_{i}" for i in range(1, 9)], "unit_id", 5)

    feature_cols = rul_bundle["feature_cols"]
    rul_cap = rul_bundle["rul_cap"]
    model = rul_bundle["model"]

    preds = model.predict(unit_df[feature_cols])
    unit_df["predicted_RUL"] = preds

    current_cycle = st.slider(
        "Current operating cycle",
        min_value=int(unit_df["cycle"].min()),
        max_value=int(unit_df["cycle"].max()),
        value=int(unit_df["cycle"].max() * 0.7),
    )

    row = unit_df[unit_df["cycle"] == current_cycle].iloc[0]
    predicted_rul = row["predicted_RUL"]
    actual_rul = row["RUL"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted RUL (cycles)", f"{predicted_rul:.0f}")
    col2.metric("Actual RUL (ground truth)", f"{actual_rul:.0f}")

    if predicted_rul > 60:
        health_status = "🟢 Healthy"
    elif predicted_rul > 25:
        health_status = "🟡 Monitor Closely"
    else:
        health_status = "🔴 Maintenance Required Soon"
    col3.metric("Health Status", health_status)

    # RUL curve chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=unit_df["cycle"], y=unit_df["RUL"].clip(upper=rul_cap),
        mode="lines", name="Actual RUL", line=dict(color="#2ca02c"),
    ))
    fig.add_trace(go.Scatter(
        x=unit_df["cycle"], y=unit_df["predicted_RUL"],
        mode="lines", name="Predicted RUL", line=dict(color="#d62728", dash="dash"),
    ))
    fig.add_vline(x=current_cycle, line_dash="dot", line_color="gray",
                  annotation_text="current cycle")
    fig.update_layout(
        xaxis_title="Operating Cycle", yaxis_title="RUL (cycles)",
        height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show raw sensor readings for this unit"):
        st.dataframe(unit_df[["cycle"] + [f"sensor_{i}" for i in range(1, 9)]], height=250)

# ---------------------------------------------------------------------------
# TAB 2 -- IHM Anomaly Detection
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Vibration-based Anomaly Detection — Heavy Machinery Bearings")
    st.markdown(
        "Simulates streaming vibration health features (CWRU bearing-style) "
        "from an industrial machine. The model is trained only on **normal** "
        "operating data and flags deviations as anomalies."
    )

    ihm_df = load_ihm_data()
    machine_ids = sorted(ihm_df["machine_id"].unique())
    selected_machine = st.selectbox("Select machine ID", machine_ids)

    machine_df = ihm_df[ihm_df["machine_id"] == selected_machine].copy()

    scaler = anomaly_bundle["scaler"]
    a_model = anomaly_bundle["model"]
    feat_cols = anomaly_bundle["feature_cols"]

    X_scaled = scaler.transform(machine_df[feat_cols])
    machine_df["anomaly_score"] = -a_model.decision_function(X_scaled)
    machine_df["predicted_anomaly"] = (a_model.predict(X_scaled) == -1).astype(int)

    timestep = st.slider(
        "Simulated time step (streaming position)",
        min_value=int(machine_df["timestep"].min()),
        max_value=int(machine_df["timestep"].max()),
        value=int(machine_df["timestep"].max() * 0.75),
    )

    visible_df = machine_df[machine_df["timestep"] <= timestep]
    current_row = visible_df.iloc[-1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Health State (ground truth)", current_row["health_state"].replace("_", " ").title())
    col2.metric("Anomaly Score", f"{current_row['anomaly_score']:.3f}")
    alert = "🚨 ANOMALY DETECTED" if current_row["predicted_anomaly"] == 1 else "✅ Normal"
    col3.metric("Model Alert", alert)

    if current_row["predicted_anomaly"] == 1:
        st.error(
            "⚠️ Anomaly detected — vibration signature deviates from healthy "
            "baseline. Recommend scheduling inspection/maintenance."
        )

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=visible_df["timestep"], y=visible_df["anomaly_score"],
        mode="lines", name="Anomaly Score", line=dict(color="#1f77b4"),
    ))
    anomaly_points = visible_df[visible_df["predicted_anomaly"] == 1]
    fig2.add_trace(go.Scatter(
        x=anomaly_points["timestep"], y=anomaly_points["anomaly_score"],
        mode="markers", name="Flagged Anomaly",
        marker=dict(color="red", size=6, symbol="x"),
    ))
    fig2.update_layout(
        xaxis_title="Time Step (streaming)", yaxis_title="Anomaly Score",
        height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Show raw vibration health features for this machine"):
        st.dataframe(
            visible_df[["timestep"] + feat_cols + ["health_state"]], height=250
        )

# ---------------------------------------------------------------------------
# TAB 3 -- Model performance / edge benchmark summary
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Model Performance & Edge Deployment Benchmark")

    rul_metrics_path = os.path.join(MODEL_DIR, "rul_metrics.joblib")
    anomaly_metrics_path = os.path.join(MODEL_DIR, "anomaly_metrics.joblib")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("### ✈️ Aerospace — RUL Model")
        if os.path.exists(rul_metrics_path):
            m = joblib.load(rul_metrics_path)
            st.metric("RMSE", f"{m['rmse']:.2f} cycles")
            st.metric("MAE", f"{m['mae']:.2f} cycles")
            st.metric("R² Score", f"{m['r2']:.3f}")
        else:
            st.info("Run `python src/train_rul_model.py` to generate metrics.")

    with colB:
        st.markdown("### ⚙️ IHM — Anomaly Detection Model")
        if os.path.exists(anomaly_metrics_path):
            m = joblib.load(anomaly_metrics_path)
            st.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
        else:
            st.info("Run `python src/train_anomaly_model.py` to generate metrics.")

    st.markdown("---")
    st.markdown("### ⚡ Edge Deployment Readiness")
    if edge_report:
        edge_df = pd.DataFrame([
            {
                "Model": edge_report["rul_model"]["model_name"],
                "Uncompressed Size (KB)": edge_report["rul_model"]["size_before_kb"],
                "Compressed Size (KB)": edge_report["rul_model"]["size_after_kb"],
                "Compression Ratio": edge_report["rul_model"]["compression_ratio"],
                "Avg Inference Latency (ms)": edge_report["rul_model"]["inference_latency_ms"]["mean_ms"],
            },
            {
                "Model": edge_report["anomaly_model"]["model_name"],
                "Uncompressed Size (KB)": edge_report["anomaly_model"]["size_before_kb"],
                "Compressed Size (KB)": edge_report["anomaly_model"]["size_after_kb"],
                "Compression Ratio": edge_report["anomaly_model"]["compression_ratio"],
                "Avg Inference Latency (ms)": edge_report["anomaly_model"]["inference_latency_ms"]["mean_ms"],
            },
        ])
        st.dataframe(edge_df, use_container_width=True, hide_index=True)
        st.caption(
            "Sub-10ms average inference latency per sample demonstrates real-time, "
            "on-device decision-making capability without dependency on continuous "
            "cloud connectivity -- the core requirement of Edge AI deployment."
        )
    else:
        st.info("Run `python src/edge_optimize.py` to generate the edge benchmark report.")

st.markdown("---")
st.caption(
    "Edge AI-based Predictive Maintenance and Anomaly Detection System | "
    "InnoVent-27 Prototype | Built with scikit-learn + Streamlit"
)
