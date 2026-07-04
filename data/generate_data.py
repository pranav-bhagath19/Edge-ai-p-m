"""
generate_data.py
================
Generates realistic SYNTHETIC sensor time-series datasets that mimic the
structure and degradation behaviour of two well-known public benchmarks:

  1. NASA CMAPSS turbofan engine degradation dataset  -> Aerospace track
  2. CWRU bearing fault dataset                        -> Industrial Heavy
                                                           Machinery (IHM) track

We generate synthetic data (instead of downloading the originals) so that
this project runs 100% offline, with no external downloads, no dataset
licensing issues, and no internet dependency at judging time. The
generation logic follows the same physical intuition as the real datasets:
sensor readings drift/degrade as a unit approaches failure, with realistic
noise, unit-to-unit variability, and injected fault events.

Run:
    python generate_data.py

Outputs (written to this data/ folder):
    aerospace_train.csv
    aerospace_test.csv
    ihm_bearing.csv
"""

import numpy as np
import pandas as pd
import os

RNG = np.random.default_rng(42)
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# 1. AEROSPACE (CMAPSS-style) run-to-failure engine degradation data
# ---------------------------------------------------------------------------
def generate_aerospace_data(n_units=40, n_sensors=8, min_life=120, max_life=320):
    """
    Simulates multiple turbofan engine units run to failure.
    Each unit has a randomly sampled total lifetime (in operating cycles).
    Sensor values drift (mostly increasing degradation signatures) as the
    unit approaches end-of-life, with unit-specific severity and noise.
    """
    rows = []
    for unit_id in range(1, n_units + 1):
        life = RNG.integers(min_life, max_life)
        # unit-specific degradation severity / initial wear / noise profile
        severity = RNG.uniform(0.8, 1.3)
        noise_level = RNG.uniform(0.5, 1.5)

        # baseline operating settings (like CMAPSS op settings)
        op_setting_1 = RNG.uniform(-0.005, 0.005)
        op_setting_2 = RNG.uniform(-0.0005, 0.0005)
        op_setting_3 = 100.0

        for cycle in range(1, life + 1):
            frac = cycle / life  # 0 -> healthy, 1 -> failure
            rul = life - cycle

            sensor_values = {}
            for s in range(1, n_sensors + 1):
                base = 400 + s * 15
                # non-linear degradation curve (accelerates near end-of-life)
                degradation = severity * (frac ** (2 + 0.3 * (s % 3))) * (20 + 3 * s)
                noise = RNG.normal(0, noise_level * (1 + s * 0.05))
                sensor_values[f"sensor_{s}"] = base + degradation + noise

            rows.append({
                "unit_id": unit_id,
                "cycle": cycle,
                "op_setting_1": op_setting_1,
                "op_setting_2": op_setting_2,
                "op_setting_3": op_setting_3,
                **sensor_values,
                "RUL": rul,
                "life": life,
            })
    return pd.DataFrame(rows)


def split_train_test(df, test_fraction=0.2):
    unit_ids = df["unit_id"].unique()
    RNG.shuffle(unit_ids)
    n_test = max(1, int(len(unit_ids) * test_fraction))
    test_units = set(unit_ids[:n_test])
    train_df = df[~df["unit_id"].isin(test_units)].reset_index(drop=True)
    test_df = df[df["unit_id"].isin(test_units)].reset_index(drop=True)
    return train_df, test_df


# ---------------------------------------------------------------------------
# 2. INDUSTRIAL HEAVY MACHINERY (CWRU bearing-style) vibration data
# ---------------------------------------------------------------------------
def generate_ihm_bearing_data(n_machines=15, n_samples_per_machine=400, fs=1000):
    """
    Simulates vibration sensor readings from industrial bearings/machines
    over their operating life. Extracts standard vibration health features
    (RMS, kurtosis, crest factor, peak-to-peak, std) the way CWRU-style
    condition monitoring pipelines do, with a gradual transition from
    'normal' -> 'early fault' -> 'severe fault'.
    """
    rows = []
    for machine_id in range(1, n_machines + 1):
        life = n_samples_per_machine
        severity = RNG.uniform(0.8, 1.4)

        for t in range(life):
            frac = t / life
            # health state labeling
            if frac < 0.55:
                state = "normal"
            elif frac < 0.85:
                state = "early_fault"
            else:
                state = "severe_fault"

            # simulate a short vibration waveform segment and compute features
            n_pts = 256
            base_freq = 50 + RNG.uniform(-2, 2)
            time_axis = np.arange(n_pts) / fs
            fault_amplitude = severity * (frac ** 3) * 4.0
            signal = (
                0.5 * np.sin(2 * np.pi * base_freq * time_axis)
                + fault_amplitude * np.sin(2 * np.pi * base_freq * 5 * time_axis)
                + RNG.normal(0, 0.15 + 0.3 * frac, n_pts)
            )

            rms = np.sqrt(np.mean(signal ** 2))
            std = np.std(signal)
            peak = np.max(np.abs(signal))
            crest_factor = peak / (rms + 1e-8)
            kurtosis = (
                np.mean((signal - signal.mean()) ** 4) / (std ** 4 + 1e-8)
                if std > 0 else 0
            )
            skewness = (
                np.mean((signal - signal.mean()) ** 3) / (std ** 3 + 1e-8)
                if std > 0 else 0
            )
            peak_to_peak = signal.max() - signal.min()

            rows.append({
                "machine_id": machine_id,
                "timestep": t,
                "rms": rms,
                "std": std,
                "peak": peak,
                "crest_factor": crest_factor,
                "kurtosis": kurtosis,
                "skewness": skewness,
                "peak_to_peak": peak_to_peak,
                "health_state": state,
                "is_anomaly": 0 if state == "normal" else 1,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Generating synthetic Aerospace (CMAPSS-style) dataset ...")
    aerospace_df = generate_aerospace_data()
    train_df, test_df = split_train_test(aerospace_df)
    train_df.to_csv(os.path.join(HERE, "aerospace_train.csv"), index=False)
    test_df.to_csv(os.path.join(HERE, "aerospace_test.csv"), index=False)
    print(f"  aerospace_train.csv -> {train_df.shape}")
    print(f"  aerospace_test.csv  -> {test_df.shape}")

    print("Generating synthetic IHM Bearing (CWRU-style) dataset ...")
    ihm_df = generate_ihm_bearing_data()
    ihm_df.to_csv(os.path.join(HERE, "ihm_bearing.csv"), index=False)
    print(f"  ihm_bearing.csv -> {ihm_df.shape}")

    print("\nData generation complete. Files saved in the data/ folder.")
