import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = Path.home() / "Downloads" / "machine_temperature_system_failure (1).csv"

LABEL_PATH = ROOT / "data" / "reference" / "nab" / "combined_windows.json"

MODEL_DIR = ROOT / "models" / "temperature_lstm"

MODEL_PATH = MODEL_DIR / "temperature_lstm_keras3_compat.keras"
SCALER_PATH = MODEL_DIR / "temperature_scaler.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
EVALUATION_PATH = MODEL_DIR / "evaluation.json"

REPORT_PATH = MODEL_DIR / "reproduction_report.json"


def main():

    print("\n===== 1. CHECKING INPUT FILES =====")

    for path in [
        CSV_PATH,
        LABEL_PATH,
        MODEL_PATH,
        SCALER_PATH,
        METADATA_PATH,
        EVALUATION_PATH
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        print(f"FOUND: {path.name}")

    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)

    with open(EVALUATION_PATH, encoding="utf-8") as f:
        saved = json.load(f)

    print("\n===== 2. LOADING DATASET =====")

    raw = pd.read_csv(CSV_PATH)

    raw["timestamp"] = pd.to_datetime(
        raw["timestamp"],
        errors="raise"
    )

    raw["value"] = pd.to_numeric(
        raw["value"],
        errors="raise"
    )

    original_rows = len(raw)

    clean_df = (
        raw.groupby("timestamp", as_index=False)
        .agg(value=("value", "mean"))
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(f"Original rows: {original_rows}")
    print(f"Cleaned rows:  {len(clean_df)}")
    print(f"Removed duplicates: {original_rows - len(clean_df)}")

    print("\n===== 3. APPLYING OFFICIAL NAB LABELS =====")

    with open(LABEL_PATH, encoding="utf-8") as f:
        official_labels = json.load(f)

    dataset_key = (
        "realKnownCause/"
        "machine_temperature_system_failure.csv"
    )

    if dataset_key not in official_labels:
        raise ValueError("NAB dataset key is missing.")

    windows = official_labels[dataset_key]

    clean_df["is_anomaly"] = 0

    window_counts = []

    for index, window in enumerate(windows, start=1):

        start = pd.to_datetime(window[0])
        end = pd.to_datetime(window[1])

        mask = (
            (clean_df["timestamp"] >= start)
            & (clean_df["timestamp"] <= end)
        )

        clean_df.loc[mask, "is_anomaly"] = 1

        count = int(mask.sum())
        window_counts.append(count)

        print(
            f"Window {index}: "
            f"{start} -> {end} | "
            f"Readings: {count}"
        )

    print("\n===== 4. RECONSTRUCTING EVENT-BASED SPLIT =====")

    test_start = pd.to_datetime(
        metadata["test_period"]["start"]
    )

    test_end = pd.to_datetime(
        metadata["test_period"]["end"]
    )

    calibration_df = clean_df[
        clean_df["timestamp"] < test_start
    ].copy()

    test_df = clean_df[
        (clean_df["timestamp"] >= test_start)
        & (clean_df["timestamp"] <= test_end)
    ].copy()

    print(f"Calibration rows: {len(calibration_df)}")
    print(f"Test rows:        {len(test_df)}")
    print(
        "Test anomalies:   "
        f"{int(test_df['is_anomaly'].sum())}"
    )

    if len(test_df) != metadata["test_rows"]:
        raise ValueError("Test row count does not match metadata.")

    if len(test_df) != saved["test_readings"]:
        raise ValueError("Test row count does not match evaluation.")

    if int(test_df["is_anomaly"].sum()) != saved["anomaly_readings"]:
        raise ValueError("Test anomaly count does not match evaluation.")

    lookback = int(metadata["lookback"])
    threshold = float(metadata["threshold"])

    if len(calibration_df) < lookback:
        raise ValueError("Insufficient historical context.")

    print("\n===== 5. LOADING MODEL AND SCALER =====")

    scaler = joblib.load(SCALER_PATH)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    if model.input_shape[1] != lookback:
        raise ValueError("Model lookback mismatch.")

    print(f"Model input: {model.input_shape}")
    print(f"Lookback: {lookback}")
    print(f"Threshold: {threshold}")

    print("\n===== 6. BUILDING TEST SEQUENCES =====")

    # Include only the 50 historical readings immediately
    # before the test period as input context.
    context = pd.concat(
        [
            calibration_df.tail(lookback),
            test_df
        ],
        ignore_index=True
    )

    scaled_values = scaler.transform(
        context[["value"]]
    ).astype(np.float32)

    X_test = np.array(
        [
            scaled_values[i - lookback:i]
            for i in range(lookback, len(scaled_values))
        ],
        dtype=np.float32
    )

    actual_scaled = scaled_values[lookback:, 0]

    y_true = test_df["is_anomaly"].to_numpy(dtype=int)

    print(f"X_test shape: {X_test.shape}")
    print(f"Actual values: {len(actual_scaled)}")
    print(f"Ground-truth labels: {len(y_true)}")

    if len(X_test) != len(test_df):
        raise ValueError("Test sequence count mismatch.")

    print("\n===== 7. RUNNING MODEL INFERENCE =====")

    predicted_scaled = model.predict(
        X_test,
        verbose=0
    ).reshape(-1)

    errors = np.abs(
        actual_scaled - predicted_scaled
    )

    y_pred = (
        errors >= threshold
    ).astype(int)

    print("Inference completed.")

    print("\n===== 8. CALCULATING METRICS =====")

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred, zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    auc = roc_auc_score(
        y_true, errors
    )

    reproduced = {
        "accuracy": float(accuracy),
        "anomaly_precision": float(precision),
        "anomaly_recall": float(recall),
        "anomaly_f1": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "test_readings": int(len(y_true)),
        "normal_readings": int((y_true == 0).sum()),
        "anomaly_readings": int((y_true == 1).sum()),
        "correctly_detected_anomalies": int(
            ((y_true == 1) & (y_pred == 1)).sum()
        ),
        "missed_anomalies": int(
            ((y_true == 1) & (y_pred == 0)).sum()
        ),
        "false_alarms": int(
            ((y_true == 0) & (y_pred == 1)).sum()
        )
    }

    print("\n===== REPRODUCED RESULTS =====")

    for key, value in reproduced.items():
        print(f"{key}: {value}")

    print("\n===== 9. COMPARING WITH SAVED EVALUATION =====")

    comparison_keys = [
        "accuracy",
        "anomaly_precision",
        "anomaly_recall",
        "anomaly_f1",
        "roc_auc",
        "confusion_matrix",
        "test_readings",
        "normal_readings",
        "anomaly_readings",
        "correctly_detected_anomalies",
        "missed_anomalies",
        "false_alarms"
    ]

    mismatches = []

    for key in comparison_keys:

        expected = saved[key]
        actual = reproduced[key]

        if key == "confusion_matrix":
            matches = expected == actual
        elif isinstance(expected, (int, float)):
            matches = np.isclose(
                expected,
                actual,
                rtol=1e-6,
                atol=1e-8
            )
        else:
            matches = expected == actual

        status = "MATCH" if matches else "MISMATCH"

        print(
            f"{key}: {status} | "
            f"saved={expected} | reproduced={actual}"
        )

        if not matches:
            mismatches.append(key)

    report = {
        "verification_type": "independent_event_based_reproduction",
        "model_version": metadata["version"],
        "dataset": metadata["dataset"],
        "sampling_interval": metadata["sampling_interval"],
        "threshold": threshold,
        "official_anomaly_window_counts": window_counts,
        "saved_evaluation": saved,
        "reproduced_evaluation": reproduced,
        "mismatches": mismatches,
        "all_metrics_match": len(mismatches) == 0,
        "limitations": [
            "Threshold calibration overlaps training history.",
            "Only one complete anomaly event is held out for testing.",
            "NAB telemetry is not calibrated physical sensor data.",
            "This research model is not integrated into the live 0.5-second sensor pipeline."
        ]
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\n===== FINAL VERIFICATION STATUS =====")

    if not mismatches:
        print("PASS: All saved metrics reproduced.")
    else:
        print("REVIEW REQUIRED: Some metrics did not match.")
        print("Mismatched fields:", mismatches)

    print(f"\nReport saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
