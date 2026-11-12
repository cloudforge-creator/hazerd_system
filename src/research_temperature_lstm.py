"""
Research Temperature LSTM Inference Module

Purpose:
    Evaluate the NAB-trained temperature LSTM independently.

Important:
    - Uses 5-minute historical temperature readings.
    - Requires 50 previous readings.
    - Uses the saved MinMaxScaler.
    - Calculates prediction error in scaled units.
    - Does not integrate with the live 0.5-second sensor pipeline.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "temperature_lstm"

MODEL_PATH = MODEL_DIR / "temperature_lstm_keras3_compat.keras"
SCALER_PATH = MODEL_DIR / "temperature_scaler.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"


class ResearchTemperatureLSTM:

    def __init__(self):

        required_files = [
            MODEL_PATH,
            SCALER_PATH,
            METADATA_PATH
        ]

        missing = [
            str(path)
            for path in required_files
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                f"Missing research model files: {missing}"
            )

        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            self.metadata = json.load(file)

        if self.metadata.get("threshold_units") != (
            "minmax_scaled_absolute_prediction_error"
        ):
            raise ValueError("Unexpected threshold units.")

        if self.metadata.get("sampling_interval") != "5 minutes":
            raise ValueError("Unexpected model sampling interval.")

        self.lookback = int(self.metadata["lookback"])
        self.threshold = float(self.metadata["threshold"])

        self.scaler = joblib.load(SCALER_PATH)

        self.model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        if self.model.input_shape[1] != self.lookback:
            raise ValueError("Model input shape does not match lookback.")

        if self.model.input_shape[2] != 1:
            raise ValueError("Unexpected model feature dimension.")

        print("[research LSTM] Model loaded successfully.")
        print(f"[research LSTM] Lookback: {self.lookback}")
        print("[research LSTM] Sampling interval: 5 minutes")
        print(f"[research LSTM] Threshold: {self.threshold}")

    def predict_next(self, previous_readings, actual_reading):

        readings = np.asarray(
            previous_readings,
            dtype=np.float32
        ).reshape(-1, 1)

        if len(readings) != self.lookback:
            raise ValueError(
                f"Expected {self.lookback} previous readings, "
                f"received {len(readings)}."
            )

        if not np.all(np.isfinite(readings)):
            raise ValueError("Historical readings contain invalid values.")

        actual_reading = float(actual_reading)

        if not np.isfinite(actual_reading):
            raise ValueError("Actual reading must be finite.")

        # Scale the previous 50 readings.
        scaled_window = self.scaler.transform(readings)

        # Model predicts the next temperature in scaled units.
        model_input = scaled_window.reshape(
            1, self.lookback, 1
        )

        predicted_scaled = float(
            self.model.predict(model_input, verbose=0)[0][0]
        )

        # Scale the actual current reading using the SAME scaler.
        actual_scaled = float(
            self.scaler.transform(
                np.array([[actual_reading]], dtype=np.float32)
            )[0][0]
        )

        # Calculate error in scaled units.
        prediction_error = abs(
            actual_scaled - predicted_scaled
        )

        # Apply the frozen research threshold.
        is_anomaly = prediction_error >= self.threshold

        # Inverse transform prediction for display only.
        predicted_temperature = float(
            self.scaler.inverse_transform(
                np.array([[predicted_scaled]])
            )[0][0]
        )

        return {
            "predicted_temperature": predicted_temperature,
            "actual_temperature": actual_reading,
            "prediction_error_scaled": prediction_error,
            "threshold": self.threshold,
            "is_anomaly": bool(is_anomaly),
            "classification": (
                "ANOMALY" if is_anomaly else "NORMAL"
            ),
            "sampling_interval": "5 minutes"
        }


if __name__ == "__main__":

    detector = ResearchTemperatureLSTM()

    # Structural smoke test only.
    # These values are NOT NAB test data and do NOT establish accuracy.
    sample_window = np.full(
        detector.lookback,
        70.0,
        dtype=np.float32
    )

    result = detector.predict_next(
        previous_readings=sample_window,
        actual_reading=70.0
    )

    print("\n===== RESEARCH LSTM SMOKE TEST =====")

    for key, value in result.items():
        print(f"{key}: {value}")

    print("\nSmoke test completed.")
