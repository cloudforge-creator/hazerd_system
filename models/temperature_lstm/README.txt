TEMPERATURE LSTM MODEL RELEASE
==============================

Model:
Temperature LSTM Anomaly Detector

Version:
1.0.1-event-based

Dataset:
NAB machine_temperature_system_failure.csv

Dataset type:
Industrial machine temperature telemetry

Sampling interval:
5 minutes

Lookback:
50 previous temperature readings


RELEASE FILES
-------------

- temperature_lstm.keras
- temperature_scaler.joblib
- metadata.json
- evaluation.json
- README.txt


INFERENCE PROCEDURE
-------------------

1. Collect the previous 50 temperature readings.

2. Transform those readings using the saved
   temperature_scaler.joblib MinMaxScaler.

3. Pass the scaled sequence to temperature_lstm.keras.

4. Obtain the predicted next temperature in scaled units.

5. Transform the actual current temperature using
   the SAME saved scaler.

6. Calculate absolute prediction error in scaled units:

   error = abs(actual_scaled - predicted_scaled)

7. Compare the error against the frozen threshold
   stored in metadata.json.

8. If:

   error >= threshold

   classify the reading as an anomaly.

9. Inverse-transform the predicted temperature only
   when displaying it in the original temperature unit.


IMPORTANT THRESHOLD INFORMATION
--------------------------------

Threshold:
0.0506867678281483

Threshold unit:
MinMax-scaled absolute prediction error.

Do not compare this threshold against a temperature
value in degrees Celsius.

Do not calculate the anomaly error after inverse
transforming the values.


EVALUATION
----------

Final event-based test:

- Test readings: 3463
- Normal readings: 2896
- Anomaly readings: 567
- Correctly detected anomalies: 510
- Missed anomalies: 57
- False alarms: 94

Accuracy: approximately 95.64%
Anomaly precision: approximately 84.44%
Anomaly recall: approximately 89.95%
Anomaly F1-score: approximately 87.11%
ROC-AUC: approximately 95.51%


EVALUATION LIMITATIONS
----------------------

- Threshold calibration history includes training data.
- Threshold calibration is not an independent validation set.
- Only one complete anomaly event was held out for final testing.
- NAB industrial machine telemetry is not calibrated
  physical temperature sensor data.
- Actual hardware validation is still required.


PROJECT INTEGRATION
-------------------

This is a research/evaluation artifact.

It is not the device-trained runtime temperature model.

The live system uses a 0.5-second sensor cycle.
This research model uses 5-minute historical readings.

Do not directly connect this model to the live sensor
pipeline without a separate, properly designed and
validated integration.

This model is not a certified fire safety device.

ADDITIONAL RELEASE ARTIFACTS
----------------------------

The complete project release also includes:

- temperature_lstm.keras
  Original exported Keras model.

- temperature_lstm_keras3_compat.keras
  Compatibility copy used for local inference with the
  project's installed Keras 3 environment. The original
  model archive is preserved unchanged.

- temperature_scaler.joblib
  Saved MinMaxScaler fitted on normal training readings.

- metadata.json
  Model configuration, dataset information, threshold,
  training/test periods and limitations.

- evaluation.json
  Saved event-based test metrics.

- reproduction_report.json
  Independent local reproduction results. All saved
  evaluation metrics were reproduced successfully.

- README.txt
  Model usage and evaluation documentation.


PROJECT SOURCE FILES
--------------------

The project also includes:

- src/research_temperature_lstm.py
  Separate research inference module.

- scripts/verify_temperature_lstm.py
  Independent test reproduction script.

- data/reference/nab/combined_windows.json
  Official NAB anomaly-window labels used for evaluation.


COMPATIBILITY NOTE
------------------

The scaler was originally saved using scikit-learn 1.6.1.
Local verification was performed with scikit-learn 1.7.2.
The scaler loaded and reproduced the saved evaluation
successfully, but the version difference should be
considered when reproducing this environment.


VERIFICATION STATUS
-------------------

Independent local evaluation:
PASS

Saved evaluation metrics matched:
12 of 12

Confusion matrix:
[[2802, 94], [57, 510]]

No model retraining or threshold modification was
performed during local reproduction.
