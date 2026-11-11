# Changes from the fire-CNN integration review

1. **Model location (critical).** `fire_cnn.keras`, `fire_cnn_meta.json` and `metrics/cnn_metrics.json` were in
   `models/experiment_2_import/`, but the code reads `models/`. The fire CNN was silently NOT loaded
   (`[ai] fire CNN not loaded (no model found)`). Files are now in `models/` and `models/metrics/`.
2. **Train/serve defect (high).** `src/fire_cnn.py` called `self.model(x)`. The saved model contains random
   augmentation layers (flip, rotation, zoom, contrast) which Keras 3 leaves ACTIVE in that call, so the same frame
   gave different probabilities each time, while the metrics were computed with `model.predict` (inference mode).
   Now `self.model(x, training=False)`. Regression test: `tests/test_dl.py::test_fire_cnn_predict_is_deterministic_with_augmentation_layers`
   (fails on the old code, passes on the new). The fake model in `test_fire_signals_average_and_camera_fault` now accepts `training=`.
3. **New tool.** `python -m ml.eval_cnn_operating_points` reports precision / recall / false-positive rate with 95% intervals at the
   engine's own fire thresholds (0.38, 0.40, 0.70, 0.80), calibration (ECE, reliability), runtime-vs-training preprocessing parity,
   and false alarms per hour on your own webcam footage (`--webcam-neg`, `--webcam-pos`).
4. `requirements-ml.txt`: note that the model was trained with TensorFlow 2.21.0.
5. **Load failure on a different TensorFlow (found when you ran it).** `[ai] fire CNN failed to load (TypeError) - camera ignored`
   means the local Keras cannot read a file saved by the newer Colab version. `src/fire_cnn.py` now (a) prints the real error text and
   (b) falls back to `models/fire_cnn_weights.npz`, plain NumPy weights exported from the same `.keras` (checked by SHA-256, so stale weights are refused).
   The fallback model reproduces the original to 2e-7 on 24 images. Create or refresh the file with `python -m ml.export_fire_weights`
   wherever the `.keras` loads (Colab right after training). New test: `test_fire_cnn_portable_weights_fallback_matches_original`.

Final verification: 150 tests passed in a full run (raw output in 2_Documentation/pytest_raw_output.txt). The CNN guard now also holds when the CRITICAL threshold or the corroboration setting is edited (src/decision_engine.py `_cnn_solo_cap`, `_cnn_corroborated`; constant `CNN_CORROBORATION_MIN`).
