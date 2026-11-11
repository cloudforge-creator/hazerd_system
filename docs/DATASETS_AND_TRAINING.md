# Datasets, download links and training scripts

Raw data stays out of Git. All commands run from the project folder in PowerShell with the virtual environment active.
**Check each dataset's licence** and note the version, class counts and seed in your report.

## What is already trained and included in this package
| Item | Where | Result |
|---|---|---|
| Fire CNN (MobileNetV2), trained on D-Fire on Colab | `models/fire_cnn.keras`, `models/fire_cnn_meta.json`, `models/fire_cnn_weights.npz`, `models/metrics/cnn_metrics.json` | 3-class test accuracy 84.1%, fire-vs-rest F1 0.881, recall 0.892, ROC AUC 0.974 (targets F1 0.90, recall 0.92: **not met**) |
| Metrics of the other studies (reference only) | `models/metrics/*.json`, `evaluation/results/` | see below; raw datasets and retrainable model files are not all bundled |

The ML hazard-state / early-warning models (`models/hazard_state.joblib`, `models/early_warning.joblib`) are **not** included: scikit-learn model files depend on the library version, so recreate them on the target machine (about 1 minute): `python -m ml.train_ml --runs 700 --seed 42`. Raw UCI, NAB, D-Fire and face datasets are not bundled. The package includes the fire CNN artifact and its reported metrics, not the original training images.

## One command for the downloadable datasets
```powershell
python scripts/get_datasets.py --links      # print all links and folders
python scripts/get_datasets.py --all        # UCI gas + NAB temperature + face models
```
UCI is fetched from the UCI repository; if that fails the script falls back to the copy in the project's GitHub repository.
(The UCI primary link could not be tested from the environment that prepared this package; the fallback was tested.)

## 1. Gas: UCI "Gas Sensor Array Drift"
- **What:** 13,910 samples, 16 metal-oxide sensors (128 features), 6 gases, 10 batches over about 3 years.
- **Link:** https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset  (dataset DOI: https://doi.org/10.24432/C5RP6W)
- **Folder:** `data/raw/gas/batch1.dat ... batch10.dat`
- **Commands:**
  ```powershell
  python -m ml.eval_uci_gas --data data/raw/gas             # baseline: 3 protocols (random split, next batch, early-to-late)
  python -m ml.train_gas_uci_pro --data data/raw/gas        # time-aware model selection + final held-out test + deployment model
  ```
- **Outputs when rerun:** `models/metrics/uci_gas_eval.json`, `models/metrics/gas_uci_training.json`, `models/gas_uci_model.joblib` (research model; not used by the runtime). The ZIP contains the recorded metrics JSON, not the raw batch files or the fitted gas-model artifact.
- **Time:** 1 minute and about 10 to 20 minutes (CPU).
- **Result so far:** random split 99.5%, next batch 80.9%, batches 1-5 to 6-10 52.2% (sensor drift). Selected model SVM (RBF, C=30); on the final batches 8-10 pooled accuracy 74.3% against 66.1% for the Random Forest baseline.
- **Caveat:** the runtime has a single MQ-2 channel, not the 128-feature array; this validates the method only.

## 2. Temperature: NAB machine temperature
- **What:** 22,695 points at 5-minute intervals with 4 labelled anomaly windows (test split holds 2).
- **Link:** https://github.com/numenta/NAB  (`data/realKnownCause/machine_temperature_system_failure.csv`, `labels/combined_windows.json`)
- **Commands:**
  ```powershell
  python -m ml.train_temp_lstm --download-nab               # downloads NAB to data/raw/temperature/
  python -m ml.train_temp_lstm --csv data/raw/temperature/machine_temperature_system_failure.csv --windows data/raw/temperature/combined_windows.json --window-key realKnownCause/machine_temperature_system_failure.csv --seed 42
  ```
- **Outputs when rerun:** `models/temp_lstm.keras`, `models/temp_lstm_meta.json`, `models/metrics/temp_lstm_NAB.json`. The ZIP contains the reported metrics JSON, not the raw NAB data or the fitted LSTM artifact.
- **Time:** about 5 to 15 minutes (CPU).
- **Result so far:** LSTM detects 2 of 2 windows with 0.338 false alarms per day; the persistence baseline gives the same event counts and a rolling z-score has fewer false alarms. **No advantage, so the LSTM is not used at runtime.**
- **Later:** retrain on your own device logs (`--source device`) and enable only if it beats the baselines.

## 3. Fire: D-Fire (already trained once; links to repeat or improve)
- **What:** 21,000+ images, YOLO labels, class ids 0 = smoke, 1 = fire. Authors: GAIA, https://github.com/gaiasd/DFireDataset
- **Links (from the authors' page):**
  - Images and labels (OneDrive): https://1drv.ms/u/c/c0bd25b6b048b01d/EbLgD7bES4FDvUN37Grxn8QBF5gIBBc7YV2qklF08GCiBw
  - Train / validation / test sets (OneDrive): https://1drv.ms/f/c/c0bd25b6b048b01d/Ema8FFze8mFIlM1Hn81BUUgBE3vnnmK4SQxybS-nHRt2pA?e=6rk0aN
  - Kaggle copy, ready to use: https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo
- **Folder:** `data/raw/fire/<any>/images/*.jpg` with sibling `<any>/labels/*.txt`
- **Where to train:** on a free Colab GPU: open `notebooks/Fire_CNN_Training_Colab.ipynb` (details in `docs/COLAB.md`). CPU training takes hours.
- **Commands (what the notebook runs):**
  ```powershell
  python -m ml.prepare_fire_dataset --root data/raw/fire --out data/fire_manifest.csv --fire-id 1 --smoke-id 0 --seed 42
  python -m ml.train_cnn --manifest data/fire_manifest.csv --epochs-head 5 --epochs-ft 5 --seed 42
  python -m ml.export_fire_weights --model models/fire_cnn.keras      # portable weights for any TensorFlow version
  ```
- **Then measure it properly (before trusting it for a shutdown):**
  ```powershell
  python -m ml.eval_cnn_operating_points --manifest data/fire_manifest.csv
  python -m ml.eval_cnn_operating_points --manifest data/fire_manifest.csv --webcam-neg data/webcam/negatives --webcam-pos data/webcam/fire
  ```
- **Included outputs:** `models/fire_cnn.keras`, `models/fire_cnn_weights.npz`, `models/fire_cnn_meta.json`, `models/metrics/cnn_metrics.json`. `cnn_operating_points.json` is generated only after running the evaluation script; it is not included in this ZIP.
- **Ideas to improve (judge on validation only):** 10 to 15 fine-tuning epochs, 288 or 320 px input, label smoothing, a few hundred labelled frames from your own camera, temperature scaling for calibration.

## 4. Faces: models and your own photos
- **Models:** YuNet (detector) and SFace (recogniser) from the OpenCV Zoo: https://github.com/opencv/opencv_zoo/tree/main/models
  ```powershell
  python scripts/download_face_models.py        # or: python scripts/get_datasets.py --face-models
  python scripts/check_face_setup.py
  ```
- **Your data:** photos of your own team **with consent**: `data/faces/<Person name>/*.jpg` (3 or more people, 5 or more photos each). Keep them out of Git.
  ```powershell
  python -m ml.eval_faces        # false-accept rate per threshold and a suggested threshold
  ```

## 5. Your own hardware data (the most valuable dataset)
- When the sensors run, `data/logs/sensor_log.csv` records every cycle. Label real events (and normal periods), then retrain:
  ```powershell
  python -m ml.train_ml --runs 700 --seed 42        # synthetic bootstrap; replace with real logs as they accumulate
  python -m ml.train_temp_lstm --source device ...   # see the header of ml/train_temp_lstm.py
  ```
- Add 1 to 2 hours of normal webcam footage (consecutive frames, no fire) and a few minutes with a supervised small flame for the fire-CNN false-alarm test above.

## Suggested order
1. `python -m ml.train_ml --runs 700 --seed 42` (needed for the ML signals)
2. `python scripts/get_datasets.py --all`, then the UCI and NAB commands (optional research studies)
3. Re-run `python -m evaluation.run_all`
4. Webcam false-alarm test for the fire CNN; keep the CNN-only CRITICAL guard enabled
5. Hardware bring-up and real data logging, then retrain and repeat the evaluation
