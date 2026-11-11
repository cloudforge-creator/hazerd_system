# Predictive AI Hazard Detection and Prevention System

Built phase by phase from the project WBS. **Software only for now - the simulator stands in for the hardware.**
When the hardware arrives: flash `arduino/hazard_sensors/hazard_sensors.ino`, then run with `--port`.

## Status

**Delivery scope:** software implementation and simulator evaluation. Physical hardware, real-camera validation, live notification delivery and field safety qualification are not included in the verified software baseline. The system is an academic prototype, not certified industrial safety equipment.
| Phase | WBS | Status |
|---|---|---|
| M1 data collection, simulator, validation | 4.0 | done |
| M3 decision engine, fusion, early-warning baseline | 6.0 | done (rule-based; AI scores plug in via `Signals`) |
| M4 response: buzzer, notifications, latched relay shutdown | 7.0 | done (notifier is a log stub) |
| Storage (SQLite) and end-to-end pipeline | 3.4, 9.x | done |
| M2a ML hazard-state + early-warning models (synthetic bootstrap), UCI gas validation script | 5.0 | done |
| M2b Temperature LSTM (NAB validated; device retrain pending) | 5.0 | done (see honest result below) |
| M2c Fire CNN: leak-safe manifest, train, runtime camera | 5.0 | **trained on D-Fire (Colab): test F1 0.881, recall 0.892, below the 0.90 / 0.92 targets**; loaded from `models/fire_cnn.keras` (portable `fire_cnn_weights.npz` fallback); not yet validated on your own camera. CNN-only evidence is capped below CRITICAL until a physical sensor corroborates it. |
| M5 Flask admin dashboard: login, roles, live monitoring, alerts, logs, config, camera, personnel, users | 8.0 | implemented; package test records describe HTTP integration checks. Browser visual review and live camera operation still require a local run; face accuracy is unmeasured |
| Testing/evaluation C1-C5, docs | 10.0, 12.0 | suite built and run on the project machine (`evaluation/results/REPORT.md`); real-hardware data pending |

## Setup (Windows PowerShell)
Use Python 3.10 for the project environment. The current packaging review passed compilation and 13 focused decision-engine tests; the complete test suite must be rerun after this release change in an environment with the project requirements installed.
```powershell
py -3.10 -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
```

## Run
```powershell
python -m src.main --sim --scenario normal   --duration 20
python -m src.main --sim --scenario gas_leak --duration 40
python -m src.main --sim --scenario overheat --duration 70
python -m src.main --sim --scenario fire     --duration 30
python -m src.main --port COM3               # real hardware, later
```
Outputs: `data/hazard.db` (SQLite) and `data/logs/sensor_log.csv` (also the raw material for ML datasets).

## Train the ML models (about 1 minute, CPU only)
```powershell
python -m ml.train_ml --runs 700 --seed 42
python -m src.main --sim --scenario gas_leak --duration 70     # prints "[ai] ML models loaded"
python -m src.main --sim --scenario gas_leak --no-ml           # rule-based baseline for comparison
```
Writes `models/hazard_state.joblib`, `models/early_warning.joblib` and `models/metrics/ml_metrics.json`.
If the model files are missing the system says so and runs rule-based only.

**These models are trained on SYNTHETIC scenarios.** The metrics show the pipeline works and how robust it is to the
effects modelled (humidity/temperature cross-sensitivity, benign spikes, noise). They are not field accuracy.
Retrain on logged hardware data once the sensors arrive.

### Real-data validation of the gas method (UCI)
Download `batch1.dat ... batch10.dat` into `data/raw/gas/`, then `python -m ml.eval_uci_gas`. Reports random-split,
next-batch (drift) and 1-5 -> 6-10 results. Quote the drift results, not just the random split.

## Evaluation suite (WBS 10: C1-C5)
```powershell
python -m ml.train_ml                       # models must exist, otherwise only the rule-based system is evaluated
python -m evaluation.run_all --quick        # smoke run, about 1 minute
python -m evaluation.run_all                # full run, roughly 10-15 minutes -> evaluation/results/REPORT.md + results.json
```
It drives the REAL pipeline (not mocks) and reports each measure against a target, marking FAIL honestly.
- **C1** state classification + critical-event detection; **C2** early-warning detection, lead time, false alarms;
  **C3** latency (full cycle incl. disk DB, decision only, AI inference only); **C4** alarm/notification/shutdown
  success and timing, false shutdowns; **C5** 1-hour accelerated soak (heap growth, DB growth), sensor-node scaling,
  dashboard load with concurrent clients, fault-injection stress.
- Read the "What this does and does not prove" section of the report before quoting numbers. In particular: C1/C2/C4
  use synthetic scenarios whose labels share the rule engine's thresholds (circular for the rules), C4 counts relay
  commands issued (not physical actuation), and C3/C5 timings are specific to the machine that ran them.
- Re-run on real logged data after the hardware arrives and quote those results.
- `pip install psutil` (optional) adds process memory (RSS) to the soak result.

## Web dashboard (admin module)
```powershell
python -m src.users add admin --role admin        # first, create a user (no default account exists)
python -m src.users add operator1 --role viewer
python -m src.main --sim --scenario gas_leak --dashboard      # then open http://127.0.0.1:5000
```
- **viewer**: sees status, charts, alerts, logs, settings. **admin**: also acknowledges alerts, edits thresholds, re-arms the machine.
- Re-arming is refused unless the system is SAFE. Threshold edits are validated, applied live, saved to
  `data/runtime_config.json` and written to the system log with the user's name.
- Security: hashed passwords, CSRF token on every POST, login lockout (5 failures -> 60 s), strict CSP, no inline scripts,
  HttpOnly cookies. It is plain HTTP and binds to this machine only; use a TLS reverse proxy or VPN before exposing it (`--host`).
- Software fault response: if the monitoring loop fails 20 cycles in a row it issues a relay-shutdown command and the
  dashboard shows a MONITORING STOPPED banner. `/healthz` returns 503 in that state. Physical relay operation and its
  power-loss default state have not been tested; verify them on a supervised low-voltage bench before making a fail-safe claim.
- Storage retention: the dashboard runner keeps the newest 14 days of sensor readings (`READINGS_KEEP_ROWS`), about 12 MB/day at 2 Hz.
- Live camera, personnel recognition/tracking, zones, user management, notification channels and HTTP sensor input: see below and docs/.

## Camera, people, notifications, users, HTTP sensors (docs/PERSONNEL.md, docs/DEPLOY.md)
```powershell
python scripts/download_face_models.py ; python scripts/check_face_setup.py
scripts\run_dashboard.bat --sim --scenario gas_leak --camera 0 --people
```
- **Camera page:** live MJPEG video with `Name (Designation)` labels, tracked identities, zones, people table. Max 3 viewers.
- **Personnel (admin):** add people with consent, enrol photos or capture from the camera, disable/delete (templates + events removed).
- **Safety link:** a person inside a hazard zone while the system is WARNING/CRITICAL triggers an `[EMERGENCY]` notification naming them.
- **Accuracy:** `python -m ml.eval_faces` measures YOUR false-accept rate and suggests the threshold. Face accuracy has NOT been measured on real faces yet.
- **Notifications:** Telegram / e-mail / webhook via `data/notifications.json` + environment variables; queued in a background thread.
- **Users page (admin):** create, disable, set passwords; cannot disable yourself or the last admin.
- **HTTP sensors:** `--http-ingest` + `arduino/esp32_http/esp32_http.ino` (ESP32 sketch untested on a board).

## Datasets and training scripts
Every dataset, its download link, the folder it goes in, the training command and the result so far are in
**docs/DATASETS_AND_TRAINING.md**. Quick start: `python scripts/get_datasets.py --all` (UCI gas, NAB, face models).
D-Fire is trained on Colab with `notebooks/Fire_CNN_Training_Colab.ipynb`. The trained fire CNN is already in `models/`.

To use and validate the CNN on your laptop camera: **docs/WEBCAM_VALIDATION.md** (`python scripts/webcam_validate.py live`).

## Deep-learning extras
```powershell
pip install -r requirements-ml.txt            # Runtime TensorFlow 2.18; CNN training metadata records TensorFlow 2.21
python -m ml.train_temp_lstm --download-nab   # then run the NAB command printed in ml/train_temp_lstm.py
python -m ml.prepare_fire_dataset --root data/raw/fire --out data/fire_manifest.csv
python -m ml.train_cnn --manifest data/fire_manifest.csv      # use Colab: docs/COLAB.md
python -m src.main --sim --scenario fire --camera demo_video.mp4
```
**Temperature LSTM, NAB result (real data, test split = 2 labeled anomalies):** the LSTM detected 2/2 windows with
about 0.34 false-alarm segments/day, but a trivial "previous value" baseline gave identical event counts and a
rolling z-score baseline had slightly fewer false alarms. The LSTM shows no measurable advantage here, so the
runtime temperature path is the rule-based score (level + rate of rise) plus the ML hazard-state model. The LSTM
is only enabled at runtime if retrained on this device's own logs, and only if it beats the baselines there.

## Architecture
`reader (serial | simulator)` -> `DecisionEngine` (rule scores + optional AI `Signals`, fusion, hysteresis, forecaster)
-> `ResponseManager` (buzzer, notify, relay) -> `Storage`. Wired together in `src/pipeline.py`.

## Design rules
- Sensor faults are explicit faults, never silent zeros. Hardware mode never falls back to simulation.
- Rule-based scores are a safety floor: AI can raise a score, never lower it.
- Any source can escalate, none can silence another. Escalation is immediate; de-escalation needs calm cycles.
- Intended relay wiring convention: energized = machine powered; software shutdown is LATCHED until a manual reset while SAFE. Verify the actual relay polarity and power-loss state on the physical board before connecting any load.
- All thresholds live in `src/config.py` and must be recalibrated on real hardware.


## Release review notes (4 November 2026)
- This ZIP is a software/simulator delivery, not a physically validated or certified industrial safety product.
- The trained fire CNN is below its F1/recall targets. CNN-only evidence is capped at WARNING; a CNN-driven CRITICAL override requires independent gas, temperature or flame evidence.
- The included C1-C5 report is dated 27 October 2026 and has three failed metric rows. Re-run it after the latest decision-engine guard change.
- Final verification run: 150 tests passed (Python 3.12, TensorFlow 2.18, scikit-learn 1.5), including regression tests for the CNN guard (it cannot be bypassed by editing the CRITICAL threshold or the corroboration setting). Re-run under Python 3.10 before submission.
- See `../../2_Documentation/FINAL_SUBMISSION_GATE.md` (when using the ZIP layout) for all remaining evidence and university sign-off requirements.
