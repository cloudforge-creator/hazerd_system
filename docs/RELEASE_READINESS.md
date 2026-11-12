# Software release readiness

Last reviewed: 2026-11-12

## Scope and release status

This checklist covers repository software only. It does not certify a physical installation, sensor calibration, relay behavior, or industrial safety performance.

The reliability changes are in PR #1. The current `fix/software-reliability` branch includes a conservative guard against ML-state-only CRITICAL shutdowns, event-level validation for early-warning threshold selection, and an Arduino Uno firmware compile check in CI.

**Status: not yet release-ready.** Local verification on 2026-11-12 reports 170 Python tests passed and a complete local C1-C5 evaluation. GitHub Actions run `43183400240` on commit `c2bb0f9` passed Python tests, Arduino Uno firmware source compilation, deterministic ML baseline training, PR smoke evaluation, and artifact upload. Full C1-C5 evaluation was intentionally skipped for the pull-request event; it runs on pushes to `main` or manual `workflow_dispatch` with a 90-minute timeout. C2 still fails its false-alarm acceptance target, so release acceptance remains blocked.

## Changes made in this release-readiness branch

- Runtime model ensemble weights are explicitly set to CNN 0.50, Random Forest 0.25, and temperature LSTM 0.25. Available model outputs are renormalized when a model is disabled or unavailable. The state classifier used for the RF slot is trained as a Random Forest.
- The weighted AI ensemble remains advisory: it can raise WARNING, but cannot cause CRITICAL shutdown without independent physical sensor evidence.

- The state classifier uses the same sensor-derived features as the physical rules. It is therefore not independent evidence. Its score is capped below CRITICAL unless a physical gas, temperature, or flame rule independently reaches CRITICAL.
- The early-warning model threshold is selected using run/event-level validation metrics (detection rate and false-alarm run rate), rather than window-level F1 alone.
- The validation report records whether both C2 acceptance targets were achieved. If no candidate threshold satisfies both, the unmet target remains visible.
- CI compiles the Arduino Uno firmware source using the AVR core and DHT dependencies. A successful compile proves source/toolchain compatibility only; it does not prove electrical behavior or physical relay actuation.

## Existing evidence and outstanding gaps

The following are the results from the current verification cycle, generated 2026-11-12. C1/C2/C4 are synthetic scenario results, not field validation. The evaluation is software-only and is not evidence of real-world hazard accuracy.

| Area | Last recorded result | Remaining acceptance work |
|---|---:|---|
| C1 macro-F1, rule + ML | 0.9082 | Passes >= 0.90; WARNING-class F1 remains 0.7534 |
| C2 early detection, rule + ML | 0.9764 | Passes >= 0.90 on synthetic held-out runs |
| C2 false-alarm runs, rule + ML | 0.1795 | FAILS <= 0.10; 49/273 clean runs false-alarmed |
| C4 gas-leak clean false shutdowns | 7/11 | Subset issue persists; investigate gas cross-sensitivity and near-critical plateaus |
| CNN fire precision / recall / F1 | 0.8704 / 0.8919 / 0.8810 | New candidate must reach recall >= 0.92 and F1 >= 0.90 on untouched test data |
| Gas UCI cross-batch macro-F1 | 0.4711 (batches 6-10) | Improve drift robustness; UCI 128-feature array is not the runtime single-channel MQ-2 |
| Temperature LSTM point F1 / recall | 0.0219 / 0.0115 | Research result only; no demonstrated advantage over persistence / rolling-z baselines |
| C3 p99 full-cycle latency (rule + ML) | 31.929 ms | Passes synthetic CI budget; measure on deployment hardware |
| C5 projected Python heap growth | 1.03 MB/day | Passes synthetic CI budget; tracemalloc excludes SQLite C allocations |

## Model-specific release gates

### Weighted CNN / RF / LSTM ensemble
- CNN weight: 0.50; Random Forest weight: 0.25; temperature LSTM weight: 0.25.
- Weights apply to available model outputs and are renormalized if a model is disabled or unavailable.
- These outputs are not independent physical safety channels; the ensemble cannot independently trigger CRITICAL shutdown.
- The temperature LSTM remains disabled unless trained on this device at the configured sampling interval.

### Early-warning model
- Train/validation/test runs remain separated by scenario run.
- Select threshold only on validation runs.
- Report event-level detection, lead time, and false alarms on held-out test runs.
- Do not claim target compliance unless both C2 metrics pass on the test split.

### Gas model
- Preserve batch-wise evaluation; random-split accuracy is not the deployment estimate.
- The UCI dataset's 16-sensor array has 128 features. It does not directly validate the project's MQ-2 ADC runtime path.
- Deployment use requires a model trained and validated on the actual runtime sensor inputs and representative device batches.

### Fire CNN
- Keep the CNN-only CRITICAL guard until independent in-domain camera validation supports a reviewed change.
- Evaluate on an untouched image test split and representative camera negatives/positives.
- Do not claim the D-Fire score is camera-specific field performance.

### Temperature LSTM
- Keep it out of the active runtime path unless it beats simple baselines on device data at the intended sampling interval.
- NAB uses five-minute samples and has very few labeled test anomaly windows; it does not validate a 0.5-second sensor stream.

## Software verification checklist

- [ ] PR #1 reviewed and merged.
- [x] Python test suite passes locally (170 passed) and in GitHub Actions run `43183400240`.
- [x] Arduino Uno firmware source compilation passes in GitHub Actions run `43183400240` (compile-only; no board upload or electrical validation).
- [x] GitHub Actions pull-request smoke evaluation and artifact upload pass (run `43183400240`).
- [ ] Full C1-C5 evaluation in hosted GitHub Actions; intentionally not run on pull requests, available through manual workflow dispatch or push to `main`.
- [x] Full C1-C5 evaluation rerun after the decision-engine change; current verified result is complete and software-only.
- [ ] C2 detection >= 0.90 AND false-alarm run rate <= 0.10 on held-out runs (detection passes at 0.9764; false-alarm rate remains 0.1795, so C2 is parked and acceptance is not achieved).
- [ ] C4 gas-leak false shutdown behavior reviewed and accepted on a sufficiently representative clean set. Current software evaluation records false-shutdown commands in 7 of 11 clean runs in the gas-leak scenario; physical relay actuation remains unvalidated.
- [ ] CNN candidate meets its test targets, or remains explicitly below target and guarded.
- [ ] Gas model drift limitation remains explicit; runtime MQ-2 model validated separately.
- [ ] C3/C5 measured again on the documented release machine.
- [ ] API, deployment, security, model, and known-limitations documentation reviewed.
- [ ] Release notes and version tag prepared.

## Hardware-dependent work intentionally excluded

- Board upload and serial protocol loopback.
- Sensor calibration against known gas/temperature references.
- Physical relay, buzzer, and power-loss tests.
- Camera validation using the final deployment camera and environment.
- Field trials and safety certification.

These require the actual hardware and supervised test conditions. Software CI cannot substitute for those tests.
