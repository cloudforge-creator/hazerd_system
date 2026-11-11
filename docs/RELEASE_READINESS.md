# Software release readiness

Last reviewed: 2026-11-11

## Scope and release status

This checklist covers repository software only. It does not certify a physical installation, sensor calibration, relay behavior, or industrial safety performance.

The reliability changes are in PR #1. The follow-up branch release/software-readiness adds a conservative guard against ML-state-only CRITICAL shutdowns, event-level validation for early-warning threshold selection, and a firmware compile check in CI.

**Status: not yet release-ready.** The follow-up changes require passing CI and a fresh full C1-C5 evaluation before the software-only release candidate can be considered.

## Changes made in this release-readiness branch

- The state classifier uses the same sensor-derived features as the physical rules. It is therefore not independent evidence. Its score is capped below CRITICAL unless a physical gas, temperature, or flame rule independently reaches CRITICAL.
- The early-warning model threshold is selected using run/event-level validation metrics (detection rate and false-alarm run rate), rather than window-level F1 alone.
- The validation report records whether both C2 acceptance targets were achieved. If no candidate threshold satisfies both, the unmet target remains visible.
- CI compiles the Arduino Uno firmware source using the AVR core and DHT dependencies. A successful compile proves source/toolchain compatibility only; it does not prove electrical behavior or physical relay actuation.

## Existing evidence and outstanding gaps

The following are the last recorded results before the changes above. They must not be presented as post-fix results until the evaluation is rerun on the new branch.

| Area | Last recorded result | Remaining acceptance work |
|---|---:|---|
| C1 macro-F1, rule + ML | 0.9082 | Re-run after decision guard; inspect WARNING class |
| C2 early detection, rule + ML | 0.9764 | Re-run with event-level validation threshold |
| C2 false-alarm runs, rule + ML | 0.1795 | Must reach <= 0.10 without losing >= 0.90 detection |
| C4 gas-leak clean false shutdowns | 7/11 | Diagnose and reduce; do not tune away genuine hazards |
| CNN fire precision / recall / F1 | 0.8704 / 0.8919 / 0.8810 | New candidate must reach recall >= 0.92 and F1 >= 0.90 on untouched test data |
| Gas UCI cross-batch macro-F1 | 0.4711 (batches 6-10) | Improve drift robustness; UCI 128-feature array is not the runtime single-channel MQ-2 |
| Temperature LSTM point F1 / recall | 0.0219 / 0.0115 | Research result only; no demonstrated advantage over persistence / rolling-z baselines |
| C3 p99 latency | 31.929 ms | Re-measure after changes on the release machine |
| C5 projected Python heap growth | 1.03 MB/day | Re-measure after changes on the release machine |

## Model-specific release gates

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
- [ ] This branch's Python test job passes.
- [ ] This branch's Arduino Uno compile job passes.
- [ ] Full C1-C5 evaluation rerun after the decision-engine change.
- [ ] C2 detection >= 0.90 and false-alarm run rate <= 0.10 on held-out runs.
- [ ] C4 gas-leak false shutdown behavior reviewed and accepted on a sufficiently representative clean set.
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
