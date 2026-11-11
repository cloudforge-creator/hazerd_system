# C1-C5 Evaluation Report

Generated 2026-11-11 11:51:11 | seed 2026 | models: rule + ML | mode: full

> **Read first.** C1/C2/C4 use synthetic held-out scenarios, not field data. C4 counts relay commands issued, not physical actuation. C3/C5 timings are specific to the machine that ran this. See limitations below.

## Summary against targets

| Measure | Result | Target | Verdict |
|---|---|---|---|
| C1 macro-F1 (rule_only) | 0.901 | >= 0.9 | **PASS** |
| C1 CRITICAL recall (rule_only) | 0.9906 | >= 0.95 | **PASS** |
| C1 false CRITICAL, clean runs (rule_only) | 0.0256 | <= 0.05 | **PASS** |
| C2 early-warning detection (rule_only) | 0.8268 | >= 0.9 | **FAIL** |
| C2 false-alarm runs (rule_only) | 0.1099 | <= 0.1 | **FAIL** |
| C4 shutdown within 5 s (rule_only) | 1.0 | >= 0.95 | **PASS** |
| C4 false shutdown, clean runs (rule_only) | 0.0256 | <= 0.05 | **PASS** |
| C1 macro-F1 (rule_plus_ml) | 0.9082 | >= 0.9 | **PASS** |
| C1 CRITICAL recall (rule_plus_ml) | 0.9994 | >= 0.95 | **PASS** |
| C1 false CRITICAL, clean runs (rule_plus_ml) | 0.0256 | <= 0.05 | **PASS** |
| C2 early-warning detection (rule_plus_ml) | 0.9764 | >= 0.9 | **PASS** |
| C2 false-alarm runs (rule_plus_ml) | 0.1795 | <= 0.1 | **FAIL** |
| C4 shutdown within 5 s (rule_plus_ml) | 1.0 | >= 0.95 | **PASS** |
| C4 false shutdown, clean runs (rule_plus_ml) | 0.0256 | <= 0.05 | **PASS** |
| C3 full-cycle p99 ms (rule_only) | 18.721 | <= 500.0 | **PASS** |
| C3 full-cycle p99 ms (rule_plus_ml) | 31.929 | <= 500.0 | **PASS** |
| C5 soak heap growth, MB/day projected | 1.03 | <= 10.0 | **PASS** |
| C5 soak errors | 0 | == 0 | **PASS** |
| C5 dashboard p95 ms under load | 75.894 | <= 250.0 | **PASS** |
| C5 dashboard errors under load | 0 | == 0 | **PASS** |
| C5 stress exceptions | 0 | == 0 | **PASS** |

Targets are the ones set in the project roadmap; a FAIL is reported as measured, not tuned away.

## Full results

```json
{
  "rule_only": {
    "C1": {
      "accuracy": 0.9787,
      "macro_f1": 0.901,
      "per_class": {
        "SAFE": {
          "precision": 0.9985,
          "recall": 0.9853,
          "f1": 0.9919
        },
        "WARNING": {
          "precision": 0.6917,
          "recall": 0.7851,
          "f1": 0.7354
        },
        "CRITICAL": {
          "precision": 0.9614,
          "recall": 0.9906,
          "f1": 0.9758
        }
      },
      "confusion_matrix(rows=true,cols=pred; SAFE,WARNING,CRITICAL)": [
        [
          93074,
          1387,
          1
        ],
        [
          138,
          3551,
          834
        ],
        [
          2,
          196,
          20817
        ]
      ],
      "critical_event_detection_rate": 1.0,
      "median_detection_delay_s(negative=earlier than truth)": 0.0,
      "false_critical_rate_on_clean_runs": 0.0256,
      "runs": {
        "with_critical": 127,
        "clean": 273
      },
      "step_accuracy_by_scenario": {
        "normal": 1.0,
        "gas_leak": 0.9353,
        "spray": 1.0,
        "humid": 1.0,
        "overheat": 0.9823,
        "fire": 0.9712,
        "slow_leak": 0.9419
      }
    },
    "C2": {
      "runs_with_critical": 127,
      "detected_before_critical": 105,
      "detection_rate": 0.8268,
      "median_lead_s": 8.0,
      "p10_lead_s": 2.7,
      "clean_runs": 273,
      "false_alarm_runs": 30,
      "false_alarm_run_rate": 0.1099,
      "by_scenario": {
        "normal": {
          "critical_runs": 0,
          "detected": 0,
          "clean_runs": 89,
          "false_alarms": 2
        },
        "gas_leak": {
          "critical_runs": 54,
          "detected": 54,
          "clean_runs": 11,
          "false_alarms": 11
        },
        "spray": {
          "critical_runs": 0,
          "detected": 0,
          "clean_runs": 55,
          "false_alarms": 2
        },
        "humid": {
          "critical_runs": 0,
          "detected": 0,
          "clean_runs": 53,
          "false_alarms": 1
        },
        "overheat": {
          "critical_runs": 16,
          "detected": 16,
          "clean_runs": 34,
          "false_alarms": 11
        },
        "fire": {
          "critical_runs": 57,
          "detected": 35,
          "clean_runs": 0,
          "false_alarms": 0
        },
        "slow_leak": {
          "critical_runs": 0,
          "detected": 0,
          "clean_runs": 31,
          "false_alarms": 3
        }
      }
    },
    "C4": {
      "gas_leak": {
        "hazard_runs": 54,
        "shutdown_issued_rate": 1.0,
        "shutdown_within_5s_of_true_critical": 1.0,
        "median_shutdown_delay_s(negative=earlier)": -0.5,
        "alarm_rate": 1.0,
        "notification_rate": 1.0
      },
      "overheat": {
        "hazard_runs": 16,
        "shutdown_issued_rate": 1.0,
        "shutdown_within_5s_of_true_critical": 1.0,
        "median_shutdown_delay_s(negative=earlier)": 0.0,
        "alarm_rate": 1.0,
        "notification_rate": 1.0
      },
      "fire": {
        "hazard_runs": 57,
        "shutdown_issued_rate": 1.0,
        "shutdown_within_5s_of_true_critical": 1.0,
        "median_shutdown_delay_s(negative=earlier)": 0.0,
        "alarm_rate": 1.0,
        "notification_rate": 1.0
      },
      "ALL_HAZARDS": {
        "hazard_runs": 127,
        "shutdown_within_5s_of_true_critical": 1.0,
        "false_shutdown_rate_on_clean_runs": 0.0256,
        "clean_runs": 273,
        "false_shutdowns_by_scenario(count/clean_runs)": {
          "normal": "0/89",
          "gas_leak": "7/11",
          "spray": "0/55",
          "humid": "0/53",
          "overheat": "0/34",
          "slow_leak": "0/31"
        }
      },
      "note": "Counts relay commands ISSUED; physical actuation is unconfirmed until the hardware loopback test."
    }
  },
  "rule_plus_ml": {
    "C1": {
      "accuracy": 0.9803,
      "macro_f1": 0.9082,
      "per_class": {
        "SAFE": {
          "precision": 0.9997,
          "recall": 0.9848,
          "f1": 0.9922
        },
        "WARNING": {
          "precision": 0.7142,
          "recall": 0.7973,
          "f1": 0.7534
        },
        "CRITICAL": {
          "precision": 0.9591,
          "recall": 0.9994,
          "f1": 0.9788
        }
      },
      "confusion_matrix(rows=true,cols=pred; SAFE,WARNING,CRITICAL)": [
        [
          93028,
          1433,
          1
        ],
        [
          22,
          3606,
          895
        ],
        [
          2,
          10,
          21003
        ]
      ],
      "critical_event_detection_rate": 1.0,
      "median_detection_delay_s(negative=earlier than truth)": 0.0,
      "false_critical_rate_on_clean_runs": 0.0256,
      "runs": {
        "with_critical": 127,
        "clean": 273
      },
      "step_accuracy_by_scenario": {
        "normal": 1.0,
        "gas_leak": 0.9456,
        "spray": 1.0,
        "humid": 1.0,
        "overheat": 0.981,
        "fire": 0.9712,
        "slow_leak": 0.9434
      }
    },
    "C2": {
      "runs_with_critical": 127,
      "detected_before_critical": 124,
      "detection_rate": 0.9764,
      "median_lead_s": 11.5,
      "p10_lead_s": 5.65,
      "clean_runs": 273,
      "false_alarm_runs": 49,
      "false_alarm_run_rate": 0.1795,
      "by_scenario": {
        "normal": {
          "critical_runs": 0,
          "detected": 0,
          "clean_runs": 89,
          "false_alarms": 2
        },
        "gas_leak": {
          "critical_runs": 54,
          "detected": 54,
          "clean_runs": 11,
          "false_alarms": 11
        },
        "spray": {
          "critical_runs": 0,
          "detected": 0,
          "clean_runs": 55,
          "false_alarms": 6
        },
        "humid": {
          "critical_runs": 0,
          "detected": 0,
          "clean_runs": 53,
          "false_alarms": 1
        },
        "overheat": {
          "critical_runs": 16,
          "detected": 16,
          "clean_runs": 34,
          "false_alarms": 21
        },
        "fire": {
          "critical_runs": 57,
          "detected": 54,
          "clean_runs": 0,
          "false_alarms": 0
        },
        "slow_leak": {
          "critical_runs": 0,
          "detected": 0,
          "clean_runs": 31,
          "false_alarms": 8
        }
      }
    },
    "C4": {
      "gas_leak": {
        "hazard_runs": 54,
        "shutdown_issued_rate": 1.0,
        "shutdown_within_5s_of_true_critical": 1.0,
        "median_shutdown_delay_s(negative=earlier)": -1.0,
        "alarm_rate": 1.0,
        "notification_rate": 1.0
      },
      "overheat": {
        "hazard_runs": 16,
        "shutdown_issued_rate": 1.0,
        "shutdown_within_5s_of_true_critical": 1.0,
        "median_shutdown_delay_s(negative=earlier)": -0.5,
        "alarm_rate": 1.0,
        "notification_rate": 1.0
      },
      "fire": {
        "hazard_runs": 57,
        "shutdown_issued_rate": 1.0,
        "shutdown_within_5s_of_true_critical": 1.0,
        "median_shutdown_delay_s(negative=earlier)": 0.0,
        "alarm_rate": 1.0,
        "notification_rate": 1.0
      },
      "ALL_HAZARDS": {
        "hazard_runs": 127,
        "shutdown_within_5s_of_true_critical": 1.0,
        "false_shutdown_rate_on_clean_runs": 0.0256,
        "clean_runs": 273,
        "false_shutdowns_by_scenario(count/clean_runs)": {
          "normal": "0/89",
          "gas_leak": "7/11",
          "spray": "0/55",
          "humid": "0/53",
          "overheat": "0/34",
          "slow_leak": "0/31"
        }
      },
      "note": "Counts relay commands ISSUED; physical actuation is unconfirmed until the hardware loopback test."
    }
  },
  "C3": {
    "rule_only": {
      "full_cycle(read+decide+respond+DB)": {
        "n": 6000,
        "mean_ms": 7.437,
        "p50_ms": 6.853,
        "p95_ms": 10.199,
        "p99_ms": 18.721,
        "max_ms": 38.095
      },
      "decision_only": {
        "n": 6000,
        "mean_ms": 0.179,
        "p50_ms": 0.166,
        "p95_ms": 0.299,
        "p99_ms": 0.359,
        "max_ms": 0.649
      },
      "ai_inference_only": {
        "n": 6000,
        "mean_ms": 0.0,
        "p50_ms": 0.0,
        "p95_ms": 0.0,
        "p99_ms": 0.0,
        "max_ms": 0.001
      }
    },
    "rule_plus_ml": {
      "full_cycle(read+decide+respond+DB)": {
        "n": 6000,
        "mean_ms": 17.934,
        "p50_ms": 17.768,
        "p95_ms": 23.191,
        "p99_ms": 31.929,
        "max_ms": 176.076
      },
      "decision_only": {
        "n": 6000,
        "mean_ms": 0.461,
        "p50_ms": 0.364,
        "p95_ms": 0.799,
        "p99_ms": 0.967,
        "max_ms": 10.895
      },
      "ai_inference_only": {
        "n": 6000,
        "mean_ms": 7.594,
        "p50_ms": 6.366,
        "p95_ms": 12.374,
        "p99_ms": 14.166,
        "max_ms": 134.059
      }
    },
    "cycle_budget_ms": 500.0
  },
  "C5": {
    "soak": {
      "simulated_hours": 1.0,
      "cycles": 7200,
      "errors": 0,
      "wall_seconds": 246.35,
      "cycles_per_second": 29.2,
      "speedup_vs_real_time": 14.6,
      "safe_cycle_p95_ms_first_10pct_after_warmup": 45.202,
      "safe_cycle_p95_ms_last_10pct": 42.12,
      "python_heap_kb_by_decile": [
        88.6,
        85.6,
        97.3,
        98.9,
        94.5,
        111.4,
        109.0,
        121.6,
        117.6,
        116.5
      ],
      "heap_growth_kb_second_half": 22.0,
      "projected_heap_growth_mb_per_day_at_2hz": 1.03,
      "process_rss_mb_start_end": [
        154.1,
        154.6
      ],
      "db_rows": 7200,
      "db_bytes_per_cycle": 71.7,
      "projected_db_mb_per_day_at_2hz": 12.4,
      "note": "heap = Python allocations only (tracemalloc); sqlite's C memory is not included."
    },
    "sensor_load": {
      "1": {
        "p95_ms_for_whole_cycle": 0.608,
        "max_ms": 0.61,
        "within_500ms_budget": true
      },
      "10": {
        "p95_ms_for_whole_cycle": 5.374,
        "max_ms": 5.433,
        "within_500ms_budget": true
      },
      "50": {
        "p95_ms_for_whole_cycle": 11.824,
        "max_ms": 11.929,
        "within_500ms_budget": true
      },
      "100": {
        "p95_ms_for_whole_cycle": 21.46,
        "max_ms": 21.671,
        "within_500ms_budget": true
      },
      "200": {
        "p95_ms_for_whole_cycle": 40.965,
        "max_ms": 41.923,
        "within_500ms_budget": true
      },
      "max_nodes_tested_within_budget": 200,
      "note": "Rule-based pipeline, in-memory DB, one Python process: a software scaling indicator, not a hardware limit."
    },
    "http_load": {
      "concurrent_clients": 10,
      "requests": 1000,
      "errors": 0,
      "requests_per_second": 228.4,
      "latency_mean_ms": 38.354,
      "latency_p50_ms": 34.763,
      "latency_p95_ms": 75.894,
      "latency_p99_ms": 79.744,
      "latency_max_ms": 81.275,
      "monitoring_cycles_completed_during_load": 94,
      "monitoring_cycles_expected_at_100hz": 437,
      "monitoring_crashed": false
    },
    "stress": {
      "fault_storm_normal_operation": {
        "cycles": 7200,
        "exceptions": 0,
        "degraded_cycle_pct": 9.47,
        "false_critical_cycles": 0
      },
      "forced_10_cycle_dropout": {
        "warning_raised_after_cycles": 5,
        "fault_limit": 6,
        "cleared_after_cycles": 5
      },
      "shutdown_under_faults_vs_clean": {
        "gas_leak": {
          "runs": 25,
          "runs_that_truly_reach_critical": 22,
          "shutdown_rate_no_faults": 0.92,
          "shutdown_rate_10pct_faults": 0.92
        },
        "overheat": {
          "runs": 25,
          "runs_that_truly_reach_critical": 8,
          "shutdown_rate_no_faults": 0.36,
          "shutdown_rate_10pct_faults": 0.36
        },
        "fire": {
          "runs": 25,
          "runs_that_truly_reach_critical": 25,
          "shutdown_rate_no_faults": 1.0,
          "shutdown_rate_10pct_faults": 1.0
        }
      }
    }
  }
}
```

## What this does and does not prove

- Synthetic held-out scenarios (seed different from training) measure software behaviour and robustness to the modelled effects (humidity/temperature cross-sensitivity, benign spikes, noise, dropouts). They do **not** measure field accuracy.
- **Circularity (C1/C4):** ground-truth labels use the same threshold values as the rule engine, applied to noise-free latent signals. High rule-based scores therefore show robustness to noise and lag, NOT that the thresholds are right for real hazards. Calibrate on hardware.
- **C1 is carried by the SAFE and CRITICAL classes.** WARNING-class F1 is much lower (see per_class); the macro-F1 pass is marginal.
- **C2 is a trade-off, not a free win.** Adding the ML early-alarm raises detection but also false-alarm runs (see C2 by_scenario: mostly overheat and slow-leak runs that warm up but never reach CRITICAL). Pick the operating point deliberately.
- False shutdowns on clean runs come from near-critical plateaus (see `false_shutdowns_by_scenario`), not from normal operation.
- The early-warning model is limited by fires: the flame sensor can trip within about 5-10 s of the first temperature rise, leaving little warning for any predictor.
- C4 counts commands issued; the hardware loopback test (relay + load) must confirm physical shutdown.
- The soak leak criterion measures sustained heap growth in the second half of the run, projected per day. An earlier draft used percent growth from the 10% mark, which was ill-defined for a ~13 KB start-up heap, so it was replaced before the full run.
- C5 projected database growth shows every cycle is stored; the dashboard runner now prunes sensor readings to the newest 14 days (config READINGS_KEEP_ROWS).
- Dashboard load: the monitoring loop was run at an artificial 100 Hz; it completed fewer cycles while 10 clients hammered the dashboard. At the real 2 Hz it needs about 1-4 ms per cycle, so headroom is large, but this contention is real.
- Re-run on real logged data after hardware arrives and quote those numbers.