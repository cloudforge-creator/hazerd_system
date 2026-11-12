"""Central configuration. Change thresholds here, never inline (WBS 6.2.3, 8.5.1)."""

SERIAL_BAUD = 9600
CYCLE_SECONDS = 0.5          # one reading cycle
STALE_AFTER_SECONDS = 3.0    # no fresh data for this long -> link fault

# Physical sanity limits. A value outside its range is a SENSOR FAULT,
# never silently treated as a normal reading (WBS 4.3.2).
VALID_RANGES = {
    "gas_raw": (0, 1023),          # 10-bit ADC on Arduino Uno
    "temperature": (-10.0, 80.0),  # DHT22 rated -40..80 C
    "humidity": (0.0, 100.0),
    "flame": (0, 1),
}

# ---------------------------------------------------------------------------
# M3 decision logic. Rule-based scoring is the SAFETY FLOOR: AI scores may raise
# a score but never lower it. Calibrate these on real hardware later.
# ---------------------------------------------------------------------------
GAS_SAFE_MAX = 300            # raw ADC at/below this -> gas score 0
GAS_CRITICAL = 600            # raw ADC at/above this -> gas score 1
GAS_CRITICAL_CONSEC = 3      # require 3 consecutive critical gas cycles (1.5 s at 2 Hz)
TEMP_SAFE_MAX = 40.0          # deg C
TEMP_CRITICAL = 70.0
TEMP_RATE_CRITICAL = 2.0      # deg C per second rise -> rate score 1
RATE_WINDOW_S = 10.0

WARNING_THRESHOLD = 0.40
CRITICAL_THRESHOLD = 0.70
FIRE_PROB_OVERRIDE = 0.80     # CNN probability; CRITICAL requires independent sensor corroboration
CNN_SOLO_MAX_SCORE = 0.69     # CNN-only evidence may raise WARNING, never CRITICAL (also kept below CRITICAL_THRESHOLD)
CNN_CORROBORATION_MIN = 0.30  # independent sensor score needed before the CNN may cause CRITICAL; NOT editable live
CORROBORATION_MIN = 0.30      # a signal above this counts as "active"
CORROBORATION_BONUS = 0.15    # added per extra active signal
CLEAR_CYCLES = 6              # consecutive lower cycles before de-escalating
FAULT_CYCLES_LIMIT = 6        # consecutive faulty cycles before raising a fault alert

# Early-warning forecaster (baseline; replaced/compared with an ML model in Phase 5)
FORECAST_WINDOW_S = 10.0
FORECAST_HORIZON_S = 10.0
FORECAST_MIN_SLOPE = 0.01     # score units per second

# M4 response
NOTIFY_COOLDOWN_S = 30.0
DB_PATH = "data/hazard.db"
LOG_PATH = "data/logs/sensor_log.csv"
READINGS_KEEP_ROWS = 2_419_200   # 14 days at 2 readings/s; older sensor_readings rows are pruned
PRUNE_EVERY_S = 3600

# ---------------------------------------------------------------------------
# ML integration (Phase M2)
# ---------------------------------------------------------------------------
MODEL_DIR = "models"
# Runtime model ensemble: CNN / Random Forest / temperature LSTM.
# Weights are normalized and applied only to available, valid model outputs.
MODEL_WEIGHTS = {"cnn": 0.50, "rf": 0.25, "lstm": 0.25}
ML_LEVEL_SCORES = {0: 0.0, 1: 0.55, 2: 0.85}   # SAFE / WARNING / CRITICAL -> risk score
ML_SOLO_MAX_SCORE = 0.69                     # ML ensemble alone cannot command a CRITICAL shutdown
ML_EARLY_CONSEC = 2                             # consecutive steps above threshold -> alarm
TREND_EARLY_CONSEC = 2                          # consecutive predictive-trend cycles -> early warning

# ---------------------------------------------------------------------------
# Personnel recognition / camera (all editable live from the dashboard unless noted)
# ---------------------------------------------------------------------------
FACE_DETECT_MODEL = "models/face/face_detection_yunet_2023mar.onnx"      # not editable live
FACE_RECOG_MODEL = "models/face/face_recognition_sface_2021dec.onnx"     # not editable live
FACE_DETECT_SCORE = 0.85
FACE_MATCH_THRESHOLD = 0.363      # SFace cosine similarity (OpenCV's recommended value); MEASURE on your own faces
FACE_MATCH_MARGIN = 0.05          # best match must beat the runner-up by this much (stops look-alike mix-ups)
FACE_MIN_SIZE_PX = 60             # faces narrower than this are tracked but not identified
PERSON_TRACK_MAX_MISSED = 10      # frames a track survives without a detection
PERSON_MIN_VOTES = 3              # consistent matches needed to confirm an identity
PERSON_UNKNOWN_AFTER = 4          # identified-attempts before a face is called Unknown
PERSON_PROCESS_FPS = 5.0
EMERGENCY_COOLDOWN_S = 15.0
ZONES_PATH = "data/zones.json"
NOTIFY_CONFIG_PATH = "data/notifications.json"
