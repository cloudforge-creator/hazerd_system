# Using and validating the fire CNN on your laptop camera

The CNN is already integrated (`fire CNN loaded` when you start the system). These steps check that it behaves on **your** camera.
Run everything from the project folder with the virtual environment active. `--source 0` is the laptop camera (try 1 if needed).

## 1. See it working (2 minutes)
```powershell
python scripts/webcam_validate.py live
```
A window shows the camera and the raw CNN fire score. These bands describe the CNN score only: LOW below 0.40, WARNING from 0.40, ELEVATED from 0.70 and HIGH from 0.80. They are **not** the final engine level. The runtime engine caps CNN-only evidence below CRITICAL; a CNN-driven CRITICAL decision requires independent gas, temperature or flame-sensor corroboration. Press q to quit.
Use it with the whole system: `python -m src.main --sim --scenario normal --camera 0 --dashboard`, then open http://127.0.0.1:5000 and the Camera page.

To time capture plus the CNN on your machine (budget 500 ms per cycle): `python scripts/benchmark_cnn.py`.

## 2. Record normal footage (30 to 60 minutes)
```powershell
python scripts/webcam_validate.py record --label normal --minutes 30
```
Use the room as you normally would: move around, lamps on and off, daylight changes, a screen, orange or red objects, steam from a cup. **Do not show fire.** At least 24 minutes are needed for a verdict (about 3,600 frames per 30 minutes, about 180 MB).

## 3. Record fire footage (3 to 5 minutes)
Play a fire video on a phone or a second screen and hold it in front of the camera:
```powershell
python scripts/webcam_validate.py record --label fire --minutes 4
```
This is a screen recapture, so it shows that the model reacts, not how it does on real flame. A supervised candle flame, away from the sensors and anything flammable, is a better test.

## 4. Get the verdict
```powershell
python scripts/webcam_validate.py report
```
It prints, for the thresholds 0.40, 0.70 and 0.80: how many false-alarm episodes per hour on your normal footage, and how many fire frames are detected, with a verdict.
It also copies the 12 highest-scoring normal frames to `data/webcam/report/highest_scoring_normal_frames/`. **Look at them**: they show what your camera scene looks like to the model as fire.
Results are saved in `data/webcam/webcam_validation.json` (put the numbers in Chapter 6 of the report).

## How to read the verdict (suggested criteria, adjustable with `--max-warn-per-hour` and `--min-fire-recall`)
- **WARNING false alarms OK** = at most 2 alert episodes per hour at score 0.40 on normal footage.
- **No high-score false alarm** = no episode at 0.70 on normal footage. The runtime safety gate still requires independent physical-sensor corroboration before a CNN-driven CRITICAL decision.
- **Reacts to fire** = detects fire on at least half of the fire frames at 0.40.
The same footage can be used with `python -m ml.eval_cnn_operating_points --manifest ... --webcam-neg data/webcam/normal --webcam-pos data/webcam/fire`. The D-Fire test metrics are not a substitute for this in-domain camera test.

## If it fails
The runtime already requires a second physical signal (flame sensor, temperature or gas) before a CNN-driven CRITICAL override. Do not remove this safeguard based only on D-Fire metrics. Any future threshold change must be justified with validation data and retested.
