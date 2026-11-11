# Live camera, person recognition and tracking

What it does: the dashboard's **Camera** page shows the live video. Faces are detected, identified against the
people you enrolled, and tracked from frame to frame, so each person keeps one identity while they move. The video
shows `Name (Designation)` in green, `Identifying...` in orange and `Unknown person #n` in red, plus your zones.

What it is NOT: it only sees faces. A person facing away, wearing a mask, or in poor light is not identified or may
not be seen at all. It supports safety procedures; it does not replace lock-out, alarms or evacuation drills.

## Set-up (once)
1. `pip install -r requirements-ml.txt` (needs OpenCV).
2. `python scripts/download_face_models.py` (two small model files into `models/face/`; manual instructions are at the
   top of that script if your network blocks the download).
3. `python scripts/check_face_setup.py` while standing in front of the webcam. You want 1 face, a face width of at least
   60 px, and a similarity between two frames well above the threshold (0.363).
4. Start: `scripts\run_dashboard.bat --port COM3 --camera 0 --people` (or `--sim --camera 0 --people` to try it without sensors;
   `--camera video.mp4` or `--camera some_folder` also work).

## Enrol people (admin, Personnel page)
- Tick the consent box only if the person has been told and agreed. Face templates are biometric data.
- Add 3-5 photos per person (one person per photo, different angles/lighting), or press **Capture from camera** while they
  look at the camera. Only numbers (templates) are stored, never the photos; delete a person and their templates and
  events are removed.
- Check your own site's rules and the people's consent before using face recognition at work. This software cannot decide that for you.

## Zones (admin, Camera page)
Edit the JSON. Coordinates are 0-1 (click the video to read them). Position = centre of the FACE, so draw zones where heads appear.
- `hazard_when_alert: true`: anyone inside while the system is WARNING/CRITICAL is flagged and an **[EMERGENCY]** notification
  naming them (e.g. `Ali Khan (Operator), Unknown person #2`) is sent. This never changes the hazard level itself.
- `allowed_designations`: if not empty, anyone else (including unknown people) entering is logged as UNAUTHORIZED_ZONE.
See `data/zones.example.json`.

## Measure the accuracy before you trust it
Put photos in `data/faces/<Person name>/*.jpg` (3+ people, 5+ photos each, taken with your camera) and run
`python -m ml.eval_faces`. It prints the false-accept rate (a stranger named as someone) per threshold and suggests the
lowest threshold with a false-accept rate <= 1%. Set `FACE_MATCH_THRESHOLD` on the Configuration page accordingly.
Report YOUR numbers; the library's published accuracy does not describe your camera.

## Known limits
- Names are drawn on the video in Latin characters only (an OpenCV limitation); the People table shows the full name.
- Identity needs several consistent frames (about 1 s); a face smaller than 60 px is tracked but not named.
- Twins/look-alikes: a match must beat the runner-up by `FACE_MATCH_MARGIN`; otherwise the person stays Unknown.
- Photos/screens of a person can fool simple face recognition (no liveness check). Do not use it as a door lock.
