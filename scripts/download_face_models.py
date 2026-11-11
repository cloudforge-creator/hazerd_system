"""Download the two OpenCV face models (YuNet detector, SFace recogniser) into models/face/.

    python scripts/download_face_models.py

If the download fails (firewall, proxy) download the two files in a browser from the OpenCV Model Zoo
(github.com/opencv/opencv_zoo -> models/face_detection_yunet and models/face_recognition_sface) and put them in
models/face/ with exactly these names:
    face_detection_yunet_2023mar.onnx
    face_recognition_sface_2021dec.onnx
"""
import os
import sys
import urllib.request

BASE = "https://github.com/opencv/opencv_zoo/raw/main/models/"
FILES = {
    "face_detection_yunet_2023mar.onnx": (BASE + "face_detection_yunet/face_detection_yunet_2023mar.onnx", 100_000),
    "face_recognition_sface_2021dec.onnx": (BASE + "face_recognition_sface/face_recognition_sface_2021dec.onnx", 10_000_000),
}
DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "face")


def sane(path: str, min_bytes: int) -> bool:
    if not os.path.exists(path) or os.path.getsize(path) < min_bytes:
        return False
    with open(path, "rb") as f:
        return not f.read(64).startswith(b"version https://git-lfs")      # a Git-LFS pointer, not the model


def main() -> int:
    os.makedirs(DEST, exist_ok=True)
    force = "--force" in sys.argv
    bad = 0
    for name, (url, min_bytes) in FILES.items():
        path = os.path.join(DEST, name)
        if sane(path, min_bytes) and not force:
            print(f"ok      {name} (already present)")
            continue
        try:
            print(f"fetch   {name} ...")
            urllib.request.urlretrieve(url, path)
        except Exception as exc:
            print(f"FAILED  {name}: {exc}")
        if not sane(path, min_bytes):
            print(f"FAILED  {name}: download is missing or not a valid model file; download it manually (see the top of this script)")
            bad += 1
        else:
            print(f"ok      {name} ({os.path.getsize(path) // 1024} KB)")
    print("\nNext: python scripts/check_face_setup.py" if not bad else "\nSome files failed; fix them, then re-run.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
