# Training the fire CNN on Google Colab (free GPU)

CPU training of MobileNetV2 on ~21k images takes hours. Colab's free GPU takes minutes.

1. Colab -> Runtime -> Change runtime type -> **T4 GPU**.
2. Put the D-Fire dataset on Google Drive (folder `DFire` containing `train/images`, `train/labels`,
   `test/images`, `test/labels`) and upload `hazard_system.zip` to Drive as well.
3. Run in Colab cells:
```python
from google.colab import drive; drive.mount('/content/drive')
!unzip -q /content/drive/MyDrive/hazard_system.zip -d /content
%cd /content/hazard_system
!pip install -q tensorflow==2.18.0 opencv-python pillow scikit-learn pandas
!python -m ml.prepare_fire_dataset --root /content/drive/MyDrive/DFire --out data/fire_manifest.csv
!python -m ml.train_cnn --manifest data/fire_manifest.csv --epochs-head 5 --epochs-ft 5
```
4. Download `models/fire_cnn.keras`, `models/fire_cnn_meta.json` and `models/metrics/cnn_metrics.json`
   (right-click in the Files panel) into the same folders in your local project.
5. Use the SAME TensorFlow version locally (`pip install -r requirements-ml.txt`) so the model loads.
6. Try it without a webcam: `python -m src.main --sim --scenario fire --camera path\to\fire_video.mp4`
   (a video file loops; a folder of images cycles).

Read `models/metrics/cnn_metrics.json`: the target is fire-vs-rest F1 >= 0.90 and recall >= 0.92 on the TEST
split. If it is not met, report the real numbers and what you tried; do not tune on the test split.
