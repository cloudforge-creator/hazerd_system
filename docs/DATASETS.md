# Datasets to download (Phase M2)
Keep raw data out of Git (see .gitignore). Put files under data/raw/.

| Model | Dataset | Where to get it | Folder |
|---|---|---|---|
| Gas | UCI "Gas Sensor Array Drift" (batch1.dat ... batch10.dat) | UCI Machine Learning Repository | data/raw/gas/ |
| Temperature | NAB "machine_temperature_system_failure.csv" + NAB labels (combined_labels.json) | github.com/numenta/NAB | data/raw/temperature/ |
| Fire | D-Fire (fire/smoke images, YOLO labels) | github.com/gaiasd/DFireDataset | data/raw/fire/ |

Check each dataset's license and record version, class counts and the split seed in docs.

## Layout the scripts expect
- Gas:  data/raw/gas/batch1.dat ... batch10.dat
- Temperature: `python -m ml.train_temp_lstm --download-nab` fetches NAB itself
- Fire: data/raw/fire/<any>/images/*.jpg with sibling <any>/labels/*.txt (YOLO). D-Fire class ids: 0 = smoke, 1 = fire.
  Images are re-split by near-duplicate group (not by D-Fire's own train/test folders) to prevent leakage.
