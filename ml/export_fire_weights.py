"""Export a trained fire CNN to portable NumPy weights.

    python -m ml.export_fire_weights --model models/fire_cnn.keras

Run it where fire_cnn.keras loads (Colab right after training, or this project's environment). The runtime uses
the .npz automatically when the .keras file cannot be loaded by the local TensorFlow/Keras version.
"""
import argparse
import os

from src.fire_cnn import WEIGHTS_NPZ, export_weights


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/fire_cnn.keras")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = a.out or os.path.join(os.path.dirname(a.model), WEIGHTS_NPZ)
    print("wrote", export_weights(a.model, out), f"({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
