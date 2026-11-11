import numpy as np
import pytest

from ml.eval_faces import evaluate
from src.personnel.registry import EMBED_DIM, normalize


def make(n_people=5, n_imgs=8, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for p in range(n_people):
        centre = rng.normal(size=EMBED_DIM)
        out[f"Person {p}"] = [normalize(centre + rng.normal(size=EMBED_DIM) * noise * np.linalg.norm(centre) / np.sqrt(EMBED_DIM)) for _ in range(n_imgs)]
    return out


def test_clean_embeddings_are_recognised_and_strangers_rejected():
    r = evaluate(make(noise=0.4))
    row = {x["threshold"]: x for x in r["sweep"]}[0.35]
    assert row["correct_name_rate"] > 0.95 and row["false_accept_rate"] < 0.02 and row["wrong_name_rate"] < 0.02
    assert r["suggested_threshold_for_FAR<=1%"] is not None and r["people"] == 5


def test_higher_threshold_never_increases_false_accepts_and_noise_hurts():
    r = evaluate(make(noise=0.4))
    fa = [x["false_accept_rate"] for x in r["sweep"]]
    assert all(b <= a + 1e-9 for a, b in zip(fa, fa[1:]))
    noisy = evaluate(make(noise=3.0, seed=1))
    clean_best = max(x["correct_name_rate"] for x in r["sweep"])
    assert max(x["correct_name_rate"] for x in noisy["sweep"]) < clean_best


def test_too_little_data_is_refused():
    with pytest.raises(ValueError, match="at least 3 people"):
        evaluate(make(n_people=2))
    with pytest.raises(ValueError):
        evaluate(make(n_imgs=3), k_enroll=3)


def test_unseparable_people_get_an_honest_warning():
    rng = np.random.default_rng(5)
    same = {f"P{i}": [normalize(rng.normal(size=EMBED_DIM) * 0.01 + 1.0) for _ in range(6)] for i in range(4)}
    r = evaluate(same)
    assert r["suggested_threshold_for_FAR<=1%"] is None and "no threshold" in r["warning"]
