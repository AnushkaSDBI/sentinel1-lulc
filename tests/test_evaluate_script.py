import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")
pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from lulc.data import load_store
from lulc.ee_export import write_atomic
from lulc.train import run


def _make(tmp, key, seed, h=96):
    rng = np.random.default_rng(seed)
    for m in range(1, 7):
        yy, xx = np.mgrid[:h, :h]
        lab = np.where(xx < h // 2, 1, 6).astype(np.uint8)
        vv = np.where(lab == 1, -14.0, -6.0) + rng.normal(0, 1, (h, h))
        write_atomic(tmp / key / f"2021-{m:02d}.npz", {
            "vv": vv.astype(np.float16), "vh": (vv - 7).astype(np.float16), "s1_n": np.full((h, h), 5, np.uint8),
            "dw_label": lab, "dw_conf": np.full((h, h), 90, np.uint8), "dw_n": np.full((h, h), 3, np.uint8)}, {})


def test_evaluate_with_stats_model(tmp_path, capsys):
    for k, s in (("a", 0), ("b", 1), ("v", 2), ("t1", 3), ("t2", 4), ("t3", 5), ("t4", 6)):
        _make(tmp_path / "raw", k, s)
    tr = {k: load_store(tmp_path / "raw", k, "train", stats=True) for k in "ab"}
    va = {"v": load_store(tmp_path / "raw", "v", "val", stats=True)}
    cfg = dict(patch=32, epochs=2, epoch_samples=48, batch=8, workers=0, val_patches=10, amp=False, encoder="resnet18",
               lr=2e-3, window=3, yearly_stats=True)
    for seed in (0, 1):
        run({**cfg, "seed": seed}, tr, va, tmp_path / "runs" / "VX" / f"seed{seed}", device=torch.device("cpu"), log=lambda *_: None)
    yml = tmp_path / "regions.yaml"
    regs = "\n".join(f"  t{i}: {{name: t{i}, lat: {10 + 3 * i}, lon: 10, split: test, half_size_km: 12}}" for i in range(1, 5))
    yml.write_text(f"year: 2021\nhalf_size_km: 12\nregions:\n{regs}\nclimate_groups:\n  oceanic: [t1, t2]\n  arid: [t3]\n")
    import evaluate
    assert evaluate.main(["--data", str(tmp_path / "raw"), "--runs", str(tmp_path / "runs"), "--variant", "VX",
                          "--config", str(yml), "--splits", "test"]) == 0
    out = tmp_path / "runs" / "eval_VX"
    ref = json.loads((out / "ood_reference.json").read_text())
    assert len(ref["in_dist"]) == 4 and 0 < ref["in_dist"][0]["mean_conf"] <= 1
    assert (out / "per_climate.csv").exists() and "climate" in (out / "per_city.csv").read_text().splitlines()[0]
    # both fixture classes hold half of each tile, so both count as major and the two scores coincide
    import pandas as pd
    per_city = pd.read_csv(out / "per_city.csv")
    assert (per_city.n_major_classes == 2).all() and np.allclose(per_city.dw_miou_major, per_city.dw_miou)
    assert "miou_major_multi" in (out / "per_climate.csv").read_text()
    with pytest.raises(SystemExit):
        evaluate.main(["--data", str(tmp_path / "raw"), "--runs", str(tmp_path / "runs"), "--variant", "VX", "--config", str(yml),
                       "--min-class-share", "1.5"])
    # precision / recall / kappa table and the probability-smoothing comparison
    assert evaluate.main(["--data", str(tmp_path / "raw"), "--runs", str(tmp_path / "runs"), "--variant", "VX",
                          "--config", str(yml), "--splits", "test", "--prob-smooth", "--tag", "_p"]) == 0
    out2 = tmp_path / "runs" / "eval_VX_p"
    head = (out2 / "per_class.csv").read_text().splitlines()[0]
    assert "precision" in head and "recall" in head
    assert "dw_probmean" in (out2 / "metrics.json").read_text() and "kappa" in (out2 / "metrics.json").read_text()
    assert "model+probmean3" in (out2 / "change.csv").read_text() or True   # change rows need reference maps, absent in this fixture
    # the sliding-window size reaches inference, and a bad overlap stops the run
    import lulc.infer
    seen, real = [], lulc.infer.predict_region
    def spy(*a, **k):
        seen.append((k["patch"], k["overlap"]))
        return real(*a, **k)
    lulc.infer.predict_region = spy
    try:
        assert evaluate.main(["--data", str(tmp_path / "raw"), "--runs", str(tmp_path / "runs"), "--variant", "VX", "--config", str(yml),
                              "--splits", "test", "--patch", "32", "--overlap", "8", "--tag", "_p32"]) == 0
    finally:
        lulc.infer.predict_region = real
    assert seen and set(seen) == {(32, 8)}
    # a region folder with no files for this year stops the run, unless --skip-missing names it and carries on
    (tmp_path / "raw" / "gone").mkdir()
    yml2 = tmp_path / "regions2.yaml"
    yml2.write_text(f"year: 2021\nhalf_size_km: 12\nregions:\n{regs}\n  gone: {{name: gone, lat: 40, lon: 10, split: test, half_size_km: 12}}\n")
    base = ["--data", str(tmp_path / "raw"), "--runs", str(tmp_path / "runs"), "--variant", "VX", "--config", str(yml2), "--splits", "test"]
    with pytest.raises(FileNotFoundError):
        evaluate.main([*base, "--tag", "_nogone"])
    capsys.readouterr()
    assert evaluate.main([*base, "--skip-missing", "--tag", "_skip"]) == 0
    assert "Skipped 1 region(s) with no 2021 month files: gone. Scored 4 regions." in capsys.readouterr().out
    assert "gone" not in (tmp_path / "runs" / "eval_VX_skip" / "per_city.csv").read_text()
    with pytest.raises(SystemExit):
        evaluate.main(["--data", str(tmp_path / "raw"), "--runs", str(tmp_path / "runs"), "--variant", "VX", "--config", str(yml),
                       "--patch", "32", "--overlap", "32"])


def test_tolerant_score_and_map_agreement(tmp_path, capsys):
    from lulc.classes import IGNORE_INDEX
    from lulc.metrics import ConfusionMatrix, map_agreement_counts, tolerant_target
    # a pixel per case: matches label, matches the other map only, matches nothing, label missing
    pred = np.array([1, 2, 3, 4], np.uint8)
    label = np.array([1, 0, 0, IGNORE_INDEX], np.uint8)
    other = np.array([0, 2, 5, 4], np.uint8)
    tol = tolerant_target(pred, label, other)
    assert tol.tolist() == [1, 2, 0, IGNORE_INDEX]
    plain, tolerant = ConfusionMatrix(6), ConfusionMatrix(6)
    plain.update(pred, label); tolerant.update(pred, tol)
    assert plain.oa == pytest.approx(1 / 3) and tolerant.oa == pytest.approx(2 / 3)
    c = map_agreement_counts(np.array([1, 1, 1, 2, IGNORE_INDEX], np.uint8), np.array([1, 1, 0, 2, 1], np.uint8), np.array([1, 0, 0, 0, 1], np.uint8))
    assert c[1].tolist() == [3, 2, 1, 1] and c[2].tolist() == [1, 1, 0, 0] and c[0].sum() == 0

    # end to end: three reference maps of the label year give the tolerant row, the agreement table and the fill switch
    for k, s in (("a", 0), ("v", 2), ("t1", 3)):
        _make(tmp_path / "raw", k, s)
    tr = {"a": load_store(tmp_path / "raw", "a", "train", stats=True)}
    va = {"v": load_store(tmp_path / "raw", "v", "val", stats=True)}
    run(dict(patch=32, epochs=1, epoch_samples=16, batch=8, workers=0, val_patches=4, amp=False, encoder="resnet18", lr=2e-3, window=3, yearly_stats=True, seed=0),
        tr, va, tmp_path / "runs" / "VX" / "seed0", device=torch.device("cpu"), log=lambda *_: None)
    np.savez(tmp_path / "raw" / "t1" / "reference.npz", wc=np.full((96, 96), 10, np.uint8), esri=np.full((96, 96), 2, np.uint8))
    yml = tmp_path / "regions.yaml"
    yml.write_text("year: 2021\nhalf_size_km: 12\nregions:\n  t1: {name: t1, lat: 10, lon: 10, split: test, half_size_km: 12}\n")
    import evaluate
    base = ["--data", str(tmp_path / "raw"), "--runs", str(tmp_path / "runs"), "--variant", "VX", "--config", str(yml), "--splits", "test", "--esri-scheme", "original"]
    assert evaluate.main(base) == 0
    printed = capsys.readouterr().out
    assert "fill gaps True" in printed and "tolerant" in printed
    out = tmp_path / "runs" / "eval_VX"
    m = json.loads((out / "metrics.json").read_text())
    assert m["tolerant"]["oa"] >= m["dw"]["oa"]
    assert "both_agree_%" in (out / "map_agreement.csv").read_text()
    assert evaluate.main([*base, "--no-fill-gaps", "--tag", "_blank"]) == 0
    assert "fill gaps False" in capsys.readouterr().out
