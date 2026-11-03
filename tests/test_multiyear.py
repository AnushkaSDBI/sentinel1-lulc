import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from lulc.config import load_years
from lulc.ee_export import write_atomic
from lulc.sharpness import boundary_counts, f1_from_counts

ROOT = Path(__file__).resolve().parents[1]


def _month(out, key, year, month, h=64, label=1):
    z = {"vv": np.full((h, h), -12.0, np.float16), "vh": np.full((h, h), -18, np.float16),
         "s1_n": np.full((h, h), 5, np.uint8), "dw_label": np.full((h, h), label, np.uint8),
         "dw_conf": np.full((h, h), 90, np.uint8), "dw_n": np.full((h, h), 3, np.uint8)}
    write_atomic(out / key / f"{year}-{month:02d}.npz", z, {})


def _train_all():
    spec = importlib.util.spec_from_file_location("train_all", ROOT / "scripts" / "train_all.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_all"] = mod
    spec.loader.exec_module(mod)
    return mod


class R:                      # minimal region stand-in
    def __init__(self, key, split): self.key, self.split = key, split


def test_years_default_and_overlap_guard(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("year: 2021\nregions: {}\n")
    assert load_years(f) == {"train": [2021], "val": [2021], "test": [2021]}      # old one-year behaviour
    f.write_text("year: 2021\nyears: {train: [2021, 2022], val: [2022], test: [2025, 2026]}\nregions: {}\n")
    assert load_years(f)["test"] == [2025, 2026]
    f.write_text("year: 2021\nyears: {train: [2021, 2025], test: [2025]}\nregions: {}\n")
    with pytest.raises(ValueError):
        load_years(f)                                                              # a test year may not be a training year


def test_multi_year_stores_are_separate_and_missing_year_is_skipped(tmp_path):
    for y in (2021, 2022):
        for m in (1, 2, 3):
            _month(tmp_path, "a", y, m)
    _month(tmp_path, "b", 2021, 1)                      # region b has no 2022 data
    ta = _train_all()
    regs = [R("a", "train"), R("b", "train")]
    one = ta.load_splits(tmp_path, regs, {"train"}, 2021, 0.0, False, True)
    assert set(one) == {"a", "b"}                       # single year keeps the old keys
    many = ta.load_splits(tmp_path, regs, {"train"}, [2021, 2022], 0.0, False, True)
    assert set(many) == {"a@2021", "a@2022", "b@2021"}
    assert many["a@2022"].key == "a@2022" and many["a@2022"].stats is not None
    with pytest.raises(SystemExit):
        ta.load_splits(tmp_path, regs, {"train"}, [2021, 2022], 0.0, False, False, True)   # agree filter is 2021 only


def test_boundary_f1_detects_soft_and_speckly_borders():
    ref = np.zeros((40, 40), np.uint8); ref[:, 20:] = 1
    assert f1_from_counts(boundary_counts(ref, ref)) == 1.0
    shifted = np.zeros_like(ref); shifted[:, 23:] = 1
    assert f1_from_counts(boundary_counts(shifted, ref, 1)) < 0.6          # 3 px off with 1 px tolerance
    assert f1_from_counts(boundary_counts(shifted, ref, 3)) == 1.0
    noisy = ref.copy(); noisy[np.random.default_rng(0).random(ref.shape) < 0.1] ^= 1
    c = boundary_counts(noisy, ref, 1)
    assert c[1] > 5 * c[3]                                                 # speckle makes far more predicted border pixels
    ign = ref.copy(); ign[:, :] = 255
    assert boundary_counts(ref, ign)[1] == 0                              # no reference -> nothing counted


def test_boundary_band_and_ce():
    torch = pytest.importorskip("torch")
    from lulc.train import boundary_band, boundary_ce
    y = torch.zeros(1, 8, 8, dtype=torch.long); y[:, :, 4:] = 1; y[:, 0, 0] = 255
    b = boundary_band(y)
    assert b[0, 3].nonzero().flatten().tolist() == [2, 3, 4, 5] and not b[0, 0, 0]
    lg, cw = torch.randn(1, 6, 8, 8), torch.ones(6)
    plain = torch.nn.functional.cross_entropy(lg, y, weight=cw, ignore_index=255)
    assert torch.allclose(boundary_ce(lg, y, cw, 0.0), plain)
    assert boundary_ce(lg, y, cw, 3.0).item() != plain.item()
