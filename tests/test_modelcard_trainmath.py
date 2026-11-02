import numpy as np
import pytest

from lulc.modelcard import build_model_card, read_model_card, write_model_card
from lulc.train_math import class_weights, cosine_warmup_lr


def _card(**kw):
    args = dict(class_names=["a", "b"], channels=["vv", "vh"],
                norm={"vv": (-10.0, 5.0), "vh": (-17.0, 5.5)},
                composite={"method": "median_db", "angle_range": None},
                window_months=3, patch=256, ignore_index=255)
    args.update(kw)
    return build_model_card(**args)


def test_model_card_round_trip(tmp_path):
    p = tmp_path / "card.json"
    write_model_card(p, _card())
    card = read_model_card(p)
    assert card["norm"]["vv"] == {"mean": -10.0, "std": 5.0}
    assert card["composite"]["method"] == "median_db"


def test_model_card_validation(tmp_path):
    with pytest.raises(ValueError):
        _card(norm={"vv": (0, 1)})
    with pytest.raises(ValueError):
        _card(norm={"vv": (0, 0.0), "vh": (0, 1)})
    p = tmp_path / "c.json"
    write_model_card(p, {**_card(), "schema_version": 99})
    with pytest.raises(ValueError):
        read_model_card(p)


def test_class_weights_rare_class_gets_more_weight_and_mean_is_one():
    w = class_weights([0.6, 0.3, 0.1])
    assert w[2] > w[1] > w[0]
    assert w.mean() == pytest.approx(1.0)


def test_class_weights_absent_class_is_zero():
    w = class_weights([0.7, 0.0, 0.3])
    assert w[1] == 0 and w[[0, 2]].mean() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        class_weights([0, 0])


def test_cosine_warmup_shape():
    total, warm, base = 100, 10, 3e-4
    lrs = [cosine_warmup_lr(s, total, warm, base) for s in range(total)]
    assert lrs[0] < lrs[warm - 1] == pytest.approx(base)
    assert all(a >= b for a, b in zip(lrs[warm:], lrs[warm + 1:]))
    assert lrs[-1] == pytest.approx(1e-6, abs=1e-8)
    with pytest.raises(ValueError):
        cosine_warmup_lr(0, 10, 10, base)
