import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from collect_results import collect  # noqa: E402


def test_collect_copies_small_files_and_and_never_weights(tmp_path):
    runs = tmp_path / "runs"
    (runs / "V12" / "seed0").mkdir(parents=True)
    (runs / "V12" / "seed0" / "history.csv").write_text("epoch,val_miou\n1,0.5\n")
    (runs / "V12" / "seed0" / "best.pt").write_bytes(b"x" * 10)
    (runs / "eval_V12").mkdir()
    (runs / "eval_V12" / "metrics.json").write_text("{}")
    (runs / "eval_V12" / "big.csv").write_bytes(b"x" * 2_000_000)
    (runs / "val_summary.csv").write_text("variant,mean\n")
    copied, skipped = collect(runs, tmp_path / "out", max_mb=1.0)
    names = {p.relative_to(tmp_path / "out" / "runs").as_posix() for p in copied}
    assert names == {"V12/seed0/history.csv", "eval_V12/metrics.json", "val_summary.csv"}
    assert {s.name for s, _ in skipped} == {"big.csv"}
    assert not list((tmp_path / "out").rglob("*.pt"))
