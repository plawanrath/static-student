"""The released checkpoint format (config.json + model.safetensors) reproduces the training checkpoint exactly."""
from pathlib import Path

import torch

from static_student.student.model import SIZES, Student
from static_student.student.release import export_student, load_released
from static_student.student import quant


def _save_raw(tmp: Path, bits: int | None) -> Path:
    torch.manual_seed(0)
    m = Student(SIZES["XS"])
    if bits:
        quant.to_qat(m, bits)
    d = tmp / "model"
    d.mkdir(parents=True)
    ck = {"config": SIZES["XS"].as_dict(), "state": m.state_dict()}
    if bits:
        ck["bits"] = bits
    torch.save(ck, d / ("student.pt" if bits else "student_fp.pt"))
    return d


def test_release_round_trip_float_and_qat(tmp_path):
    ids = torch.randint(0, 256, (2, 40))
    for bits in (None, 4):
        src = _save_raw(tmp_path / str(bits), bits)
        cfg = export_student(src, tmp_path / f"out{bits}")
        assert cfg["bits"] == bits and (tmp_path / f"out{bits}" / "model.safetensors").exists()
        released, cfg2 = load_released(tmp_path / f"out{bits}")
        raw = torch.load(next(src.glob("student*.pt")), map_location="cpu")
        orig = Student(SIZES["XS"])
        if bits:
            quant.to_qat(orig, bits)
        orig.load_state_dict(raw["state"])
        orig.eval()
        with torch.no_grad():
            a, b = released(ids), orig(ids)
        assert cfg2["config"] == SIZES["XS"].as_dict()
        assert all(torch.equal(a[k], b[k]) for k in a)
