"""Released checkpoint format: `config.json` + `model.safetensors`, as published on the Hugging Face Hub.

A student checkpoint on disk during training is `student.pt` (QAT) or `student_fp.pt` (float), a pickled dict with
`config`, `state` and, for QAT stages, `bits`. The release keeps that file for the RUNBOOK commands and adds a
framework-neutral copy: the state dict as safetensors and the constructor arguments as JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from static_student.student import quant
from static_student.student.model import Student, StudentConfig

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"
RAW_NAMES = ("student.pt", "student_fp.pt")


def raw_checkpoint(model_dir: Path) -> Path:
    for n in RAW_NAMES:
        if (model_dir / n).exists():
            return model_dir / n
    raise FileNotFoundError(f"no {RAW_NAMES} in {model_dir}")


def export_student(model_dir: Path, out_dir: Path) -> dict:
    """Write config.json + model.safetensors for one student directory; returns the config written."""
    ck = torch.load(raw_checkpoint(model_dir), map_location="cpu")
    state = {k: v.contiguous() for k, v in ck["state"].items()}
    cfg = {
        "model_type": "static_student",
        "architecture": "static_student.student.model.Student",
        "config": dict(ck["config"]),
        "bits": int(ck.get("bits") or 0) or None,
        "n_params": int(sum(v.numel() for v in state.values())),
        "torch_dtype": "float32",
        "raw_checkpoint": raw_checkpoint(model_dir).name,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(state, str(out_dir / WEIGHTS_NAME), metadata={"format": "pt"})
    (out_dir / CONFIG_NAME).write_text(json.dumps(cfg, indent=1) + "\n")
    return cfg


def load_released(path_or_repo: str | Path, revision: str | None = None) -> tuple[Student, dict]:
    """Load a released student from a local directory or a Hub repo id (downloads config + weights only)."""
    p = Path(path_or_repo)
    if (p / CONFIG_NAME).exists():
        cfg_path, w_path = p / CONFIG_NAME, p / WEIGHTS_NAME
    else:
        from huggingface_hub import hf_hub_download

        cfg_path = Path(hf_hub_download(str(path_or_repo), CONFIG_NAME, revision=revision))
        w_path = Path(hf_hub_download(str(path_or_repo), WEIGHTS_NAME, revision=revision))
    cfg = json.loads(Path(cfg_path).read_text())
    m = Student(StudentConfig(**cfg["config"]))
    if cfg.get("bits"):
        quant.to_qat(m, cfg["bits"])
    m.load_state_dict(load_file(str(w_path)))
    return m.eval(), cfg
