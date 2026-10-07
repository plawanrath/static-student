"""Shared constants for the checkpoint release: repo naming, grouping, citation."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "models"
RELEASE = REPO / "release"
NAMESPACE = "plawanrath"
PREFIX = "static-student-"
GITHUB = "https://github.com/plawanrath/static-student"
COLLECTION_TITLE = "static-student: Weights as Constants checkpoints"
COLLECTION_DESCRIPTION = (  # the Hub caps this at 150 characters
    "Byte-level telemetry students (float, 4/3/2-bit QAT) and ModernBERT toxicity classifiers, compiled into binaries as constants."
)

BIBTEX = """@software{rath2026staticstudent,
  author  = {Rath, Plawan Kumar},
  title   = {static-student: Weights as Constants --- compiling distilled task models into executables as read-only data},
  year    = {2026},
  doi     = {10.5281/zenodo.23202887},
  url     = {https://github.com/plawanrath/static-student},
  license = {Apache-2.0},
  version = {0.1.0}
}"""


def repo_id(name: str) -> str:
    return f"{NAMESPACE}/{PREFIX}{name}"


def model_dirs() -> list[Path]:
    return sorted(p for p in MODELS.iterdir() if p.is_dir() and not p.name.startswith("."))


def kind(name: str) -> str:
    if name.startswith("modernbert"):
        return "modernbert"
    if name.startswith("tel-mlx"):
        return "student"
    return "student-superseded"
