"""W1 curriculum smoke: stub teacher, 200 families, 50k-row pool, bucket / acceptance report.

    .venv/bin/python scripts/w01_curriculum_smoke.py        ->  results/w01_curriculum_smoke/

The held-out token list is handed to the curriculum build as a file, which keeps the curriculum package itself free
of any import of the held-out drift module.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
tokens = REPO / "data/cache/heldout_tokens.txt"
tokens.parent.mkdir(parents=True, exist_ok=True)
tokens.write_text(subprocess.run([sys.executable, "-m", "static_student.drift.heldout"], check=True, capture_output=True, text=True).stdout)
subprocess.run([sys.executable, "-m", "static_student.curriculum.build", "specs/telemetry.spec", "--families", "200", "--rows", "50000",
                "--seed", "0", "--out", "results/w01_curriculum_smoke", "--reject-tokens", str(tokens)], check=True, cwd=REPO)
