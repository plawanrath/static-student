#!/usr/bin/env bash
# W2 chain: curriculum with D13/D14 -> S student -> G0 -> teacher family run (resumable: rerun the last command with --resume).
cd "$(dirname "$0")/.." || exit 1
set -x
.venv/bin/python -m static_student.curriculum.build specs/telemetry.spec --families 12000 --rows 20000 --seed 3 --run telemetry-stub-s3-12k --reject-tokens data/cache/heldout_tokens.txt
.venv/bin/python -m static_student.student.train --families data/curriculum/telemetry-stub-s3-12k/families.jsonl --size S --out models/tel-stub12k-v2-S-fp --train-rows 1000000 --epochs 3
.venv/bin/python scripts/w02_g0_rescue.py --model models/tel-stub12k-v2-S-fp --seeds 20
.venv/bin/python -m static_student.curriculum.build specs/telemetry.spec --teacher mlx --families 1300 --rows 50000 --seed 0 --run telemetry-mlx-s0 --out results/w02_teacher_telemetry --reject-tokens data/cache/heldout_tokens.txt --card-drop session_id client_id rtt --resume
echo "chain exit $?"
