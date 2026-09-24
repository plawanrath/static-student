#!/usr/bin/env bash
# The M student (ADR-0002 capacity check): float, then the 4/3/2-bit ladder, then a certificate for each.
cd "$(dirname "$0")/.." || exit 1
run() { echo "[m] START $*"; "$@"; echo "[m] EXIT $? :: $*"; }
FAM=data/curriculum/telemetry-mlx-s0-x12k/families.jsonl
FP=models/tel-mlx-M-fp; PREV=$FP
[ -f $FP/student_fp.pt ] || run .venv/bin/python -m static_student.student.train --families $FAM --size M --out $FP --train-rows 1000000 --epochs 3
for BITS in 4 3 2; do
  OUT=models/tel-mlx-M-w$BITS
  [ -f $OUT/student.pt ] || run .venv/bin/python -m static_student.student.qat --init $PREV --teacher $FP --bits $BITS --out $OUT
  PREV=$OUT
done
echo "[m] done"
