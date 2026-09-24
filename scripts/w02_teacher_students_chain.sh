#!/usr/bin/env bash
# Real-teacher students: expand the teacher curriculum with drift rewrites, train S and XS, run the QAT ladder, certify every model.
cd "$(dirname "$0")/.." || exit 1
set -x
RUN=telemetry-mlx-s0-x12k; FAM=data/curriculum/$RUN/families.jsonl
[ -f $FAM ] || .venv/bin/python scripts/w02_expand_curriculum.py --src telemetry-mlx-s0 --families 12000 --seed 0
for SIZE in S XS; do
  FP=models/tel-mlx-$SIZE-fp; PREV=$FP
  [ -f $FP/student_fp.pt ] || .venv/bin/python -m static_student.student.train --families $FAM --size $SIZE --out $FP --train-rows 1000000 --epochs 3
  .venv/bin/python scripts/w02_certify_pilot.py --model $FP --no-broken
  for BITS in 4 3 2; do
    OUT=models/tel-mlx-$SIZE-w$BITS
    [ -f $OUT/student.pt ] || .venv/bin/python -m static_student.student.qat --init $PREV --teacher $FP --bits $BITS --out $OUT
    .venv/bin/python scripts/w02_certify_pilot.py --model $OUT --no-broken
    PREV=$OUT
  done
done
echo "teacher students chain exit $?"
