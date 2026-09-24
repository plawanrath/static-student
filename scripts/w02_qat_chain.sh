#!/usr/bin/env bash
# QAT ladder 4 -> 3 -> 2 for the S and XS students, each stage initialized from the previous one and distilled from the float student.
cd "$(dirname "$0")/.." || exit 1
set -x
FAM=data/curriculum/telemetry-stub-s3-12k/families.jsonl
[ -f models/tel-stub12k-v2-XS-fp/student_fp.pt ] || .venv/bin/python -m static_student.student.train --families $FAM --size XS --out models/tel-stub12k-v2-XS-fp --train-rows 1000000 --epochs 3
for SIZE in S XS; do
  FP=models/tel-stub12k-v2-$SIZE-fp; PREV=$FP
  for BITS in 4 3 2; do
    OUT=models/tel-stub12k-v2-$SIZE-w$BITS
    [ -f $OUT/student.pt ] || .venv/bin/python -m static_student.student.qat --init $PREV --teacher $FP --bits $BITS --out $OUT
    PREV=$OUT
  done
done
echo "qat chain exit $?"
