#!/usr/bin/env bash
# ADR-0009: the activation scale is now per token, so every QAT stage is retrained. The float students are unchanged
# (activation quantization lives only inside QLinear), so only the ladders are rebuilt, then every model is certified.
# Each stage records its exit code, so a stage that dies is visible in the log instead of looking like a quiet stop.
cd "$(dirname "$0")/.." || exit 1
run() { echo "[chain] START $*" ; "$@" ; echo "[chain] EXIT $? :: $*" ; }
for SIZE in S XS; do
  FP=models/tel-mlx-$SIZE-fp; PREV=$FP
  for BITS in 4 3 2; do
    OUT=models/tel-mlx-$SIZE-w$BITS
    rm -rf $OUT
    run .venv/bin/python -m static_student.student.qat --init $PREV --teacher $FP --bits $BITS --out $OUT
    [ -f $OUT/student.pt ] || { echo "[chain] ABORT: $OUT was not produced"; exit 1; }
    run .venv/bin/python scripts/w02_certify_pilot.py --model $OUT --no-broken
    PREV=$OUT
  done
done
run .venv/bin/python scripts/w03_kernel_parity.py --model models/tel-mlx-S-w4 --n 10000
run .venv/bin/python scripts/w03_kernel_parity.py --model models/tel-mlx-XS-w4 --n 10000
echo "[chain] done"
