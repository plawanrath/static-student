#!/usr/bin/env bash
# A certified binary for every shipped configuration. Each build scores its calibration pool through its own compiled
# kernel, sharded across cores, so the certificate belongs to the bits in that binary.
cd "$(dirname "$0")/.." || exit 1
run() { echo "[all] START $*"; "$@"; echo "[all] EXIT $? :: $*"; }
for SIZE in S XS; do
  for BITS in 4 3 2; do
    run .venv/bin/python -m static_student.build --spec specs/telemetry.spec --model models/tel-mlx-$SIZE-w$BITS \
        --out build/tel_${SIZE}_w${BITS} --cal 20000
  done
done
echo "[all] done"
