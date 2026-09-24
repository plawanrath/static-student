# RUNBOOK

## 0. One-time setup

```bash
bash scripts/env/setup.sh            # Python 3.11 venv from the frozen lock for this platform, plus `pip install -e .`
bash scripts/env/setup.sh --models   # also download the pinned build-time teacher (Apple silicon, ~25 GB)
bash scripts/env/setup.sh --docker   # Linux x86-64 host: build the pinned C toolchain container
```

The teacher is pinned in `scripts/env/models.lock.txt` (`.venv/bin/python scripts/env/pin_models.py` re-resolves
`models.txt`). It runs only at build time; nothing it produces other than the curriculum reaches a binary.

`scripts/env/requirements.txt` is the loose, hand-edited set. `scripts/env/requirements.lock.<platform>.txt` is the
frozen set from a real install on that platform (`bash scripts/env/lock.sh` regenerates it). On a platform without a
lock, `setup.sh` resolves from the loose set and writes a new lock.

## 1. Tests

```bash
.venv/bin/python -m pytest -q -m "not slow and not gpu"     # fast subset (CI)
.venv/bin/python -m pytest -q                               # everything
```

Native pieces (the legacy parser) need a C++17 compiler, `make`, `pkg-config` and RE2 (`brew install re2` /
`apt-get install libre2-dev`): `make -C csrc` builds into `build/`. The Python harness builds on demand.

## 2. Data

`data/README.md` lists every tracked file, every pool (n, seed) and every third-party source with its hash and licence.

```bash
bash scripts/data/fetch_logevol.sh                                           # LOGEVOL -> data/raw/logevol (hash-checked)
git clone https://github.com/ua-parser/uap-core.git data/raw/uap-core        # full history; scripts pin the commit
```

## 3. Experiments

One runner per experiment under `scripts/wNN_*.py`, writing `results/wNN_<exp>/` (summary JSON + per-row JSONL).
Long runs: `.venv/bin/python scripts/... 2>&1 | tee logs/<name>.log`.

| result | command |
|---|---|
| `results/w01_curriculum_smoke/` (stub teacher: 200 families, acceptance rate, 50k-row pool) | `.venv/bin/python scripts/w01_curriculum_smoke.py` |
| `results/w01_drift_smoke/` (frozen legacy parser alone on drift timelines: no-match, silent-wrong, yield per track and step; regime per operator instance) | `.venv/bin/python scripts/w01_drift_smoke.py` |
| `results/w01_ua_timeline/` (first-seen date per uap-core fixture, freeze-date candidates, frozen regex list on later fixtures) | `.venv/bin/python scripts/w01_ua_timeline.py` |
| `results/w02_teacher_pilot/` (real teacher, 40 families: acceptance by bucket and rejection reason, seconds) | `.venv/bin/python -m static_student.curriculum.build specs/telemetry.spec --teacher mlx --families 40 --rows 5000 --seed 0 --run telemetry-mlx-pilot --out results/w02_teacher_pilot --reject-tokens data/cache/heldout_tokens.txt` (run `scripts/w01_curriculum_smoke.py` once first: it writes the token file) |
| `results/w01_drift_smoke_rate025/` (same smoke at 0.25 operator arrivals per source and step; the default is 0.1) | `.venv/bin/python scripts/w01_drift_smoke.py --rate 0.25` |
| `results/w01_ua_rolling/` (regex list frozen at every 1 January 2015–2026; failure split by staleness of the list; pools and vocabulary per freeze date) | `.venv/bin/python scripts/w01_ua_rolling.py` |
| `data/curriculum/telemetry-stub-s2-12k/` (12,000 stub families) | `.venv/bin/python -m static_student.curriculum.build specs/telemetry.spec --families 12000 --rows 20000 --seed 2 --run telemetry-stub-s2-12k --reject-tokens data/cache/heldout_tokens.txt` |
| `models/tel-stub12k-XS-fp/` (float XS student; about 20 min on Apple silicon) | `.venv/bin/python -m static_student.student.train --families data/curriculum/telemetry-stub-s2-12k/families.jsonl --size XS --out models/tel-stub12k-XS-fp --train-rows 1000000 --epochs 3` |
| `results/w02_g0_tel-stub12k-XS-fp/` (gate G0: rescue rate, risk on rescued lines, yield gain, per track and step; defer head and max-probability scores) | `.venv/bin/python scripts/w02_g0_rescue.py --model models/tel-stub12k-XS-fp` |
| `models/tel-stub12k-S-fp/`, `results/w02_g0_tel-stub12k-S-fp/` (same for the S student; about 45 min) | `.venv/bin/python -m static_student.student.train --families data/curriculum/telemetry-stub-s2-12k/families.jsonl --size S --out models/tel-stub12k-S-fp --train-rows 1000000 --epochs 3 && .venv/bin/python scripts/w02_g0_rescue.py --model models/tel-stub12k-S-fp` |
| `results/w02_operator_regimes/` (regime of every drift operator for the frozen legacy parser, one rewrite at a time) | `.venv/bin/python scripts/w02_operator_regimes.py` |
| `results/w01_drift_smoke/` as currently recorded (20 timelines per track) | `.venv/bin/python scripts/w01_drift_smoke.py --seeds 20` |
| curriculum with D13/D14, S student, G0 on 20 timelines, then the teacher family run (overnight; rerun the last line with `--resume` after an interruption) | `bash scripts/w02_night_chain.sh` |
| `results/w02_certify_pilot_<model>/` (Learn-then-Test certificate for a float student, and the failing build for a corrupted one) | `.venv/bin/python scripts/w02_certify_pilot.py --model models/tel-stub12k-v2-S-fp --device mps` |
| `models/tel-stub12k-v2-{S,XS}-w{4,3,2}/` (QAT ladder, each stage distilled from the float student; about 5 h) | `bash scripts/w02_qat_chain.sh` |
| `data/curriculum/telemetry-mlx-s0-x12k/`, `models/tel-mlx-*`, `results/w02_certify_pilot_tel-mlx-*/` (teacher curriculum expanded with drift rewrites; S and XS; 4/3/2-bit ladder; one certificate per model; about 5 h) | `bash scripts/w02_teacher_students_chain.sh` |
| `build/gen_<model>/` + `results/w03_kernel_parity_<model>/` (generate C, build it, compare the kernel with the NumPy reference bit for bit and with the float model) | `.venv/bin/python scripts/w03_kernel_parity.py --model models/tel-mlx-S-w4 --n 10000` |
| retrain every QAT stage under ADR-0009, certify each, then run both parity checks (about 3 h) | `bash scripts/w03_requantize_chain.sh` |
| `build/<name>/ss_hybrid` (one command from a spec line and a trained student to a certified binary; no binary if the contract cannot be certified) | `.venv/bin/python -m static_student.build --spec specs/telemetry.spec --model models/tel-mlx-S-w4 --out build/tel_S_w4` |
| the certificate carried by a built binary | `build/<name>/ss_hybrid --certificate` |
| `results/w03_startup/` (start-to-first-struct for the null binary, every built configuration and ONNX Runtime, measured by a C parent) | `.venv/bin/python scripts/w03_startup.py --launches 400` |
| `build/onnx/<name>/student.onnx` (export for the engine baselines) | `.venv/bin/python -m static_student.export_onnx --model models/tel-mlx-S-fp --out build/onnx/tel_S_fp` |
| a certified binary for every configuration | `bash scripts/w03_build_all.sh` |
| `results/w03_throughput/` (per-payload latency and throughput against the legacy parser, swept over no-match rate, audit probability with and without the asynchronous queue, and student budget) | `.venv/bin/python scripts/w03_throughput.py --model S_w4 --n 200000` |
