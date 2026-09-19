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
| `results/w01_drift_smoke_rate010/` (same smoke at 0.1 operator arrivals per source and step) | `.venv/bin/python scripts/w01_drift_smoke.py --rate 0.1` |
| `results/w01_ua_rolling/` (regex list frozen at every 1 January 2015–2026; failure split by staleness of the list; pools and vocabulary per freeze date) | `.venv/bin/python scripts/w01_ua_rolling.py` |
