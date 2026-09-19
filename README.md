# Weights as Constants: Compiling Distilled Task Models into Executables as Read-Only Data

A build pipeline that distills a tiny task model from a one-line spec and links it into a native binary as read-only data, with a generated C kernel, a calibrated deferral threshold, and a deterministic fallback.

```
spec line -> teacher-written curriculum -> tiny quantization-aware student + defer head + threshold
          -> generated C kernel + const weight array -> standard linker -> one native binary
```

At run time the unchanged legacy parser runs first. The student kernel, reading its weights in place, handles only the
inputs the legacy parser rejects, and answers only when its confidence clears a threshold that the build certified
on the exact integer kernel that ships (the build fails if the contract cannot be met). On a small sample of inputs
the legacy parser accepts, the student runs as a second opinion; no-match rate and confident-disagreement rate are
exported as drift signals. Both paths emit the same struct type.

## Layout

- `specs/` — the English spec lines the pipeline compiles (`telemetry.spec`, `useragent.spec`).
- `csrc/` — C sources shared by every binary: the output struct and parser signature (`metric_struct.h`).
- `static_student/` — Python package: curriculum generation, drift operators, student training, certification,
  code generation, audit counters, statistics (`stats.py`: bootstrap CIs, 10k resamples, used for every comparison).
- `scripts/` — one runner per experiment, writing to `results/wNN_<exp>/`; `scripts/env/` builds the environment.
- `results/` — committed summaries (JSON) and per-row outputs (JSONL).
- `data/` — small tracked data; every file is described in `data/README.md`.
- `tests/` — fast tests run in CI.
- `RUNBOOK.md` — environment setup and the exact command for every reported number.

## Hardware

One Apple-silicon Mac (unified memory) for the build pipeline and ARM measurements; one x86-64 Linux host for the
second architecture. No GPU is required.

## License

See `LICENSE`.
