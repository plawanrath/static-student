# static-student — Weights as Constants

Compiling distilled task models into executables as read-only data.

> **Status (October 2026): concluded, released as-is.** This was a research project aimed at a systems paper. The
> research did not reach a publishable result and is not being continued. The code, the measured results and all
> trained checkpoints are released so that the pipeline and the models can be reused. The checkpoints are on the
> Hugging Face Hub in the collection
> **[plawanrath/static-student](https://huggingface.co/collections/plawanrath/static-student)**; every model card
> links back here. Please see [Citing](#citing) if you use any of it.

## What it does

A build pipeline that distills a tiny task model from a one-line spec and links it into a native binary as read-only
data, with a generated C kernel, a calibrated deferral threshold, and a deterministic fallback.

```
spec line -> teacher-written curriculum -> tiny quantization-aware student + defer head + threshold
          -> generated C kernel + const weight array -> standard linker -> one native binary
```

At run time the unchanged legacy parser runs first. The student kernel, reading its weights in place, handles only the
inputs the legacy parser rejects, and answers only when its confidence clears a threshold that the build certified
on the exact integer kernel that ships (the build fails if the contract cannot be met). On a small sample of inputs
the legacy parser accepts, the student runs as a second opinion; no-match rate and confident-disagreement rate are
exported as drift signals. Both paths emit the same struct type.

The same compile-as-constants path was then applied to an off-the-shelf encoder (a ModernBERT-large toxicity
classifier, fine-tuned here) to test whether a threshold calibrated in a framework survives the kernel that ships.

## What was measured

All numbers regenerate from the commands in `RUNBOOK.md`; summaries live in `results/`. Every comparison carries a
paired bootstrap confidence interval (10,000 resamples, `static_student/stats.py`). The findings, including the
negative ones, in brief:

- **The emitted C kernel is bit-identical to the NumPy integer reference** (0 ULP) for every shipped student
  configuration, and the compiled ModernBERT-large kernels at 8 and 4 bit are bit-exact on 1,000/1,000 inputs
  (`results/w03_kernel_parity_*`, `results/w03_modernbert_toxicity_large_w{8,4}`).
- **Certification on the shipped kernel works, and refuses damaged builds.** All nine telemetry configurations
  (XS/S/M at 4/3/2 bit) certify a risk bound of 1% at δ = 0.05 on a locked 20,000-row calibration pool; five kinds of
  deliberate damage produce no executable (`build/tel_*/build_report.json` via `scripts/w03_build_all.sh`,
  `results/w03_broken_student`).
- **Certified coverage is flat in bit width.** S covers 78.9–79.8% and M about 79.8% at 4, 3 and 2 bit while the
  weight array shrinks by about 40%; only XS at 2 bit loses coverage (55.3% vs 60.8%). This contradicted the
  pre-registered hypothesis that lower precision would cost coverage, and is reported as a failed prediction.
- **A threshold calibrated in PyTorch does not transfer to an int8 ONNX Runtime export of public encoders**: the
  dynamic-int8 path changed 1.3–6.6% of accept decisions at the fp32-calibrated threshold, in one case collapsing
  accuracy to 49% (`results/w02_cert_gap_pilot`). fp16 paths, threads and batching changed at most 0.05%.
- **Cross-ISA identity holds** for the compiled kernels: outputs byte-identical on Apple M3 Ultra and Intel Xeon
  Platinum 8488C over 10,000 inputs, with identical guarantee records (`results/w03_x86`, `results/w03_x86_mb`).
- **Start-up** on an idle Apple-silicon machine, 1,000 launches: null binary 12.5 ms, XS student 15.9 ms, S student
  25.0 ms, ONNX Runtime 51.7 ms (`results/w03_startup`).
- **Throughput:** the hybrid matches the legacy parser at no drift; a rescue costs about 18 ms against about 1 µs for
  the regex path; the asynchronous audit queue recovers 141× at audit probability 0.01 (`results/w03_throughput`).
- **Where it fell short.** The x86 portable C path is slow (2.5 s per ModernBERT-large input single-threaded, 8× the
  ARM time); synthetic drift never produced silent-wrong outputs on held-out tracks, so the rescue-under-drift claim
  rests on in-curriculum drift only; the second task family (User-Agent strings) was taken to a timeline analysis only.

## Released checkpoints

25 model directories, 2.5 GB, each one a Hugging Face repo named `plawanrath/static-student-<name>`.

| group | repos | what |
|---|---|---|
| `tel-mlx-{XS,S,M}-fp` | 3 | float byte-level students trained on the teacher-written telemetry curriculum (`data/curriculum/telemetry-mlx-s0-x12k`) |
| `tel-mlx-{XS,S,M}-w{4,3,2}` | 9 | the quantization-aware ladder distilled from the float students; these are the weights the certified binaries carry |
| `tel-stub*` | 11 | earlier pilots on a stub-teacher curriculum; superseded, kept for the record |
| `modernbert-{base,large}-civil` | 2 | `answerdotai/ModernBERT-{base,large}` fine-tuned on Civil Comments (toxicity ≥ 0.5, 40k balanced rows, 1 epoch) |

Student sizes: XS = 3 layers × 96 (0.39 M parameters), S = 4 × 192 (2.0 M), M = 6 × 384 (11 M). Each student repo
has `model.safetensors` + `config.json` (loadable with the snippet below), the original `student.pt` the RUNBOOK
commands expect, and `train_report.json`.

```python
from static_student.student.release import load_released
model, meta = load_released("plawanrath/static-student-tel-mlx-S-w4")   # a Hub id or a local directory
```

To put a downloaded student back where the RUNBOOK commands look for it:

```bash
.venv/bin/python scripts/release/download_hf.py tel-mlx-S-w4        # -> models/tel-mlx-S-w4/
```

The ModernBERT repos load with `transformers` (`ModernBertForSequenceClassification`, labels `non_toxic` / `toxic`).

## Layout

- `specs/` — the English spec lines the pipeline compiles (`telemetry.spec`, `useragent.spec`).
- `csrc/` — C sources shared by every binary: the output struct and parser signature (`metric_struct.h`), the frozen
  legacy parser, the hybrid entry point, the start-up harness, the ONNX Runtime baseline host.
- `static_student/` — Python package: curriculum generation, drift operators, student training and QAT, certification
  (Learn-then-Test), code generation (packed layout, NumPy integer reference, C emitter), the ModernBERT integer
  reference, the compiler (`build.py`), audit counters, statistics.
- `scripts/` — one runner per experiment, writing to `results/wNN_<exp>/`; `scripts/env/` builds the environment;
  `scripts/release/` exports and uploads the checkpoints.
- `results/` — committed summaries (JSON) and per-row outputs (JSONL/TXT). Compiled binaries inside result bundles are
  not tracked.
- `data/` — small tracked data; every file and every third-party source is described in `data/README.md`.
- `tests/` — fast tests run in CI.
- `RUNBOOK.md` — environment setup and the exact command for every reported number.

Comments in the code cite design decisions as `ADR-00NN`. Those records were kept in a private planning tree that is
not part of this release; the comment next to each reference states the decision itself.

## Hardware

One Apple-silicon Mac (unified memory) for the build pipeline and ARM measurements; a rented x86-64 Linux host for the
cross-ISA check. No GPU is required. The build-time teacher (Mistral Small 3.2 24B, MLX 8-bit) needs about 25 GB of
memory and is only required to write new curricula.

## Citing

```bibtex
@software{rath2026staticstudent,
  author  = {Rath, Plawan Kumar},
  title   = {static-student: Weights as Constants --- compiling distilled task models into executables as read-only data},
  year    = {2026},
  url     = {https://github.com/plawanrath/static-student},
  license = {Apache-2.0},
  version = {0.1.0}
}
```

`CITATION.cff` carries the same entry; GitHub's "Cite this repository" button renders it in APA and BibTeX. Each
Hugging Face model card repeats it. The ModernBERT checkpoints derive from `answerdotai/ModernBERT` (Apache-2.0) and
`google/civil_comments` (CC0); please cite those too if you use them.

## License

Apache License 2.0, see `LICENSE`. The same license applies to the released checkpoints.
