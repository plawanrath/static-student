"""Export every models/<name>/ directory into release/<name>/ with a model card, ready for upload_hf.py.

Students: config.json + model.safetensors (static_student.student.release), the original student.pt / student_fp.pt
the RUNBOOK commands expect, train_report.json, and the kernel certificate from build/tel_<size>_w<bits>/ when present.
ModernBERT: the directory as saved by transformers, plus the card. Training checkpoints (checkpoint_last.pt) are not
released.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BIBTEX, GITHUB, MODELS, RELEASE, kind, model_dirs, repo_id  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from static_student.student.release import export_student, raw_checkpoint  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SIZE_DESC = {"XS": "3 layers, d_model 96, 4 heads", "S": "4 layers, d_model 192, 4 heads", "M": "6 layers, d_model 384, 6 heads"}


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def _certificate(name: str) -> dict | None:
    # the certified binaries were built from the teacher-curriculum QAT students only: build/tel_<size>_w<bits>
    parts = name.split("-")  # tel-mlx-S-w4
    if len(parts) != 4 or parts[1] != "mlx" or not parts[3].startswith("w"):
        return None
    rep = REPO / "build" / f"tel_{parts[2]}_{parts[3]}" / "build_report.json"
    if not rep.exists():
        return None
    return json.loads(rep.read_text()).get("certificate")


def citation_block() -> str:
    return (
        "## Citation\n\n"
        "If you use this checkpoint, please cite the repository:\n\n"
        f"```bibtex\n{BIBTEX}\n```\n\n"
        f"`CITATION.cff` in the repository carries the same entry (GitHub's \"Cite this repository\" button).\n"
    )


def student_card(name: str, cfg: dict, report: dict) -> str:
    size = report["size"]
    bits = cfg.get("bits")
    superseded = kind(name) == "student-superseded"
    # the two oldest reports predate the families_path field; same family counts as the s2-12k run / the first stub smoke
    fallback = "telemetry-stub-s2-12k" if "12k" in name else "telemetry-stub-s0 (early smoke run)"
    curriculum = Path(report["families_path"]).parent.name if "families_path" in report else fallback
    teacher_written = "mlx" in curriculum
    dev = [e for e in report["log"] if isinstance(e.get("epoch"), int) and e["epoch"] >= 0]
    final_dev = report["log"][-1]["dev_struct_exact"]
    tags = ["static-student", "weights-as-constants", "byte-level", "telemetry-parsing", "selective-prediction",
            "quantization-aware-training" if bits else "distillation", f"size-{size}"]
    if bits:
        tags.append(f"{bits}-bit")
    if superseded:
        tags.append("superseded")
    fm = {
        "license": "apache-2.0",
        "library_name": "pytorch",
        "language": ["en"],
        "tags": tags,
        "pipeline_tag": "token-classification",
    }
    if bits:
        fm["base_model"] = repo_id(report["init"].split("/")[-1])
        fm["base_model_relation"] = "quantized"
    lines = ["---", _yaml(fm), "---", ""]
    lines.append(f"# {repo_id(name)}")
    lines.append("")
    if superseded:
        lines.append("> **Superseded pilot.** Trained on a stub-teacher curriculum before the real teacher curriculum existed. "
                     "Kept for the record; the `tel-mlx-*` models in the same collection are the ones every reported "
                     "number was taken on.")
        lines.append("")
    what = (f"A {bits}-bit quantization-aware byte-level telemetry student ({size}: {SIZE_DESC[size]}), distilled from "
            f"the float student `{repo_id(report['init'].split('/')[-1])}`" if bits else
            f"A float byte-level telemetry student ({size}: {SIZE_DESC[size]}), trained from scratch")
    lines += [
        f"{what}, from the **static-student** project (*Weights as Constants*: compiling distilled task models into "
        f"native executables as read-only data). Code, RUNBOOK and results: {GITHUB}.",
        "",
        "## What the model does",
        "",
        "Input: the raw bytes of one telemetry payload (a metric line in one of many ad-hoc formats). Output: per-byte "
        "span pointers for the metric name, value and unit, a unit class, a value score, and a **defer logit**. The "
        "student is meant to run only on payloads a frozen regex parser rejects, and to answer only when its "
        "confidence clears a threshold certified on the integer kernel that ships; everything else is deferred. "
        "It is a research artifact for one synthetic task family, not a general log parser.",
        "",
        "## Training",
        "",
        "| | |",
        "|---|---|",
        f"| curriculum | `data/curriculum/{curriculum}` ({'written by Mistral Small 3.2 24B from the spec line, expanded with drift rewrites' if teacher_written else 'stub teacher'}) |",
        f"| families (train / dev / defer-fit) | {report['families']['train']} / {report['families']['dev']} / {report['families']['defer_fit']} |" if "families" in report else f"| init / distillation teacher | `{report.get('init')}` / `{report.get('teacher')}` |",
        f"| rendered training rows | {report['train_rows']:,} |",
        f"| epochs | {report['epochs']} |",
        f"| seed | {report['seed']} |",
        f"| parameters | {cfg['n_params']:,} |",
    ]
    if bits:
        lines.append(f"| packed weight bytes (trunk / total at {bits} bit) | {report['bytes']['trunk']:,} / {report['bytes']['total']:,} |")
        lines.append(f"| dev struct-exact after quantization, before fine-tuning | {_fmt_pct(report['log'][0]['dev_struct_exact'])} |")
    lines.append(f"| dev struct-exact, final{' (step sizes snapped to float16, as shipped)' if bits else ''} | {_fmt_pct(final_dev)} |")
    if "defer_fit_error_rate" in report:
        lines.append(f"| defer-head fit: error rate / BCE | {report['defer_fit_error_rate']:.4f} / {report['defer_head_bce']:.4f} |")
    lines.append(f"| wall time (Apple silicon, MPS) | {report['seconds'] / 60:.0f} min |")
    cert = _certificate(name)
    if cert:
        lines += [
            "",
            "## Certificate of the compiled kernel",
            "",
            "The weights were compiled into a C kernel with the weights as a constant array; the deferral threshold below "
            "was certified (Learn-then-Test) on that exact integer kernel, on a locked calibration pool, and travels in "
            "the binary.",
            "",
            "| | |",
            "|---|---|",
            f"| risk bound α / confidence δ / min coverage | {cert['alpha']} / {cert['delta']} / {cert['min_coverage']} |",
            f"| calibration pool n | {cert['n_calibration']:,} |",
            f"| threshold τ | {cert['tau']:.6f} |",
            f"| certified coverage | {_fmt_pct(cert['coverage'])} ({cert['n_accepted']:,} accepted, {cert['k_wrong']} wrong) |",
            f"| empirical risk on accepted | {_fmt_pct(cert['empirical_risk'])} |",
            f"| thresholds tested | {cert['thresholds_tested']} |",
            f"| pool sha256 | `{cert['pool_sha256'][:16]}…` |",
        ]
    lines += [
        "",
        "## Files",
        "",
        "- `model.safetensors`, `config.json` — framework-neutral weights and constructor arguments.",
        f"- `{raw_checkpoint(MODELS / name).name}` — the original PyTorch checkpoint the RUNBOOK commands expect "
        "(`torch.load`, dict with `config`, `state`" + (", `bits`" if bits else "") + ").",
        "- `train_report.json` — the training log above.",
        "",
        "## How to load",
        "",
        "```bash",
        f"git clone {GITHUB} && cd static-student && bash scripts/env/setup.sh",
        "```",
        "",
        "```python",
        "from static_student.student.release import load_released",
        f'model, cfg = load_released("{repo_id(name)}")',
        "```",
        "",
        "To compile it into a certified binary (needs a C compiler, `make`, RE2):",
        "",
        "```bash",
        f".venv/bin/python scripts/release/download_hf.py {name}",
        f".venv/bin/python -m static_student.build --spec specs/telemetry.spec --model models/{name} --out build/{name}",
        "```",
        "",
        "## Limitations",
        "",
        "- Trained and evaluated on synthetic telemetry payloads rendered from format families; never on production "
        "traffic. Numbers above are on held-out synthetic families.",
        "- The certificate is a statistical bound on the rate of wrong accepted answers under the calibration "
        "distribution; it says nothing about inputs from other distributions.",
        "- The research project was concluded without a publication; the model is released as-is for reuse and "
        "reproduction and is not maintained.",
        "",
        citation_block(),
    ]
    return "\n".join(lines)


def modernbert_card(name: str, report: dict) -> str:
    size = "large" if "large" in name else "base"
    fm = {
        "license": "apache-2.0",
        "library_name": "transformers",
        "language": ["en"],
        "base_model": report["backbone"],
        "base_model_relation": "finetune",
        "datasets": ["google/civil_comments"],
        "pipeline_tag": "text-classification",
        "tags": ["static-student", "weights-as-constants", "modernbert", "toxicity", "text-classification"],
    }
    kernel = {}
    for bits in (8, 4):
        p = REPO / "results" / f"w03_modernbert_toxicity_{size}_w{bits}" / "summary.json"
        if p.exists():
            kernel[bits] = json.loads(p.read_text())
    lines = ["---", _yaml(fm), "---", "", f"# {repo_id(name)}", ""]
    lines += [
        f"`{report['backbone']}` (revision `{report['backbone_revision']}`) fine-tuned for binary toxicity "
        "classification on Civil Comments, from the **static-student** project (*Weights as Constants*). It was trained "
        "as a realistic-size test model for compiling an encoder's weights into a native binary as a read-only "
        f"constant and certifying a decision threshold on the kernel that ships. Code and results: {GITHUB}.",
        "",
        "## Training",
        "",
        "| | |",
        "|---|---|",
        f"| data | `{report['data']}`, split `{report['train_split']}` |",
        f"| label | `toxic` if `{report['label']}`, else `non_toxic` |",
        f"| rows | {report['n_train']:,}, positive share {report['positive_share']} (balanced subsample) |",
        f"| epochs / steps / batch | {report['epochs']} / {report['steps']} / {report['batch']} |",
        f"| learning rate / max length | {report['lr']} / {report['max_len']} |",
        f"| seed | {report['seed']} |",
        f"| final training loss (last 100 steps) | {report['final_loss_100']:.3f} |",
        f"| wall time (Apple silicon, MPS) | {report['seconds'] / 60:.0f} min |",
    ]
    if kernel:
        lines += ["", "## Compiled-kernel results (from the repository's `results/`)", "", "| bits | weight bytes as a constant | kernel bit-exact to integer reference | accuracy fp32 → kernel | argmax agreement |", "|---|---|---|---|---|"]
        for bits, s in kernel.items():
            lines.append(f"| {bits} | {s['weight_bytes']:,} | {s['parity']['bit_exact']}/{s['parity']['n']} | "
                         f"{_fmt_pct(s['accuracy']['float_fp32'])} → {_fmt_pct(s['accuracy']['kernel'])} | {_fmt_pct(s['accuracy']['argmax_agreement'])} |")
        lines.append("")
        lines.append("Accuracy is on a held-out Civil Comments test pool; see `results/w03_modernbert_toxicity_large_w{8,4}/summary.json` for pool sizes, hashes and the guarantee records.")
    lines += [
        "",
        "## How to use",
        "",
        "```python",
        "from transformers import AutoTokenizer, AutoModelForSequenceClassification",
        f'tok = AutoTokenizer.from_pretrained("{repo_id(name)}")',
        f'model = AutoModelForSequenceClassification.from_pretrained("{repo_id(name)}")',
        'logits = model(**tok("you are wonderful", return_tensors="pt")).logits',
        "print(model.config.id2label[int(logits.argmax())])",
        "```",
        "",
        "## Intended use and limitations",
        "",
        "- A test model for the compile-as-constants pipeline, released so the repository's numbers can be reproduced. "
        "It is not a moderation system.",
        "- One epoch on 40k rows of English comments; toxicity labels are crowd annotations with known demographic "
        "biases (see the Civil Comments dataset card). Expect false positives on identity terms and dialect.",
        "- Max sequence length 128 tokens at training time.",
        "- The research project was concluded without a publication; the model is not maintained.",
        "",
        citation_block(),
        "",
        "Please also cite ModernBERT (Warner et al., 2024) and the Civil Comments dataset (Borkan et al., 2019).",
    ]
    return "\n".join(lines)


def _yaml(d: dict) -> str:
    out = []
    for k, v in d.items():
        if isinstance(v, list):
            out.append(f"{k}:")
            out += [f"  - {x}" for x in v]
        else:
            out.append(f"{k}: {v}")
    return "\n".join(out)


def export_one(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    report = json.loads((src / "train_report.json").read_text())
    if kind(src.name) == "modernbert":
        for f in src.iterdir():
            shutil.copy2(f, dst / f.name)
        card = modernbert_card(src.name, report)
    else:
        cfg = export_student(src, dst)
        for n in ("train_report.json", raw_checkpoint(src).name):
            shutil.copy2(src / n, dst / n)
        card = student_card(src.name, cfg, report)
    (dst / "README.md").write_text(card + "\n")
    shutil.copy2(REPO / "LICENSE", dst / "LICENSE")  # Apache-2.0, same as the code


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="model directory names; default: all under models/")
    args = ap.parse_args()
    dirs = [MODELS / n for n in args.names] if args.names else model_dirs()
    for d in dirs:
        export_one(d, RELEASE / d.name)
        print(f"{d.name:28s} -> {RELEASE / d.name}  ({sum(f.stat().st_size for f in (RELEASE / d.name).iterdir()) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
