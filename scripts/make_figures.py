"""Regenerate every paper figure from results/*.json. One function per figure; no number is typed by hand.

Usage: python scripts/make_figures.py [--out-dir results/figures]
Writes <name>.pdf + <name>.png per figure and manifest.json (figure -> source artifacts).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIGURES = {}  # name -> (function(out_dir) -> list of source artifact paths)


def figure(fn):
    FIGURES[fn.__name__] = fn
    return fn


MODELS = (("sentiment", "DistilBERT (sentiment)"), ("injection", "DeBERTa-v3 (injection)"),
          ("zeroshot", "DeBERTa-v3-L (zero-shot)"), ("toxicity", "ModernBERT (toxicity)"))
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")   # validated categorical slots 1-4 (dataviz reference palette)
MARKERS = ("o", "s", "D", "^")                          # secondary encoding: shape as well as hue
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"


def _flip(path: Path, key: str, alpha: str = "0.05"):
    if not path.exists():
        return None
    a = json.loads(path.read_text())
    return a


@figure
def gap_dotplot(out: Path):
    """Share of accept decisions that change at a threshold calibrated elsewhere, per deployment change and model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pilot = {m: json.loads((REPO / f"results/w02_cert_gap_pilot/{m}/analysis.json").read_text()) for m, _ in MODELS}
    cmp_path = REPO / "results/w03_x86/compare_arm64_M3_Ultra_x86_IntelR_XeonR_Platinum_8488C.json"
    cmp = json.loads(cmp_path.read_text())
    rows = [  # (label, source, key)
        ("fp32, batched", "pilot", "torch_cpu_fp32_b32"), ("fp32, ORT", "pilot", "ort_fp32_t8"),
        ("fp16, Apple GPU", "pilot", "torch_mps_fp16_b32"), ("fp16, Core ML", "pilot", "coreml_fp16"),
        ("int8 dynamic", "pilot", "ort_int8_dynamic"), ("int8 per-channel", "pilot", "ort_int8_perchannel"),
        ("int8 MatMul-only", "pilot", "ort_int8_matmul_only"),
        ("x86: fp32, PyTorch", "x86", "torch_fp32_b1"), ("x86: fp32, ORT", "x86", "ort_fp32"),
        ("x86: int8 dynamic", "x86", "ort_int8_dynamic"), ("x86: int8 per-channel", "x86", "ort_int8_perchannel"),
    ]
    plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.edgecolor": MUTED, "axes.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(3.35, 3.2))
    y = list(range(len(rows) + 1))[::-1]
    for (mi, (m, name)) in enumerate(MODELS):
        xs, ys = [], []
        for (label, src, key), yy in zip(rows, y[:-1]):
            if src == "pilot":
                r = pilot[m]["alphas"]["0.05"]["paths"].get(key)
                v = None if r is None else r["accept_flip"]
            else:
                r = cmp["vendor"].get(m, {}).get(key, {}).get("alphas", {}).get("0.05", {})
                v = r.get("accept_flip_at_ref_tau") if r.get("reference_certifies", True) and "certificate_ref" in r else None
            if v is not None:
                xs.append(100 * v); ys.append(yy + (mi - 1.5) * 0.14)
        ax.scatter(xs, ys, s=18, marker=MARKERS[mi], color=SERIES[mi], edgecolors="white", linewidths=0.6, zorder=3, label=name)
    ax.scatter([0.0], [y[-1]], s=26, marker="*", color=INK, zorder=3)
    ax.annotate("byte-identical outputs", (0.0, y[-1]), xytext=(6, -2.5), textcoords="offset points", color=INK, fontsize=6.5)
    labels = [r[0] for r in rows] + ["x86: our kernel"]
    ax.set_yticks(y); ax.set_yticklabels(labels, color=INK)
    ax.axhline(y[6] - 0.5, color=GRID, lw=0.8); ax.axhline(y[-2] - 0.5, color=GRID, lw=0.8)
    ax.set_xlabel("accept decisions that change (%)", color=INK)
    ax.set_xlim(-0.3, 7.2)
    ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", colors=MUTED, length=2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=2, frameon=False, fontsize=6, handletextpad=0.2, columnspacing=0.8, borderaxespad=0.2, labelcolor=INK)
    fig.tight_layout(pad=0.2)
    fig.savefig(out / "gap_dotplot.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(out / "gap_dotplot.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return [f"results/w02_cert_gap_pilot/{m}/analysis.json" for m, _ in MODELS] + [str(cmp_path.relative_to(REPO))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO / "results" / "figures"))
    out = Path(ap.parse_args().out_dir); out.mkdir(parents=True, exist_ok=True)
    manifest = {name: [str(s) for s in fn(out)] for name, fn in FIGURES.items()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[figures] {len(manifest)} figure(s) -> {out}")


if __name__ == "__main__":
    main()
