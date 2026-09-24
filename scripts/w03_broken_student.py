"""Gate G1c, second half: a student that has been damaged must not produce a binary.

    .venv/bin/python scripts/w03_broken_student.py --model models/tel-mlx-S-w4   ->  results/w03_broken_student/

The certificate is computed on the compiled kernel's own scores, so the only way to check that the build really
refuses a bad model is to damage a trained one, run the whole pipeline on it, and confirm that no executable appears.
Damage of two kinds. *Model* damage changes the trained network: permuted pointer-head rows, sign-flipped weights in
one block, random weights in the last block. *Packing* damage changes the bytes on their way into the array, which is
where a real build accident lives: group scales shuffled so that they no longer correspond to the levels they scale,
and one block's levels zeroed.

A case that is worth stating plainly: re-quantizing with a different step size is **not** damage. `QLinear` keeps a
latent float weight and derives its levels as round(weight / step), so perturbing the step changes the grid while the
reconstructed weights, and the accuracy, survive (measured: 99.62% to 99.46% exact). Such a model certifies, and it
should. The certificate bounds the error rate of the bits that ship; it is not an integrity check, and it cannot tell
that those bits are not the ones the build intended. That is what the weight hash recorded in the certificate is for.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

import numpy as np

from static_student import build as build_mod
from static_student.codegen import pack as pack_mod
from static_student.student import quant
from static_student.student.qat import load_student

MODEL_DAMAGE = ("permuted_pointer_head", "sign_flipped_block", "random_last_block")
PACK_DAMAGE = ("shuffled_group_scales", "zeroed_block_levels")


def corrupt_packed(kind: str):
    """Wrap the packer so the damage lands on the bytes that go into SS_W[], which is where a build accident lives."""
    original = pack_mod.pack

    def wrapped(model):
        p = original(model)
        rng = np.random.default_rng(0)
        for blk in p.blocks:
            for name in ("q", "k", "v", "o", "up", "down"):
                lin = blk[name]
                if kind == "shuffled_group_scales":
                    lin.scales[:] = lin.scales[rng.permutation(lin.scales.shape[0])]
                elif kind == "zeroed_block_levels" and name == "down":
                    lin.levels[:] = 0
        return p
    return wrapped

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results/w03_broken_student"


def damage(model, kind: str):
    with torch.no_grad():
        if kind == "permuted_pointer_head":
            model.pointers.weight.copy_(model.pointers.weight[torch.tensor([2, 3, 0, 1])])
        elif kind == "sign_flipped_block":
            for name in quant.TRUNK_LINEARS:
                getattr(model.blocks[0], name).weight.mul_(-1)
        elif kind == "random_last_block":
            for name in quant.TRUNK_LINEARS:
                w = getattr(model.blocks[-1], name).weight
                w.copy_(torch.randn_like(w) * w.std())
        else:
            raise ValueError(kind)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cal", type=int, default=20_000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    src = REPO / args.model
    results = {"model": args.model, "cal": args.cal, "cases": {}}
    for kind in MODEL_DAMAGE + PACK_DAMAGE:
        work = REPO / "build" / f"broken_{kind}"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        model, ck = load_student(src)
        model.eval()
        if kind in MODEL_DAMAGE:
            damage(model, kind)
        pack_mod.pack, saved_pack = (corrupt_packed(kind) if kind in PACK_DAMAGE else pack_mod.pack), pack_mod.pack
        torch.save({"config": ck["config"], "bits": ck.get("bits"), "state": {k: v.cpu() for k, v in model.state_dict().items()}}, work / "student.pt")
        shutil.copy(src / "train_report.json", work / "train_report.json")
        try:
            rep = build_mod.main(["--spec", "specs/telemetry.spec", "--model", str(work.relative_to(REPO)),
                                  "--out", str((REPO / "build" / f"broken_out_{kind}").relative_to(REPO)), "--cal", str(args.cal)])
            binary = REPO / "build" / f"broken_out_{kind}" / "ss_hybrid"
            results["cases"][kind] = {"build_failed": False, "binary_exists": binary.exists(),
                                      "certificate": rep["certificate"], "VERDICT": "PRODUCED A BINARY"}
        except build_mod.BuildFailed as e:
            binary = REPO / "build" / f"broken_out_{kind}" / "ss_hybrid"
            results["cases"][kind] = {"build_failed": True, "binary_exists": binary.exists(), "reason": str(e)[:300],
                                      "VERDICT": "refused"}
        finally:
            pack_mod.pack = saved_pack
        print(f"{kind:<26} {results['cases'][kind]['VERDICT']:<18} binary_exists={results['cases'][kind]['binary_exists']}", flush=True)
    results["G1c_build_fails_on_a_broken_student"] = all(
        c["build_failed"] and not c["binary_exists"] for c in results["cases"].values())
    (OUT / "summary.json").write_text(json.dumps(results, indent=1, default=float) + "\n")
    print("G1c broken-student clause:", "PASS" if results["G1c_build_fails_on_a_broken_student"] else "FAIL")


if __name__ == "__main__":
    main()
