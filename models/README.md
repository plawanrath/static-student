# models/

The trained checkpoints are not stored in this repository. All 25 are on the Hugging Face Hub, Apache-2.0, one repo
per directory name, in the collection **plawanrath/static-student**:

https://huggingface.co/collections/plawanrath/static-student-weights-as-constants-checkpoints-6ac5c510784a04a1bb253d0a

| directory here | Hub repo |
|---|---|
| `tel-mlx-{XS,S,M}-fp` | `plawanrath/static-student-tel-mlx-<size>-fp` (float byte-level students, teacher curriculum) |
| `tel-mlx-{XS,S,M}-w{4,3,2}` | `plawanrath/static-student-tel-mlx-<size>-w<bits>` (QAT ladder; the weights the certified binaries carry) |
| `tel-stub*` | `plawanrath/static-student-tel-stub*` (superseded stub-curriculum pilots) |
| `modernbert-{base,large}-civil` | `plawanrath/static-student-modernbert-<size>-civil` (ModernBERT toxicity classifiers) |

Download into this directory so the RUNBOOK commands find them:

```bash
.venv/bin/python scripts/release/download_hf.py tel-mlx-S-w4 tel-mlx-S-fp   # named models
.venv/bin/python scripts/release/download_hf.py --all                      # everything (2.5 GB)
```

Each student repo holds `model.safetensors` + `config.json` (`static_student.student.release.load_released`), the
original `student.pt` / `student_fp.pt` the training and build commands expect, `train_report.json` and `LICENSE`.
The ModernBERT repos load with `transformers`. See the model cards for training details, certificates and citation.
