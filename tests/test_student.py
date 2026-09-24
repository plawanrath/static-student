import random

import numpy as np
import torch

from static_student.student import data
from static_student.student.model import CLS, MAX_LEN, PAD, SIZES, Student
from static_student.tasks import telemetry as T

FAM = {"family_id": "f0", "container": "kv", "sep": " ", "assign": "=", "envelope": "<iso_ts> ", "defer_reason": None, "slots": [
    {"role": "latency", "key": "lat", "value": {"gen": "duration", "unit": "ms", "numfmt": "decimal"}},
    {"role": "user", "key": "user", "value": {"gen": "id", "scheme": "uuid"}}]}


def test_gold_pointers_decode_back_to_the_gold_struct():
    rows = data.render_rows([FAM], 200, "t")
    enc = data.encode(rows)
    assert enc["ids"].shape == (200, MAX_LEN) and (enc["ids"][:, 0] == CLS).all()
    for i, r in enumerate(rows):
        assert data.decode(r.payload, enc["ptr"][i], int(enc["unit"][i])) == (r.latency_us, r.user_id)
        assert (enc["ids"][i, 1 + len(r.payload):] == PAD).all()


def test_decode_refuses_absent_inverted_and_malformed_spans():
    r = data.render_rows([FAM], 1, "t")[0]
    ptr = data.encode([r])["ptr"][0]
    assert data.decode(r.payload, np.array([0, 0, 0, 0]), 2) is None
    assert data.decode(r.payload, ptr[[1, 0, 2, 3]], 2) is None
    assert data.decode(r.payload, np.array([1, 5, ptr[2], ptr[3]]), 2) is None  # "2026-" is not a number
    assert data.decode(r.payload, np.array([1, 4, ptr[2], ptr[3]]), 2) is not None  # "2026" is: a wrong span can still be well formed
    defer_row = T.render({**FAM, "defer_reason": "non_numeric", "slots": [{**FAM["slots"][0], "value": {"gen": "duration", "unit": "ms", "numfmt": "nonnumeric"}}, FAM["slots"][1]]}, random.Random(0))
    e = data.encode([defer_row])
    assert e["defer"][0] == 1 and (e["ptr"][0] == 0).all() and e["unit"][0] == -100


def test_split_is_by_family_and_stable():
    fams = [{**FAM, "family_id": f"f{i:05d}"} for i in range(500)]
    a, b = data.split_families(fams), data.split_families(fams[:250])
    ids = {k: {f["family_id"] for f in v} for k, v in a.items()}
    assert not ids["train"] & ids["dev"] and not ids["train"] & ids["defer_fit"] and 350 < len(ids["train"]) < 450
    assert {f["family_id"] for f in b["dev"]} <= ids["dev"]


def test_model_shapes_sizes_and_padding_invariance():
    m = Student(SIZES["XS"]).eval()
    assert 0.3e6 < m.n_params()["trunk"] < 0.4e6
    enc = data.encode(data.render_rows([FAM], 4, "t"))
    o = m(torch.from_numpy(enc["ids"]))
    assert o["pointers"].shape == (4, 4, MAX_LEN) and o["unit"].shape == (4, 5) and o["defer"].shape == (4,)
    n = int((enc["ids"][0] != PAD).sum())
    assert o["pointers"][0, :, n:].max() < -1e3  # padding can never be selected
