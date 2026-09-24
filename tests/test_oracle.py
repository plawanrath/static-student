"""The integer reference: shapes, the padding invariant that ADR-0009 buys, and agreement with the float model."""
import numpy as np
import pytest
import torch

from static_student.codegen import oracle, pack
from static_student.student import data, quant
from static_student.student.model import PAD, SIZES, Student

FAM = {"family_id": "f0", "container": "kv", "sep": " ", "assign": "=", "envelope": "<iso_ts> ", "defer_reason": None, "slots": [
    {"role": "latency", "key": "lat", "value": {"gen": "duration", "unit": "ms", "numfmt": "decimal"}},
    {"role": "user", "key": "user", "value": {"gen": "id", "scheme": "uuid"}}]}


@pytest.fixture(scope="module")
def fixture():
    torch.manual_seed(0)
    m = quant.to_qat(Student(SIZES["XS"]), 4)
    for q in m.modules():
        if isinstance(q, quant.QLinear):
            q.snap_scales()
    m.eval()
    ids = data.encode(data.render_rows([FAM], 6, "oracle-test"))["ids"]
    return m, pack.pack(m), ids


def test_packed_layout_round_trips_and_reports_bytes(fixture):
    m, p, _ = fixture
    assert p.bits == 4 and len(p.blocks) == SIZES["XS"].layers
    b = p.bytes()
    assert b["total"] == b["trunk"] + b["embeddings_and_heads"] + b["norms"] and 150_000 < b["total"] < 350_000
    q = m.blocks[0].q
    levels, scales = q.integer_weights()
    assert np.array_equal(p.blocks[0]["q"].levels, levels.numpy().astype(np.int32))
    assert np.array_equal(p.blocks[0]["q"].scales, scales.numpy().astype(np.float32))
    assert p.sha256() == pack.pack(m).sha256() and len(p.manifest()["sha256"]) == 64


def test_shapes_and_padding_can_never_be_pointed_at(fixture):
    _, p, ids = fixture
    out = oracle.forward(p, ids)
    assert out["pointers"].shape == (6, 4, ids.shape[1]) and out["unit"].shape == (6, 5) and out["defer"].shape == (6,)
    n = int((ids[0] != PAD).sum())
    assert out["pointers"][0, :, n:].max() <= -1e4


def test_trailing_padding_changes_nothing(fixture):
    """ADR-0009's payoff: the kernel may stop at the end of the payload. Checked bit for bit, not approximately."""
    _, p, ids = fixture
    full = oracle.forward(p, ids, trim=False)
    trimmed = oracle.forward(p, ids, trim=True)
    assert all(np.array_equal(full[k], trimmed[k]) for k in full)
    one = ids[:1]
    used = int((one != PAD).sum())
    short = oracle.forward(p, one[:, :used + 3])
    assert np.array_equal(short["defer"], oracle.forward(p, one)["defer"])
    assert np.array_equal(short["unit"], oracle.forward(p, one)["unit"])


def test_reference_agrees_with_the_float_model_on_every_decision(fixture):
    m, p, ids = fixture
    with torch.no_grad():
        t = m(torch.from_numpy(ids))
    d = oracle.decisions(oracle.forward(p, ids))
    assert np.array_equal(d["ptr"], t["pointers"].argmax(-1).numpy())
    assert np.array_equal(d["unit"], t["unit"].argmax(-1).numpy())


def test_decisions_reproduce_the_struct(fixture):
    _, p, ids = fixture
    rows = data.render_rows([FAM], 6, "oracle-test")
    d = oracle.decisions(oracle.forward(p, ids))
    assert d["maxprob"].shape == (6,) and (d["maxprob"] <= 1e-6).all()
    enc = data.encode(rows)
    for i, r in enumerate(rows):  # gold spans must decode through the same path the kernel uses
        assert data.decode(r.payload, enc["ptr"][i], int(enc["unit"][i])) == (r.latency_us, r.user_id)
