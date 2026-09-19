import io
import pickle

import pytest

from static_student import datasets


def test_unpickler_refuses_anything_outside_the_allow_list():
    payload = pickle.dumps({"a": [1, 2]})
    assert datasets._AllowListUnpickler(io.BytesIO(payload)).load() == {"a": [1, 2]}
    import os
    evil = pickle.dumps(os.getcwd)  # a global reference: loading it would import os
    with pytest.raises(pickle.UnpicklingError):
        datasets._AllowListUnpickler(io.BytesIO(evil)).load()


@pytest.mark.skipif(not (datasets.REPO / "data/raw/logevol/Logevol").exists(), reason="LOGEVOL not downloaded")
def test_logevol_loads_with_the_documented_shape():
    d = datasets.load_logevol("spark2", "valid")
    s = next(iter(d.values()))
    assert {"label", "templates", "Content"} <= set(s) and len(s["templates"]) == len(s["Content"])
