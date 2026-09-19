"""Held-out drift must stay invisible to everything that produces training data."""
import re
import subprocess
import sys
from pathlib import Path

from static_student.curriculum import build, teacher
from static_student.curriculum.spec import parse_spec

REPO = Path(__file__).resolve().parent.parent
VISIBLE = [*(REPO / "static_student/curriculum").rglob("*.py"), *(REPO / "static_student/tasks").rglob("*.py"),
           REPO / "static_student/drift/operators.py", REPO / "static_student/drift/__init__.py"]


def _tokens() -> set[str]:
    out = subprocess.run([sys.executable, "-m", "static_student.drift.heldout"], check=True, capture_output=True, text=True, cwd=REPO).stdout
    return set(out.split())


def test_importing_the_curriculum_never_loads_the_heldout_module():
    code = ("import sys, static_student.curriculum.build, static_student.curriculum.teacher, static_student.tasks.telemetry;"
            "sys.exit(int('static_student.drift.heldout' in sys.modules))")
    assert subprocess.run([sys.executable, "-c", code], cwd=REPO).returncode == 0


def test_no_visible_source_file_names_the_heldout_module_or_its_tokens():
    toks = _tokens()
    assert len(toks) > 30
    for path in VISIBLE:
        src = path.read_text()
        assert not re.search(r"^\s*(from|import)\s+.*heldout", src, re.M), path
        literals = {w.lower() for s in re.findall(r"\"([^\"\n]*)\"|'([^'\n]*)'", src) for part in s for w in re.split(r"[^A-Za-z0-9]+", part) if w}
        assert not literals & toks, (path.name, sorted(literals & toks))


def test_task_card_and_accepted_families_are_free_of_heldout_tokens():
    toks = _tokens()
    t = teacher.StubTeacher()
    card = t.task_card(parse_spec(REPO / "specs/telemetry.spec"))
    families, stats = build.propose_families(t, card, 60, seed=1, reject=toks)
    assert len(families) == 60
    assert all(not build.family_tokens(f) & toks for f in families)
    card_words = {w.lower() for w in re.split(r"[^A-Za-z0-9]+", str(card)) if w}
    assert not card_words & toks


def test_a_family_using_a_heldout_token_is_rejected():
    fam = {"bucket": "clean", "container": "kv", "sep": " ", "assign": "=", "defer_reason": None, "slots": [
        {"role": "latency", "key": "turnaround", "value": {"gen": "duration", "unit": "ms"}},
        {"role": "user", "key": "user", "value": {"gen": "id"}}]}
    assert build.validate(fam, "clean", set(), _tokens(), seed=0) == "held_out_token"
    assert build.validate(fam, "clean", set(), set(), seed=0) is None
