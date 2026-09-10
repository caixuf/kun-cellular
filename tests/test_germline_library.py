"""Read-only projection tests against the native executable, not fabricated SQL fixtures."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import uuid

import pytest

from tools.germline_library import GermlineLibrary, LibraryError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def db_path():
    directory = ROOT / "build-r8" / ("python-" + uuid.uuid4().hex)
    directory.mkdir(parents=True)
    path = directory / "native.db"
    yield path
    shutil.rmtree(directory)


def resolve_knowledge_demo():
    candidates = []
    env = os.environ.get("KUN_KNOWLEDGE_DEMO")
    if env:
        candidates.append(Path(env))
    candidates.extend([
        ROOT / "build" / "knowledge_transfer_demo",
        ROOT / "build-r8" / "knowledge_transfer_demo",
    ])
    return next((p for p in candidates if p.is_file()), None)


def run_native(path):
    binary = resolve_knowledge_demo()
    assert binary is not None, (
        "Build knowledge_transfer_demo before running R8 integration tests "
        f"(looked under build/ and build-r8/, or set KUN_KNOWLEDGE_DEMO)"
    )
    result = subprocess.run([str(binary), "demo", str(path)], check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def test_missing_library_never_creates_files(db_path):
    missing = db_path.parent / "missing" / "library.db"
    adapter = GermlineLibrary(missing)
    assert adapter.snapshot() == {
        "schema_version": 3, "support": "native-r8-read-only", "available": False,
        "entries": [], "audit_log": [],
    }
    assert not missing.parent.exists()


def test_native_abc_roundtrip_and_thread_safe_refresh(db_path):
    report = run_native(db_path)
    assert report["A"]["selected_gain"] == 2
    assert report["B"]["before"] == -2 and report["B"]["after"] == -4
    assert report["B"]["improved"] == -6 and report["C"]["after"] == -6
    assert report["B"]["germline_unchanged"] and report["C"]["germline_unchanged"]
    assert report["B"]["growth_paid"] == report["C"]["growth_paid"] == 2.5
    assert report["controls"]["no_book"]["selected_gain"] == 3
    assert report["controls"]["wrong_book"]["success"] is False
    assert report["total"]["graph_executions"] > report["B"]["training_executions"]
    assert report["total"]["failed_attempts"] > 0
    adapter = GermlineLibrary(db_path)
    before = hashlib.sha256(db_path.read_bytes()).hexdigest()
    snapshot = adapter.snapshot()
    assert snapshot["available"] is True and len(snapshot["entries"]) >= 3
    assert all(e["content_digest"] for e in snapshot["entries"])
    v2 = next(e for e in snapshot["entries"] if e["entry_id"] == "scale" and e["version"] == "2")
    assert v2["origin"] == "phenotype-cultural"
    assert v2["parents"] == [{"entry_id": "scale", "version": "1"}]
    assert all(e["tier"] == "research-trace" for e in v2["evidence"])
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: adapter.snapshot(), range(24)))
    assert all(result == snapshot for result in results)
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before
    binary = resolve_knowledge_demo()
    assert binary is not None
    reopened = json.loads(subprocess.check_output([str(binary), "inspect", str(db_path)], text=True))
    assert reopened["fresh_output"] == -6
    assert adapter.snapshot()["audit_log"][-1]["event"] == "borrow"
    with pytest.raises(LibraryError, match="C\\+\\+-owned"):
        adapter.borrow()
    adapter.close()


def test_unknown_schema_and_corruption_are_errors(db_path):
    db_path.write_bytes(b"not sqlite")
    with pytest.raises(LibraryError):
        GermlineLibrary(db_path).snapshot()
    db_path.unlink()
    run_native(db_path)
    with sqlite3.connect(db_path) as db:
        db.execute("PRAGMA user_version=999")
    before = db_path.read_bytes()
    with pytest.raises(LibraryError, match="schema"):
        GermlineLibrary(db_path).snapshot()
    assert db_path.read_bytes() == before
