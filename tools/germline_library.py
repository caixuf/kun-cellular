"""Thread-safe read-only observatory for native R8 SQLite schema 3.

Python does not admit, execute, validate, borrow, or adopt native knowledge.
Each snapshot opens a read-only connection and a consistent read transaction.
"""
from __future__ import annotations

import hashlib
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
from typing import Any


class LibraryError(ValueError):
    """The native library exists but cannot be read correctly."""


class GermlineLibrary:
    SCHEMA_VERSION = 3
    APPLICATION_ID = 1263881803

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path).absolute()

    def snapshot(self) -> dict[str, Any]:
        result = {
            "schema_version": self.SCHEMA_VERSION,
            "support": "native-r8-read-only",
            "available": False,
            "entries": [],
            "audit_log": [],
        }
        if not self.path.exists():
            return result
        try:
            # Never mode=rwc, CREATE TABLE, mkdir, or an instance/thread-bound connection.
            with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)) as db:
                db.row_factory = sqlite3.Row
                db.execute("BEGIN")
                version = db.execute("PRAGMA user_version").fetchone()[0]
                app = db.execute("PRAGMA application_id").fetchone()[0]
                if version != self.SCHEMA_VERSION or app != self.APPLICATION_ID:
                    raise LibraryError(f"unsupported native R8 schema/application: {version}/{app}")
                entries = db.execute(
                    "SELECT k.*,s.status,s.borrows,s.failures,s.reason "
                    "FROM knowledge_objects k JOIN object_status s USING(object_id,version) "
                    "ORDER BY object_id,version"
                ).fetchall()
                for row in entries:
                    artifact = bytes(row["artifact"])
                    if hashlib.sha256(artifact).hexdigest() != row["content_digest"]:
                        raise LibraryError("native knowledge artifact digest mismatch")
                    key = (row["object_id"], row["version"])
                    evidence = []
                    for e in db.execute(
                        "SELECT * FROM evidence WHERE object_id=? AND version=? ORDER BY sequence", key
                    ):
                        if (hashlib.sha256(bytes(e["report"])).hexdigest() != e["report_digest"]
                                or e["content_digest"] != row["content_digest"]
                                or e["environment"] != row["environment"]):
                            raise LibraryError("native evidence/content/environment digest mismatch")
                        evidence.append({
                            "sequence": e["sequence"], "tier": "research-trace",
                            "passed": bool(e["passed"]), "protocol": e["protocol"],
                            "manifest_digest": e["manifest_digest"],
                            "report": "sha256:" + e["report_digest"],
                            "graph_executions": e["graph_executions"],
                            "cell_visits": e["cell_visits"], "edge_visits": e["edge_visits"],
                            "elapsed_ns": e["elapsed_ns"],
                        })
                    if row["status"] == "research-validated" and (
                            not evidence or not evidence[-1]["passed"]):
                        raise LibraryError("research eligibility without passing native evidence")
                    parents = [
                        {"entry_id": p[0], "version": p[1]}
                        for p in db.execute(
                            "SELECT parent_id,parent_version FROM parents WHERE object_id=? "
                            "AND version=? ORDER BY parent_id,parent_version", key
                        )
                    ]
                    result["entries"].append({
                        "entry_id": row["object_id"], "version": row["version"],
                        "title": row["title"], "status": row["status"], "origin": row["origin"],
                        "artifact_size": len(artifact), "native_encoding": "KUN-NATIVE-KNOWLEDGE/v1",
                        "content_digest": row["content_digest"],
                        "environment_contract": row["environment"], "interface_id": row["interface_id"],
                        "source_lineage": row["producer_lineage"], "parents": parents,
                        "borrows": row["borrows"], "failures": row["failures"],
                        "retirement_reason": row["reason"], "evidence": evidence,
                    })
                result["audit_log"] = [
                    {"sequence": row["sequence"], "event": row["event"],
                     "entry_id": row["object_id"], "version": row["version"],
                     "actor_id": row["actor"], "detail": row["detail"]}
                    for row in db.execute("SELECT * FROM events ORDER BY sequence")
                ]
                result["available"] = True
                db.commit()
            return result
        except (sqlite3.Error, OSError, KeyError, TypeError, ValueError) as error:
            if isinstance(error, LibraryError):
                raise
            raise LibraryError(f"native R8 library read failed: {error}") from error

    def entries(self) -> list[dict[str, Any]]:
        return self.snapshot()["entries"]

    @property
    def audit_log(self) -> list[dict[str, Any]]:
        return self.snapshot()["audit_log"]

    def close(self) -> None:
        """No connection is retained between calls."""

    def export_snapshot(self, path: str | os.PathLike[str]) -> None:
        destination = Path(path)
        if destination.suffix.lower() != ".json":
            raise LibraryError("snapshot export requires a .json path")
        destination.write_text(json.dumps(self.snapshot(), indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")

    def __getattr__(self, name: str) -> Any:
        if name in {"admit", "publish", "borrow", "adopt", "retire", "evaluate"}:
            raise LibraryError(f"{name} is C++-owned; Python is a read-only observatory")
        raise AttributeError(name)
