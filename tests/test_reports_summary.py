# AI disclosure: parts of this file were written with AI assistance (Claude, Anthropic), then reviewed and tested by the LUMEN team. See README, "AI use".
"""Teacher view of the reports: class codes, the per-class summary and its CSV filter (no network, temp database)."""
from __future__ import annotations

import csv
import io
import sqlite3

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import reports

KEY = "test-key"


class Client:
    """Calls the route functions directly (starlette's TestClient needs an extra package CI doesn't install)."""

    def post(self, _path, json):
        reports.create_report(reports.ReportIn(**json))
        return SimpleNamespace(status_code=200)

    def get(self, path, params=None):
        p = dict(params or {})
        try:
            if path == "/api/reports/summary":
                body = reports.reports_summary(**p)
                return SimpleNamespace(status_code=200, json=lambda: body)
            resp = reports.reports_csv(SimpleNamespace(base_url="http://test/"), **p)
            return SimpleNamespace(status_code=200, text=resp.body.decode())
        except HTTPException as e:
            return SimpleNamespace(status_code=e.status_code)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(reports, "DB_PATH", tmp_path / "reports.db")
    monkeypatch.setattr(reports, "REPORTS_KEY", KEY)
    return Client()


def _rep(client_id, name, ra, obj, status="known", cls=None):
    d = dict(client=client_id, name=name, ra=ra, dec=-10.0, field_ra=ra, field_dec=-10.0, fov=0.1,
             object_name=obj, status=status, molecules={})
    if cls is not None:
        d["class_code"] = cls
    return d


def _seed(c):
    for body in (_rep("browser-aaa1", "Aziz", 10.0, "Barnard's Star", cls="7b"),
                 _rep("browser-aaa1", "Aziz", 11.0, None, status="candidate", cls="7B"),
                 _rep("browser-bbb2", "Malika", 10.00001, "Barnard's Star", cls="7B"),
                 _rep("browser-ccc3", "", 12.0, "WISE 0855-0714", cls="8A"),
                 _rep("browser-ddd4", "Old", 13.0, "Luhman 16")):          # sent before class codes existed
        assert c.post("/api/reports", json=body).status_code == 200


def test_summary_needs_key(client):
    assert client.get("/api/reports/summary").status_code == 401
    assert client.get("/api/reports/summary", params={"key": "nope"}).status_code == 401


def test_summary_counts_students_objects_and_classes(client):
    _seed(client)
    s = client.get("/api/reports/summary", params={"key": KEY}).json()
    assert s["totals"] == {"reports": 5, "students": 4, "objects": 4, "candidates": 1}
    aziz = next(x for x in s["students"] if x["name"] == "Aziz")
    assert aziz["class_code"] == "7B" and aziz["reports"] == 2 and aziz["candidates"] == 1
    assert "Barnard's Star" in aziz["objects"] and any(o.startswith("Unidentified at") for o in aziz["objects"])
    assert s["students"][0]["name"] == "Aziz"                       # most reports first
    barnard = next(o for o in s["objects"] if o["object_name"] == "Barnard's Star")
    assert barnard["reports"] == 2 and sorted(barnard["students"]) == ["Aziz", "Malika"]
    assert any(x["name"] == "Anonymous" for x in s["students"])
    assert {c["class_code"]: c["reports"] for c in s["classes"]} == {None: 1, "7B": 3, "8A": 1}
    assert sum(d["reports"] for d in s["days"]) == 5 and all(len(d["day"]) == 10 for d in s["days"])


def test_summary_filters_by_class_code_case_insensitively(client):
    _seed(client)
    s = client.get("/api/reports/summary", params={"key": KEY, "class_code": " 7b "}).json()
    assert s["class_code"] == "7B"
    assert s["totals"]["reports"] == 3 and {x["name"] for x in s["students"]} == {"Aziz", "Malika"}
    assert client.get("/api/reports/summary", params={"key": KEY, "class_code": "9Z"}).json()["totals"]["reports"] == 0


def test_csv_has_class_column_and_filter(client):
    _seed(client)
    rows = list(csv.DictReader(io.StringIO(client.get("/api/reports.csv", params={"key": KEY}).text)))
    assert len(rows) == 5 and "class_code" in rows[0]
    only = list(csv.DictReader(io.StringIO(
        client.get("/api/reports.csv", params={"key": KEY, "class_code": "8a"}).text)))
    assert [r["class_code"] for r in only] == ["8A"]


def test_bad_class_code_is_rejected(client):
    with pytest.raises(ValidationError):
        client.post("/api/reports", json=_rep("browser-eee5", "X", 1.0, "Y", cls="<script>"))


def test_old_database_gets_class_column(tmp_path, monkeypatch):
    """The Railway volume already has a finds table without class_code; opening it must add the column."""
    db = tmp_path / "old.db"
    with sqlite3.connect(db) as con:
        con.execute("""CREATE TABLE finds (id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL NOT NULL,
            client TEXT NOT NULL, name TEXT, ra REAL, dec REAL, field_ra REAL, field_dec REAL, fov REAL,
            object_name TEXT, status TEXT, verdict TEXT, motion REAL, motion_err REAL, pa REAL,
            catalog_motion REAL, variability REAL, molecules TEXT, detector INTEGER, cadence TEXT)""")
        con.execute("INSERT INTO finds (created, client, name, ra, dec, object_name, status, molecules) "
                    "VALUES (1.7e9, 'old-client', 'Old', 1, 2, 'Vega', 'known', '{}')")
    monkeypatch.setattr(reports, "DB_PATH", db)
    rows = reports._all()
    assert rows[0]["object_name"] == "Vega" and rows[0]["class_code"] is None
    assert reports.summarize(rows)["totals"]["reports"] == 1
