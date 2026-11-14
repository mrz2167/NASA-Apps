"""Общая таблица находок (citizen science): отчёты учеников -> таблица для преподавателя.

    POST   /api/reports              отправить отчёт (из окна «Field report» на сайте)
    GET    /api/reports?class=CODE   отчёты класса (код класса — общий секрет класса)
    GET    /api/reports?key=KEY      все отчёты (ключ преподавателя, переменная INSTRUCTOR_KEY)
    GET    /api/reports.csv?…        то же в CSV (для Excel / Google Sheets)
    DELETE /api/reports/{id}?key=KEY удалить (только с ключом преподавателя)
    страница /reports                таблица для преподавателя

Хранилище — SQLite, путь REPORTS_DB (по умолчанию data/reports.db). На Railway файловая система
сбрасывается при каждом деплое: подключите volume и укажите REPORTS_DB=/data/reports.db.

Имена учеников видны только по коду класса или ключу преподавателя — публичного списка нет.
"""
from __future__ import annotations

import csv
import hmac
import io
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("REPORTS_DB", ROOT / "data" / "reports.db"))
INSTRUCTOR_KEY = os.environ.get("INSTRUCTOR_KEY", "")
MOLECULES = ("water", "methane", "co2", "co")
SAME_OBJECT_DEG = 3 / 3600          # тот же ученик, тот же объект ближе 3″ — не дублируем
MAX_PER_CLIENT_PER_HOUR = 60

router = APIRouter()
_lock = threading.Lock()


def _db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created REAL NOT NULL,
        client TEXT NOT NULL,
        name TEXT, class_code TEXT,
        source TEXT NOT NULL,              -- free | level5
        ra REAL, dec REAL, field_ra REAL, field_dec REAL, fov REAL,
        object_name TEXT, status TEXT, verdict TEXT,
        motion REAL, motion_err REAL, pa REAL, catalog_motion REAL,
        variability REAL, molecules TEXT,
        detector INTEGER, cadence TEXT, note TEXT)""")
    con.execute("CREATE INDEX IF NOT EXISTS reports_class ON reports(class_code)")
    return con


def _clean_code(s: str | None) -> str:
    return "".join(ch for ch in (s or "").strip().upper() if ch.isalnum() or ch in "-_")[:24]


class ReportIn(BaseModel):
    client: str = Field(min_length=6, max_length=64)
    name: str = Field("", max_length=40)
    class_code: str = Field("", max_length=24)
    source: str = Field(pattern="^(free|level5)$")
    ra: float | None = Field(None, ge=0, lt=360)
    dec: float | None = Field(None, ge=-90, le=90)
    field_ra: float | None = Field(None, ge=0, lt=360)
    field_dec: float | None = Field(None, ge=-90, le=90)
    fov: float | None = Field(None, gt=0, le=1)
    object_name: str | None = Field(None, max_length=80)
    status: str = Field("candidate", pattern="^(known|candidate|star)$")
    verdict: str | None = Field(None, max_length=120)
    motion: float | None = None
    motion_err: float | None = None
    pa: float | None = None
    catalog_motion: float | None = None
    variability: float | None = None
    molecules: dict[str, float] = {}
    detector: int | None = Field(None, ge=1, le=6)
    cadence: str | None = Field(None, pattern="^(pass|visit)$")
    note: str = Field("", max_length=300)

    @field_validator("molecules")
    @classmethod
    def _mols(cls, v):
        return {k: round(float(min(max(v.get(k) or 0.0, 0.0), 1.0)), 3) for k in MOLECULES}


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["molecules"] = json.loads(d["molecules"] or "{}")
    d["created_iso"] = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(d["created"]))
    d.pop("client", None)
    return d


def _is_instructor(key: str | None) -> bool:
    return bool(INSTRUCTOR_KEY) and bool(key) and hmac.compare_digest(key, INSTRUCTOR_KEY)


@router.post("/api/reports")
def create_report(rep: ReportIn):
    code = _clean_code(rep.class_code)
    now = time.time()
    with _lock, _db() as con:
        n = con.execute("SELECT COUNT(*) FROM reports WHERE client=? AND created>?", (rep.client, now - 3600)).fetchone()[0]
        if n >= MAX_PER_CLIENT_PER_HOUR:
            raise HTTPException(429, "Too many reports from this browser in the last hour.")
        if rep.ra is not None and rep.dec is not None:
            dup = con.execute(
                "SELECT id FROM reports WHERE client=? AND source=? AND ABS(ra-?)<? AND ABS(dec-?)<?",
                (rep.client, rep.source, rep.ra, SAME_OBJECT_DEG * 2, rep.dec, SAME_OBJECT_DEG)).fetchone()
            if dup:   # повторная отправка той же находки: обновляем имя/класс/заметку, номер тот же
                con.execute("UPDATE reports SET name=?, class_code=?, note=? WHERE id=?",
                            (rep.name.strip(), code, rep.note.strip(), dup[0]))
                return {"id": dup[0], "duplicate": True, "class_code": code}
        cur = con.execute(
            """INSERT INTO reports (created, client, name, class_code, source, ra, dec, field_ra, field_dec, fov,
               object_name, status, verdict, motion, motion_err, pa, catalog_motion, variability, molecules,
               detector, cadence, note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (now, rep.client, rep.name.strip(), code, rep.source, rep.ra, rep.dec, rep.field_ra, rep.field_dec,
             rep.fov, rep.object_name, rep.status, rep.verdict, rep.motion, rep.motion_err, rep.pa,
             rep.catalog_motion, rep.variability, json.dumps(rep.molecules), rep.detector, rep.cadence,
             rep.note.strip()))
        return {"id": cur.lastrowid, "duplicate": False, "class_code": code}


def _select(class_code: str | None, key: str | None) -> list[dict]:
    code = _clean_code(class_code)
    if _is_instructor(key):
        q, args = ("SELECT * FROM reports WHERE class_code=? ORDER BY created DESC", (code,)) if code \
            else ("SELECT * FROM reports ORDER BY created DESC", ())
    elif code:
        q, args = "SELECT * FROM reports WHERE class_code=? ORDER BY created DESC", (code,)
    else:
        raise HTTPException(401, "Enter a class code (or the instructor key) to see reports.")
    with _db() as con:
        return [_row(r) for r in con.execute(q + " LIMIT 5000", args)]


@router.get("/api/reports")
def list_reports(class_code: str | None = Query(None, alias="class"), key: str | None = None):
    rows = _select(class_code, key)
    return {"reports": rows, "count": len(rows), "instructor": _is_instructor(key),
            "persistent_hint": None if os.environ.get("REPORTS_DB") else
            "Reports are stored in data/reports.db inside the app. On Railway, mount a volume and set REPORTS_DB."}


@router.get("/api/reports.csv")
def reports_csv(request: Request, class_code: str | None = Query(None, alias="class"), key: str | None = None):
    rows = _select(class_code, key)
    base = str(request.base_url).rstrip("/")
    cols = ["id", "created_iso", "name", "class_code", "source", "object_name", "status", "verdict", "ra", "dec",
            "motion", "motion_err", "pa", "catalog_motion", "variability", *MOLECULES, "detector", "note", "sky_link"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        r.update(r.pop("molecules"))
        if r.get("field_ra") is not None:
            r["sky_link"] = f"{base}/?ra={r['field_ra']:.5f}&dec={r['field_dec']:.5f}&fov={r.get('fov') or 0.1}"
        # текст учеников не должен стать формулой в Excel/Sheets (=HYPERLINK(…) и т.п.)
        for k in ("name", "class_code", "object_name", "verdict", "note"):
            v = r.get(k)
            if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r"):
                r[k] = "'" + v
        w.writerow(r)
    name = f"lumen-reports-{_clean_code(class_code) or 'all'}.csv"
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@router.delete("/api/reports/{rid}")
def delete_report(rid: int, key: str | None = None):
    if not _is_instructor(key):
        raise HTTPException(403, "Only the instructor key can delete reports.")
    with _lock, _db() as con:
        n = con.execute("DELETE FROM reports WHERE id=?", (rid,)).rowcount
    if not n:
        raise HTTPException(404, "No such report.")
    return {"deleted": rid}


@router.get("/reports", include_in_schema=False)
def reports_page():
    return FileResponse(ROOT / "web" / "reports.html")
