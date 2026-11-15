"""Находки пользователей: кнопка «Send report» в окне Field report -> запись на нашем сервере.

    POST   /api/reports              принять отчёт (свободный режим, реальные данные SPHEREx)
    GET    /api/reports?key=KEY      все отчёты (ключ команды, переменная REPORTS_KEY)
    GET    /api/reports.csv?key=KEY  то же в CSV (Excel / Google Sheets)
    GET    /api/reports/summary?key=KEY[&class_code=X]  сводка для учителя: по ученикам, объектам, дням, классам
    DELETE /api/reports/{id}?key=KEY удалить мусорный отчёт
    страница /reports                таблица для команды (нужен ключ)

Хранилище — SQLite, путь REPORTS_DB (по умолчанию cache/reports.db: на Railway это постоянный диск).
Резервные копии — backup.py (/api/admin/backup/…), скачивает их scripts/backup.py.
Список отчётов закрыт ключом: в отчётах могут быть имена пользователей.
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
from contextlib import closing
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parent
# по умолчанию — в cache/: на Railway это постоянный диск (volume в /app/cache), деплой его не стирает
DB_PATH = Path(os.environ.get("REPORTS_DB", ROOT / "cache" / "reports.db"))
REPORTS_KEY = os.environ.get("REPORTS_KEY", "")
MOLECULES = ("water", "methane", "co2", "co")
SAME_OBJECT_DEG = 3 / 3600          # тот же браузер, тот же объект ближе 3″ — не дублируем
MAX_PER_CLIENT_PER_HOUR = 60

router = APIRouter()
_lock = threading.Lock()


def _db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS finds (   -- not "reports": that name had a class-code schema
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created REAL NOT NULL,
        client TEXT NOT NULL,
        name TEXT,
        ra REAL, dec REAL, field_ra REAL, field_dec REAL, fov REAL,
        object_name TEXT, status TEXT, verdict TEXT,
        motion REAL, motion_err REAL, pa REAL, catalog_motion REAL,
        variability REAL, molecules TEXT,
        detector INTEGER, cadence TEXT)""")
    cols = {r[1] for r in con.execute("PRAGMA table_info(finds)")}
    if "class_code" not in cols:                 # older databases on the Railway volume
        con.execute("ALTER TABLE finds ADD COLUMN class_code TEXT")
    return con


def _norm_class(v: str | None) -> str | None:
    v = (v or "").strip().upper()
    return v or None


class ReportIn(BaseModel):
    client: str = Field(min_length=6, max_length=64)
    name: str = Field("", max_length=40)
    ra: float = Field(ge=0, lt=360)
    dec: float = Field(ge=-90, le=90)
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
    class_code: str | None = Field(None, max_length=20, pattern=r"^[A-Za-z0-9_\- ]*$")   # ?class=7B on the site link

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


def _require_key(key: str | None) -> None:
    if not REPORTS_KEY:
        raise HTTPException(403, "Viewing reports is off: set REPORTS_KEY on the server.")
    if not key or not hmac.compare_digest(key, REPORTS_KEY):
        raise HTTPException(401, "Wrong or missing key.")


@router.post("/api/reports")
def create_report(rep: ReportIn):
    now = time.time()
    with _lock, closing(_db()) as con, con:
        n = con.execute("SELECT COUNT(*) FROM finds WHERE client=? AND created>?", (rep.client, now - 3600)).fetchone()[0]
        if n >= MAX_PER_CLIENT_PER_HOUR:
            raise HTTPException(429, "Too many reports from this browser in the last hour.")
        dup = con.execute("SELECT id FROM finds WHERE client=? AND ABS(ra-?)<? AND ABS(dec-?)<?",
                          (rep.client, rep.ra, SAME_OBJECT_DEG * 2, rep.dec, SAME_OBJECT_DEG)).fetchone()
        if dup:
            return {"id": dup[0], "duplicate": True}
        cur = con.execute(
            """INSERT INTO finds (created, client, name, ra, dec, field_ra, field_dec, fov, object_name, status,
               verdict, motion, motion_err, pa, catalog_motion, variability, molecules, detector, cadence, class_code)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (now, rep.client, rep.name.strip(), rep.ra, rep.dec, rep.field_ra, rep.field_dec, rep.fov,
             rep.object_name, rep.status, rep.verdict, rep.motion, rep.motion_err, rep.pa, rep.catalog_motion,
             rep.variability, json.dumps(rep.molecules), rep.detector, rep.cadence, _norm_class(rep.class_code)))
        return {"id": cur.lastrowid, "duplicate": False}


def _all() -> list[dict]:
    with closing(_db()) as con:
        return [_row(r) for r in con.execute("SELECT * FROM finds ORDER BY created DESC LIMIT 10000")]


@router.get("/api/reports")
def list_reports(key: str | None = None):
    _require_key(key)
    rows = _all()
    return {"reports": rows, "count": len(rows),
            "persistent_hint": None}


@router.get("/api/reports.csv")
def reports_csv(request: Request, key: str | None = None, class_code: str | None = None):
    _require_key(key)
    cc = _norm_class(class_code)
    base = str(request.base_url).rstrip("/")
    cols = ["id", "created_iso", "name", "class_code", "object_name", "status", "verdict", "ra", "dec", "motion", "motion_err",
            "pa", "catalog_motion", "variability", *MOLECULES, "detector", "cadence", "sky_link"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in _all():
        if cc and r.get("class_code") != cc:
            continue
        r.update(r.pop("molecules"))
        if r.get("field_ra") is not None:
            r["sky_link"] = f"{base}/?ra={r['field_ra']:.5f}&dec={r['field_dec']:.5f}&fov={r.get('fov') or 0.1}"
        for k in ("name", "class_code", "object_name", "verdict"):    # текст не должен стать формулой в Excel/Sheets
            v = r.get(k)
            if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r"):
                r[k] = "'" + v
        w.writerow(r)
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="lumen-reports.csv"'})


def summarize(rows: list[dict], class_code: str | None = None) -> dict:
    """Class overview for a teacher: who reported what and when. Rows are _row() dicts, newest first."""
    cc = _norm_class(class_code)
    if cc:
        rows = [r for r in rows if (r.get("class_code") or None) == cc]
    students: dict[tuple, dict] = {}
    objects: dict[str, dict] = {}
    days: dict[str, int] = {}
    classes: dict[str, int] = {}
    for r in rows:
        name = (r.get("name") or "").strip() or "Anonymous"
        k = (name.lower(), r.get("class_code") or "")
        s = students.setdefault(k, {"name": name, "class_code": r.get("class_code"), "reports": 0, "candidates": 0,
                                    "objects": [], "first": r["created"], "last": r["created"]})
        s["reports"] += 1
        s["candidates"] += r.get("status") == "candidate"
        s["first"], s["last"] = min(s["first"], r["created"]), max(s["last"], r["created"])
        obj = r.get("object_name") or f"Unidentified at {r['ra']:.4f}, {r['dec']:+.4f}"
        if obj not in s["objects"]:
            s["objects"].append(obj)
        o = objects.setdefault(obj, {"object_name": obj, "status": r.get("status"), "reports": 0, "students": [],
                                     "ra": r["ra"], "dec": r["dec"], "first": r["created"]})
        o["reports"] += 1
        o["first"] = min(o["first"], r["created"])
        if name not in o["students"]:
            o["students"].append(name)
        day = time.strftime("%Y-%m-%d", time.gmtime(r["created"]))
        days[day] = days.get(day, 0) + 1
        classes[r.get("class_code") or ""] = classes.get(r.get("class_code") or "", 0) + 1
    iso = lambda t: time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(t))
    for d in (*students.values(), *objects.values()):
        for f in ("first", "last"):
            if f in d:
                d[f + "_iso"] = iso(d[f])
    return {
        "class_code": cc,
        "totals": {"reports": len(rows), "students": len(students), "objects": len(objects),
                   "candidates": sum(r.get("status") == "candidate" for r in rows)},
        "students": sorted(students.values(), key=lambda s: (-s["reports"], s["name"].lower())),
        "objects": sorted(objects.values(), key=lambda o: (-o["reports"], o["object_name"].lower())),
        "days": [{"day": d, "reports": n} for d, n in sorted(days.items())],
        "classes": [{"class_code": c or None, "reports": n} for c, n in sorted(classes.items())],
    }


@router.get("/api/reports/summary")
def reports_summary(key: str | None = None, class_code: str | None = None):
    _require_key(key)
    return summarize(_all(), class_code)


@router.delete("/api/reports/{rid}")
def delete_report(rid: int, key: str | None = None):
    _require_key(key)
    with _lock, closing(_db()) as con, con:
        n = con.execute("DELETE FROM finds WHERE id=?", (rid,)).rowcount
    if not n:
        raise HTTPException(404, "No such report.")
    return {"deleted": rid}


@router.get("/reports", include_in_schema=False)
def reports_page():
    return FileResponse(ROOT / "web" / "reports.html")
