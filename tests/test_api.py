"""Быстрые проверки перед деплоем (без сети и без кэша SPHEREx): запускаются в GitHub Actions на каждый push.

    pip install -r requirements.txt pytest && pytest -q tests

Поднимают настоящий сервер (uvicorn) на свободном порту с временной базой отчётов.
Сценарии на реальных данных (полный прогон игры, линзы, WISE) — Playwright-тесты в tests/e2e.
"""
from __future__ import annotations

import csv
import io
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parent.parent
KEY = "test-key"


@pytest.fixture(scope="session")
def base(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("srv")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(os.environ, SKYWATCH_WARM="0", REPORTS_DB=str(tmp / "reports.db"), REPORTS_KEY=KEY, BACKUP_KEY=KEY)
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--port", str(port)], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            if requests.get(url + "/api/health", timeout=2).status_code in (200, 503):
                break
        except requests.ConnectionError:
            time.sleep(0.5)
    else:
        proc.kill()
        pytest.fail("server did not start:\n" + proc.stdout.read().decode(errors="replace")[-3000:])
    yield url
    proc.terminate()
    proc.wait(10)


def _report(**kw):
    r = dict(client="pytest-client-1", ra=77.9852, dec=-45.0595, field_ra=77.98521, field_dec=-45.0595, fov=0.1,
             object_name="Kapteyn's Star", status="known", verdict="Nearby star on the move", motion=8.1,
             molecules={"water": 0.31}, detector=4, cadence="pass")
    r.update(kw)
    return r


# ---------- site and health ----------

def test_health(base):
    h = requests.get(base + "/api/health").json()
    assert h["ok"] is True and h["reports_db"] is True and "wise" in h


def test_pages_served(base):
    home = requests.get(base + "/")
    assert home.status_code == 200 and "<canvas" in home.text
    assert requests.get(base + "/reports").status_code == 200


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("page", ["index.html", "reports.html"])
def test_page_scripts_parse(page, tmp_path):
    html = (ROOT / "web" / page).read_text()
    scripts = re.findall(r"<script>\n(.*?)\n</script>", html, re.S)
    assert scripts, "no inline script found"
    js = tmp_path / "page.js"
    js.write_text("\n".join(scripts))
    r = subprocess.run(["node", "--check", str(js)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ---------- API validation (bad input -> 422, never a crash) ----------

@pytest.mark.parametrize("path", [
    "/api/frames?ra=400&dec=0", "/api/frames?ra=10&dec=95", "/api/frames?ra=abc&dec=0",
    "/api/object?ra=-1&dec=0", "/api/track?ra=10&dec=0&fra=10&fdec=0&detector=9",
    "/api/frames?ra=10&dec=0&source=hubble",
])
def test_bad_params_are_422(base, path):
    assert requests.get(base + path).status_code == 422


# ---------- user reports ----------

def test_report_flow(base):
    r = requests.post(base + "/api/reports", json=_report())
    assert r.status_code == 200 and r.json()["duplicate"] is False
    rid = r.json()["id"]
    again = requests.post(base + "/api/reports", json=_report()).json()
    assert again == {"id": rid, "duplicate": True}
    assert requests.post(base + "/api/reports", json=_report(status="hacked")).status_code == 422
    assert requests.get(base + "/api/reports").status_code == 401
    assert requests.get(base + "/api/reports", params={"key": "wrong"}).status_code == 401
    listed = requests.get(base + "/api/reports", params={"key": KEY}).json()
    assert any(x["id"] == rid for x in listed["reports"]) and all("client" not in x for x in listed["reports"])


def test_reports_csv_is_formula_safe(base):
    requests.post(base + "/api/reports", json=_report(client="pytest-client-2", ra=10, dec=1, name="=HYPERLINK(1)"))
    text = requests.get(base + "/api/reports.csv", params={"key": KEY}).text
    names = [row["name"] for row in csv.DictReader(io.StringIO(text))]
    assert "'=HYPERLINK(1)" in names and "=HYPERLINK(1)" not in names


# ---------- backups ----------

def test_backup_needs_key(base):
    assert requests.get(base + "/api/admin/backup/reports.db").status_code == 401
    assert requests.get(base + "/api/admin/backup/cache.tar", params={"key": "wrong"}).status_code == 401


def test_backup_reports_db_is_valid(base, tmp_path):
    requests.post(base + "/api/reports", json=_report(client="pytest-client-3", ra=20, dec=2))
    r = requests.get(base + "/api/admin/backup/reports.db", params={"key": KEY})
    assert r.status_code == 200
    db = tmp_path / "copy.db"
    db.write_bytes(r.content)
    with sqlite3.connect(db) as con:
        assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert con.execute("SELECT COUNT(*) FROM finds").fetchone()[0] >= 2


def test_backup_cache_tar_is_readable(base):
    r = requests.get(base + "/api/admin/backup/cache.tar", params={"key": KEY})
    assert r.status_code == 200
    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:") as t:
        t.getmembers()   # an empty cache gives an empty but valid archive


# ---------- WISE: a dead third-party service must fail fast, not hang every request ----------

def test_wise_fails_fast_when_wiseview_is_down(monkeypatch):
    sys.path.insert(0, str(ROOT))
    import wise
    monkeypatch.setattr(wise, "API", "http://10.255.255.1/png-animation")   # unroutable
    monkeypatch.setattr(wise, "API_TIMEOUT_S", 2)
    monkeypatch.setattr(wise, "CACHE", Path("/nonexistent-for-test"))
    monkeypatch.setitem(wise._state, "down_until", 0.0)
    with pytest.raises(wise.WiseUnavailable, match="temporarily unavailable"):
        wise.get_wise(10.0, 20.0, 0.1)
    t = time.time()
    with pytest.raises(wise.WiseUnavailable):
        wise.get_wise(11.0, 21.0, 0.1)
    assert time.time() - t < 0.5 and wise.status()["ok"] is False


# ---------- click selection maths (web/index.html) ----------

@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_grid_sky_round_trip(tmp_path):
    html = (ROOT / "web" / "index.html").read_text()

    def grab(name):
        i = html.index(f"function {name}(")
        depth, j = 0, html.index("{", i)
        while True:
            depth += {"{": 1, "}": -1}.get(html[j], 0)
            if depth == 0:
                return html[i:j + 1]
            j += 1

    js = tmp_path / "sel.js"
    js.write_text("const CONFIG={PIX_ARCSEC:6.15}; const S={field:{ra:359.99,dec:-7.3,N:59}, apiFrames:{pixel_arcsec:6.15,center:{x:29.5,y:29.5}}};\n"
                  "const pixArc=()=>S.apiFrames.pixel_arcsec, fieldCentre=()=>S.apiFrames.center;\n"
                  + grab("gridToSky") + "\n" + grab("skyToGrid") + "\n"
                  "const s=gridToSky(12.3,40.1), b=skyToGrid(s.ra,s.dec);\n"
                  "if(Math.abs(b.x-12.3)>1e-6||Math.abs(b.y-40.1)>1e-6) throw new Error('round trip '+JSON.stringify(b));\n"
                  "const c=gridToSky(29.5,29.5); if(Math.abs(c.ra-359.99)>1e-9) throw new Error('centre');\n"
                  "if(!(gridToSky(19.5,29.5).ra > 359.99 || gridToSky(19.5,29.5).ra < 1)) throw new Error('east is left');\n")
    r = subprocess.run(["node", str(js)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ---------- "Surprise me" / site fields open without IRSA (seed/queries in git) ----------

def test_seeded_field_needs_no_irsa(monkeypatch, tmp_path):
    sys.path.insert(0, str(ROOT))
    import data
    seeds = sorted(data.SEED.glob("*.ecsv"))
    assert seeds, "seed/queries is empty: run make_seed.py"
    from astropy.table import Table
    meta = Table.read(seeds[0]).meta
    monkeypatch.setattr(data, "CACHE", tmp_path)                       # nothing cached
    monkeypatch.setattr(data, "_tap", lambda *a, **k: (_ for _ in ()).throw(AssertionError("IRSA was called")))
    tab = data.find_images(float(meta["ra"]), float(meta["dec"]), float(meta["size_deg"]))
    assert len(tab) > 0


def test_every_surprise_has_a_seed():
    """Every site field should open without IRSA. Fails until make_seed.py has been run with IRSA up."""
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("SKYWATCH_WARM", "0")
    import make_seed
    missing = [f"{n} {s}°" for n, ra, dec in make_seed.fields() for s in make_seed.SIZES
               if not (make_seed.SEED / make_seed.seed_name(ra, dec, s)).exists()]
    if missing:
        pytest.xfail(f"{len(missing)} seeds missing (run make_seed.py when IRSA is up): {missing[:3]}…")
