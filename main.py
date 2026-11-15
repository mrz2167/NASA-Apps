# AI disclosure: parts of this file were written with AI assistance (Claude, Anthropic), then reviewed and tested by the LUMEN team. See README, "AI use".
"""LUMEN Skywatch — бэкенд по контракту API.md.

    uvicorn main:app --reload            # http://localhost:8000
    в index.html: const CONFIG = { API_BASE: "http://localhost:8000", PIX_ARCSEC: 6.15 };

GET /api/frames?ra=&dec=&fov=0.1&detector=4&mode=blink
GET /api/object?ra=&dec=
GET /api/health
"""
from __future__ import annotations

import base64
import io
import os
import threading
import time
import warnings
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Annotated, Literal

import numpy as np
from astropy.coordinates import SkyCoord
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from sky import SkyUnavailable, analyze, get_sky
from render import fill_saturated, stretch_limits, to_uint8
from identify import _centroid, _check_catalogs, _deadline, gaia_cone, match_gaia, who_is_it
from data import CACHE
import skycache
from astropy.table import Table

warnings.filterwarnings("ignore")

app = FastAPI(
    title="LUMEN Skywatch API",
    description="Real NASA SPHEREx data for LUMEN Space Academy: blink frames, difference images, "
                "molecule lens maps, moving-object candidates and object cards. "
                "All coordinates are ICRS degrees. Invalid parameters return 422 with a readable message.",
)


# ---------- проверка параметров: 422 с понятным английским сообщением ----------

def _param_name(e: dict) -> str:
    return ".".join(str(x) for x in e.get("loc", ()) if x not in ("query", "body", "path")) or "request"


def _err_text(e: dict) -> str:
    msg, got = e.get("msg", "invalid value"), e.get("input")
    if got is None or isinstance(got, (dict, list)):
        return f"Invalid parameter '{_param_name(e)}': {msg}."
    return f"Invalid parameter '{_param_name(e)}': {msg} (got {got!r})."


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    return JSONResponse(status_code=422, content={
        "detail": " ".join(_err_text(e) for e in errors) or "Invalid request parameters.",
        "errors": [{"param": _param_name(e), "message": e.get("msg")} for e in errors],
    })


@app.exception_handler(Exception)
async def _unexpected_error(request: Request, exc: Exception):
    print(f"[error] {request.url.path}: {exc!r}", flush=True)
    return JSONResponse(status_code=500, content={
        "detail": "Internal server error while processing SPHEREx data. Please try again or pick another field."})


class FieldUnavailable(HTTPException):
    """503 for a field that can't be built now. The body keeps the readable `detail` and adds a stable `code`:
    `archive_down` when NASA's archive (IRSA) isn't answering, so the site can offer WISE and the nearest
    ready fields instead of a dead end; `no_data` for the rest (not observed yet, too few images)."""

    def __init__(self, e: Exception):
        msg = str(e)
        low = msg.lower()
        self.code = "archive_down" if ("not responding" in low or "unavailable" in low) else "no_data"
        super().__init__(503, msg)


@app.exception_handler(FieldUnavailable)
async def _field_unavailable(request: Request, exc: FieldUnavailable):
    return JSONResponse(status_code=503, content={"detail": exc.detail, "code": exc.code})


# Общие параметры запросов (ограничения видны в /docs)
RA = Annotated[float, Query(ge=0, le=360, description="Right ascension of the field centre, degrees (0–360, ICRS).")]
DEC = Annotated[float, Query(ge=-90, le=90, description="Declination of the field centre, degrees (−90…+90, ICRS).")]
FOV = Annotated[float, Query(gt=0.01, le=0.5, description="Field of view (side of the square), degrees: more than "
                                                          "0.01 and at most 0.5. The site uses 0.05, 0.1 or 0.2.")]
DETECTOR = Annotated[int, Query(ge=1, le=6, description="SPHEREx detector 1–6 (D1 ≈ 0.75 µm … D6 ≈ 5 µm).")]
CADENCE = Annotated[Literal["pass", "visit"], Query(
    description="`pass`: one frame per six-month sky pass; `visit`: each pass split into observing visits.")]
SOURCE = Annotated[Literal["spherex", "wise"], Query(
    description="Survey: `spherex` (2025–26, six detectors) or `wise` (WISE/NEOWISE W2 4.6 µm, 2010–2024).")]
MODE = Annotated[Literal["blink"], Query(description="Display mode; only `blink` is supported for now.")]
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])

from reports import router as reports_router  # noqa: E402  общая таблица находок (/api/reports, /reports)
app.include_router(reports_router)
from backup import router as backup_router  # noqa: E402  резервные копии (/api/admin/backup/…)
app.include_router(backup_router)

PIX_ARCSEC = 6.15
DIFF_POS = np.array([0xF6, 0xB5, 0x44], float)   # ярче — янтарный
DIFF_NEG = np.array([0x3D, 0x5B, 0xD9], float)   # темнее — синий
DIFF_ZERO = np.array([0x07, 0x0A, 0x20], float)  # без изменений — фон


WEB = Path(__file__).resolve().parent / "web"


@app.get("/", include_in_schema=False)
def site():
    return FileResponse(WEB / "index.html")


FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="6" fill="#0B1030"/>'
    '<path fill="#F6B544" d="M16 4l3.2 8.1 8.6.6-6.6 5.5 2.1 8.4L16 22l-7.3 4.6 2.1-8.4-6.6-5.5 8.6-.6z"/>'
    '</svg>'
)


@app.get("/favicon.ico", include_in_schema=False)
@app.get("/favicon.svg", include_in_schema=False)
def favicon():
    return Response(FAVICON_SVG, media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/i18n/{name}", include_in_schema=False)
def i18n_file(name: str):
    """Interface translations (web/i18n/<lang>.json, key = the English text)."""
    if name not in ("ru.json", "uz.json", "en.json") or not (WEB / "i18n" / name).is_file():
        raise HTTPException(404, "No such file")
    return FileResponse(WEB / "i18n" / name, media_type="application/json")


@app.get("/levels/{name}", include_in_schema=False)
def level_file(name: str):
    """Real frames for the training levels (make_training.py -> web/levels/)."""
    import re
    if not re.fullmatch(r"[a-z0-9_]+\.(png|json)", name) or not (WEB / "levels" / name).is_file():
        raise HTTPException(404, "No such file")
    return FileResponse(WEB / "levels" / name)


# ---------- кэш наборов кадров ----------

_SKIES: "OrderedDict[tuple, dict]" = OrderedDict()
_LOCK = threading.Lock()
_MAX_SKIES = 32          # ~6–30 MB per field; the demo fields, visits and WISE all fit


# ---------- progress of a field being built, and where its exposure list came from ----------
# /api/frames is one blocking request; while it builds a new field (10–200 s) the site polls /api/progress.
# The record lives only while the build runs: fields already in memory or on disk never get one.
_PROGRESS: dict = {}                 # field key -> {"stage", "done", "total", "t0"}
_BUILD = threading.local()           # .key: field this thread is building; .search: how its list was found


def _prog(stage: str, done: int | None = None, total: int | None = None) -> None:
    key = getattr(_BUILD, "key", None)
    if key is None:
        return
    old = _PROGRESS.get(key)
    _BUILD.fresh = True
    _PROGRESS[key] = {"stage": stage, "done": done, "total": total, "t0": old["t0"] if old else time.time()}


def _on_saved(i: int, n: int) -> None:        # cutouts read from our own disk (offline pass)
    _prog("build" if i >= n else "saved", i, n)


def _on_download(i: int, n: int) -> None:     # cutouts fetched from the AWS bucket / IRSA (or already on disk)
    _prog("build" if i >= n else "download", i, n)


def _watch_search() -> None:
    """Note which route data.find_images takes: IRSA's search, our own index of the AWS bucket, or neither
    (a stored query). Wraps two helpers of data.py from here, so data.py itself stays untouched (editing it
    would invalidate every saved sky, see skycache.CODE_HASH). If the helpers are renamed, this does nothing."""
    import data as _data
    tap, index = getattr(_data, "_tap", None), getattr(_data, "_from_index", None)
    if tap is None or index is None or getattr(tap, "_watched", False):
        return

    def _tap(*a, **kw):
        _prog("irsa")
        out = tap(*a, **kw)
        _BUILD.search = "irsa"
        return out

    def _from_index(*a, **kw):
        _prog("index")
        out = index(*a, **kw)
        if out is not None:
            _BUILD.search = "aws-index"
        return out

    _tap._watched = True
    _data._tap, _data._from_index = _tap, _from_index


_watch_search()


def _tracks_progress(load):
    """Decorator for _load_sky: names the field this thread is working on and clears its record afterwards."""
    import functools

    @functools.wraps(load)
    def wrapped(ra, dec, fov, cadence="pass", det=None, remember=True):
        c, d = (cadence, det) if cadence == "visit" else ("pass", None)
        key = (round(ra, 4), round(dec, 4), round(fov, 4), c, d)        # same key as in _load_sky
        prev = getattr(_BUILD, "key", None)
        _BUILD.key = key
        if prev is None:
            _BUILD.search = "saved"
        try:
            return load(ra, dec, fov, cadence, det, remember)
        finally:
            _BUILD.key = prev
            _PROGRESS.pop(key, None)
    return wrapped


@_tracks_progress
def _load_sky(ra: float, dec: float, fov: float, cadence: str = "pass", det: int | None = None,
              remember: bool = True) -> dict:
    """Кадры для места: сначала полный набор из кэша на диске, иначе быстрый режим.

    remember=False — собрать и сохранить на диск, но не класть в память (прогрев полей «Surprise me»
    не должен вытеснять из памяти демо-поля).

    cadence="visit" — проходы порезаны на сеансы съёмки (epochs.split_cadence); такие кадры собираются
    только для детектора det: сеансы у каждого детектора свои, и пересобирать все шесть незачем.
    """
    if cadence != "visit":
        cadence, det = "pass", None
    dets = [det] if det else None
    key = (round(ra, 4), round(dec, 4), round(fov, 4), cadence, det)
    with _LOCK:
        hit = _SKIES.get(key)
        if hit is not None:
            _SKIES.move_to_end(key)
    if hit is not None:
        _start_gaia(hit)                      # no-op once loaded; retries a field whose Gaia failed a while ago
        return hit
    sky = None
    if fov < 0.1:
        try:
            big = _load_sky(ra, dec, 0.1, cadence, det)
            return _crop_sky(big, fov)
        except SkyUnavailable:
            pass
    sky = skycache.get(key)                   # built before (survives restarts on the volume): no FITS reading
    if sky is not None:
        if remember:
            _remember(key, sky)
        _start_gaia(sky)
        return sky
    degraded = False
    try:
        full = get_sky(ra, dec, fov, quick=False, offline=True, detectors=dets, cadence=cadence,
                       on_progress=_on_saved)
        if full["n_loaded"] >= 0.8 * full["n_wanted"]:
            sky = full
    except SkyUnavailable:
        pass
    if sky is None:
        kw = dict(quick=True, detectors=dets, cadence=cadence)
        _prog("search")
        try:
            sky = get_sky(ra, dec, fov, on_progress=_on_download, **kw)
        except SkyUnavailable as net_err:
            try:
                sky = get_sky(ra, dec, fov, offline=True, on_progress=_on_saved, **kw)  # архив лёг — что есть в кэше
            except SkyUnavailable:
                raise net_err from None       # say "the archive isn't answering", not "offline mode"
            degraded = True
    sky["search"] = getattr(_BUILD, "search", "saved")   # "irsa" | "aws-index" | "saved": shown on the site
    if not degraded:                          # a stop-gap set from a dead archive isn't worth keeping for a week
        sky = skycache.put(key, sky)          # to disk for the next restart; keep the slim copy in memory
    if remember:
        _remember(key, sky)
    _start_gaia(sky)
    return sky


def _remember(key: tuple, sky: dict) -> None:
    with _LOCK:
        _SKIES[key] = sky
        _SKIES.move_to_end(key)
        while len(_SKIES) > _MAX_SKIES:
            _, old = _SKIES.popitem(last=False)
            _forget(id(old))


def _forget(sid: int) -> None:
    """Drop everything computed for an evicted sky (call with _LOCK held): cards, analyses, catalogue
    answers. They are keyed by id(sky), and Python may hand the same id to the next sky."""
    try:
        for store in (_OBJECTS, _ANALYSES, _PRE_SEEN):
            for k in [k for k in list(store) if k[0] == sid]:
                store.pop(k, None)
        with _CAT_LOCK:
            for k in [k for k in _CAT_DONE if k[0] == sid]:
                _CAT_DONE.pop(k, None)
    except Exception:
        pass


_GAIA: dict = {}          # ключ поля -> Table | None
_GAIA_THREADS: dict = {}
_GAIA_FAILED: dict = {}   # field key -> when its Gaia query last failed (tried again after GAIA_RETRY_S)
_GAIA_DOWN_UNTIL = 0.0    # a Gaia query just failed or timed out: don't make clicks wait for the next ones
GAIA_FIELD_TIMEOUT_S = 60.0   # normal answer is ~20 s; pyvo has no connect timeout of its own
GAIA_RETRY_S = 180.0


def _gaia_down() -> bool:
    """Gaia comes through IRSA. True while IRSA is known not to answer (the exposure search fell back to
    our own index, or a Gaia query has just failed)."""
    if time.time() < _GAIA_DOWN_UNTIL:
        return True
    try:
        from data import archive_status
        return archive_status().get("irsa") == "down"
    except Exception:
        return False


def _gaia_key(sky: dict) -> str:
    c = sky["coord"]
    return f"{c.ra.deg:.4f}_{c.dec.deg:+.4f}_{sky['size_deg']}"


def _start_gaia(sky: dict) -> None:
    """В фоне один раз загрузить Gaia DR3 для поля (диск-кэш cache/gaia/)."""
    key = _gaia_key(sky)
    if _GAIA.get(key) is not None or key in _GAIA_THREADS:
        return
    path = CACHE / "gaia" / f"{key}.ecsv"
    if path.exists():
        _GAIA[key] = Table.read(path)
        return
    # IRSA down: a query would hang for minutes and every click in the field would wait for it
    if time.time() - _GAIA_FAILED.get(key, 0.0) < GAIA_RETRY_S or _gaia_down():
        return

    def job():
        global _GAIA_DOWN_UNTIL
        try:
            radius = sky["size_deg"] / 2 * 1.45 * 3600  # описанный круг квадрата
            t = _deadline(lambda: gaia_cone(sky["coord"], radius, top=3000, gmax=20.5), GAIA_FIELD_TIMEOUT_S)
            path.parent.mkdir(parents=True, exist_ok=True)
            t.write(path, overwrite=True)
            _GAIA[key] = t
        except Exception:
            _GAIA[key] = None
            _GAIA_FAILED[key] = time.time()
            _GAIA_DOWN_UNTIL = time.time() + GAIA_RETRY_S
        finally:
            _GAIA_THREADS.pop(key, None)

    th = threading.Thread(target=job, daemon=True)
    _GAIA_THREADS[key] = th
    th.start()


def _gaia_for(sky: dict, wait_s: float = 0.0):
    key = _gaia_key(sky)
    th = _GAIA_THREADS.get(key)
    if th is not None and wait_s > 0:
        th.join(wait_s)
    return _GAIA.get(key)


def _crop_sky(sky: dict, fov: float) -> dict:
    """Центральный квадрат fov из набора кадров побольше (без повторной загрузки)."""
    from epochs import diff_frames, frames_by_detector
    ny, nx = sky["shape"]
    n = max(8, int(round(fov * 3600 / PIX_ARCSEC)))
    y0, x0 = (ny - n) // 2, (nx - n) // 2
    frames = [(d, det, im[y0:y0 + n, x0:x0 + n]) for d, det, im in sky["frames"]]
    out = dict(sky, frames=frames, shape=(n, n), size_deg=fov, wcs=sky["wcs"][y0:y0 + n, x0:x0 + n],
               by_det=frames_by_detector(frames), diffs=diff_frames(frames))
    return out


def _sky_containing(ra: float, dec: float) -> dict | None:
    """Уже загруженный набор кадров, в поле которого лежит точка (для клика по объекту)."""
    c = SkyCoord(ra, dec, unit="deg")
    with _LOCK:   # только полные наборы: в сеансовых лежит один детектор, спектра из них не собрать
        skies = [s for s in reversed(_SKIES.values()) if s.get("cadence", "pass") == "pass"]
    for sky in skies:
        x, y = (float(v) for v in sky["wcs"].world_to_pixel(c))
        ny, nx = sky["shape"]
        if 3 <= x < nx - 3 and 3 <= y < ny - 3:
            return sky
    return None


# ---------- картинки ----------

def _png_uri(rgb_or_gray: np.ndarray) -> str:
    img = Image.fromarray(np.flipud(rgb_or_gray))  # север вверх, восток влево
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _diff_rgb(d: np.ndarray, lim: float) -> np.ndarray:
    x = np.clip(np.nan_to_num(d) / lim, -1, 1)[..., None]
    rgb = np.where(x >= 0, DIFF_ZERO + x * (DIFF_POS - DIFF_ZERO), DIFF_ZERO + (-x) * (DIFF_NEG - DIFF_ZERO))
    return rgb.astype(np.uint8)


def _epoch_labels(dates: list[str], cadence: str = "pass") -> list[str]:
    if cadence == "visit":
        return [f"{d} · Frame {k + 1}" for k, d in enumerate(dates)]
    return [f"{d[:4]} · Epoch {k + 1}" for k, d in enumerate(dates)]


def _to_display(x: float, y: float, ny: int) -> dict:
    """Пиксель сетки (numpy, ось y вверх, центр пикселя = целое) -> координаты фронтенда
    (x вправо, y вниз, 0 в левом верхнем углу)."""
    return {"x": round(x + 0.5, 2), "y": round(ny - (y + 0.5), 2)}


def _sources(img: np.ndarray, n: int = 40, sky: dict | None = None) -> list[dict]:
    """Яркие источники кадра (локальные максимумы > 5σ) в координатах фронтенда."""
    from photutils.detection import find_peaks
    v = img[np.isfinite(img)]
    if v.size == 0:
        return []
    med = np.median(v)
    sig = 1.4826 * np.median(np.abs(v - med))
    filled = fill_saturated(img)
    t = find_peaks(np.nan_to_num(filled - med), threshold=5 * sig, box_size=5, npeaks=3 * n, border_width=2)
    if t is None:
        return []
    t.sort("peak_value", reverse=True)
    # Пересвеченное ядро после fill_saturated — плато: find_peaks выдаёт на нём россыпь соседних «пиков».
    # Склеиваем пики ближе SOURCE_MERGE_PX к более яркому в один источник (центр — среднее по плато).
    groups: list[list] = []   # [x_ведущего, y_ведущего, пик_ведущего, [точки плато]]
    for r in t:
        x, y, p = float(r["x_peak"]), float(r["y_peak"]), float(r["peak_value"])
        for g in groups:
            if (g[0] - x) ** 2 + (g[1] - y) ** 2 <= SOURCE_MERGE_PX ** 2:
                if p >= 0.98 * g[2]:
                    g[3].append((x, y))
                break
        else:
            groups.append([x, y, p, [(x, y)]])
    pts = [tuple(np.mean(g[3], axis=0)) for g in groups][:n]
    ny = img.shape[0]

    # Gaia: один номер Gaia — не больше чем одному источнику (ближайшему), остальным — общая подпись
    gaia = _gaia_for(sky) if sky is not None else None
    best: dict[str, tuple[float, int]] = {}   # имя Gaia -> (расстояние, индекс источника)
    if gaia is not None:
        for k, (x, y) in enumerate(pts):
            m = match_gaia(sky["wcs"].pixel_to_world(x, y), gaia, radius_arcsec=6.0)
            if m and (m["name"] not in best or m["sep_arcsec"] < best[m["name"]][0]):
                best[m["name"]] = (m["sep_arcsec"], k)
    names = {k: name for name, (_, k) in best.items()}
    return [dict(_to_display(x, y, ny), id=names.get(k, f"SPHEREx src {k + 1}")) for k, (x, y) in enumerate(pts)]


SOURCE_MERGE_PX = 3.0   # find_peaks(box_size=5) сам разводит пики на 3+ пкс; ближе — это плато пересвеченной звезды


def _clean(o):
    """numpy-типы -> Python, NaN/inf -> None (иначе JSON ломается)."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


# ---------- эндпоинты ----------

@app.get("/api/health")
def health():
    """Liveness check: `ok`, how many fields are held in memory, and the warm-up after a restart.
    Returns 503 when something essential is broken (reports database, disk), so an uptime monitor alerts.
    `wise.ok` is false while WiseView (third-party, WISE frames) is down: a warning, not a failure.

    `warm.state`: `running` while the demo fields are being loaded, `demo_ready` once every case and
    "Try" field is in memory (the demo is fast from here), `done` when the "Surprise me" fields are
    also saved to disk.
    """
    import shutil
    from fastapi.responses import JSONResponse
    import reports as _reports
    import wise as _wise_mod
    problems = []
    try:
        CACHE.mkdir(parents=True, exist_ok=True)      # a fresh checkout (CI) has no cache folder yet
        du = shutil.disk_usage(CACHE)
        disk = {"free_gb": round(du.free / 1e9, 1), "used_gb": round(du.used / 1e9, 1)}
        if du.free < 0.3e9:
            problems.append(f"disk almost full ({du.free / 1e9:.2f} GB free)")
    except OSError:
        disk = None
        problems.append("cache disk not readable")
    try:                                   # отчёты пользователей: база открывается и в неё можно писать
        from contextlib import closing
        with closing(_reports._db()) as con:
            con.execute("BEGIN IMMEDIATE"); con.rollback()
        reports_ok = True
    except Exception as e:
        reports_ok = False
        problems.append(f"reports database: {e}")
    body = {"ok": not problems, "problems": problems, "cached_fields": len(_SKIES), "warm": dict(_WARM),
            "disk": disk, "reports_db": reports_ok, "wise": _wise_mod.status(),
            "archive": dict(__import__("data").archive_status(), watch=dict(_IRSA_WATCH))}   # irsa ok/down; is our own AWS index taking over
    body["cleanup"] = cleanup.status()     # old visitor fields freed when the disk runs low (cleanup.py)
    # 503 — чтобы внешний монитор (UptimeRobot) прислал тревогу; WISE — сторонний сервис, он только предупреждение
    return body if not problems else JSONResponse(body, status_code=503)


def _det_items(sky: dict, detector: int) -> list:
    """Кадры детектора в том порядке, в каком их показывает /api/frames (без шумных кадров)."""
    items = [(i, f) for i, f in enumerate(sky["frames"]) if f[1] == detector]
    if not items:
        raise HTTPException(404, f"No frames for detector {detector} in this field; available detectors: "
                                 f"{sorted(int(d) for d in sky['by_det'])}.")
    info = sky.get("info") or []
    if len(items) > 2 and len(info) == len(sky["frames"]):   # без шумных кадров из 2–4 снимков
        n = np.array([info[i]["n_exposures"] for i, _ in items])
        keep = n >= 0.25 * np.median(n)
        if keep.sum() >= 2:
            items = [it for it, k in zip(items, keep) if k]
    if len(items) > MAX_FRAMES:   # глубокие поля у полюсов эклиптики дают до сотни сеансов
        pick = np.linspace(0, len(items) - 1, MAX_FRAMES).round().astype(int)
        items = [items[k] for k in pick]
    return items


MAX_FRAMES = 36   # больше кадров мигать бессмысленно (и PNG-ответ раздувается)


def _track(sky: dict, imgs: list, coord: SkyCoord) -> list[dict]:
    """Положение объекта у coord на каждом кадре (координаты фронтенда)."""
    ny = sky["shape"][0]
    x0, y0 = (float(v) for v in sky["wcs"].world_to_pixel(coord))
    out = []
    for im in imgs:
        c = _centroid(im, x0, y0)
        out.append(_to_display(*(c if c is not None else (x0, y0)), ny))
    return out


@app.get("/api/frames")
def frames(ra: RA, dec: DEC, fov: FOV = 0.1, detector: DETECTOR = 4, mode: MODE = "blink",
           cadence: CADENCE = "pass", source: SOURCE = "spherex"):
    """Blink frames for one detector, stacked per epoch and reprojected onto one grid (north up, east left).

    Also returns difference images (each frame minus the first), clickable bright sources (Gaia DR3 IDs where
    matched; each Gaia ID goes to at most one source, the nearest) and the track of the central object.
    `cadence=pass` gives one frame per sky pass; `cadence=visit` splits each pass into observing visits.
    `source=wise` returns WISE/NEOWISE W2 frames from 2010–2024 instead (detector and cadence are ignored).
    """
    if source == "wise":
        return _wise_frames(ra, dec, fov)
    _BUILD.fresh = False
    try:
        sky = _load_sky(ra, dec, fov, cadence, detector)
    except SkyUnavailable as e:
        raise FieldUnavailable(e)             # 503 with code "archive_down" when IRSA is not answering

    items = _det_items(sky, detector)
    info = sky.get("info") or []

    imgs = [f[2] for _, f in items]
    dates = [f[0] for _, f in items]
    shown = [fill_saturated(im) for im in imgs]
    lo, hi = stretch_limits(shown)
    labels = _epoch_labels(dates, cadence)
    out_frames = [{"label": lab, "date": d, "png": _png_uri(to_uint8(im, lo, hi))}
                  for lab, d, im in zip(labels, dates, shown)]

    # разности: кадр минус первый, с подгонкой общей яркости
    from epochs import _flux_scale
    ref = imgs[0]
    diffs = [im - _flux_scale(im, ref) * ref for im in imgs]
    finite = np.concatenate([d[np.isfinite(d)] for d in diffs[1:]]) if len(diffs) > 1 else np.array([1.0])
    lim = float(np.percentile(np.abs(finite), 99.5)) or 1.0
    out_diff = [{"label": lab, "png": _png_uri(_diff_rgb(d, lim))} for lab, d in zip(labels, diffs)]

    # объект в центре: положение на каждом кадре
    positions = _track(sky, imgs, sky["coord"])

    try:   # химия и движение — по полному набору (все детекторы, полугодовые эпохи)
        base = sky if cadence == "pass" else _load_sky(ra, dec, fov)
        a = _analysis(base, base["coord"])
        molecules = a["scores"] or None
        mol_snr = _mol_snr(a)
    except Exception:
        molecules, mol_snr = None, None
    _precache(sky, imgs[-1], detector)        # object cards of the centre and the brightest dots, in the background

    return _clean({
        "size_px": int(sky["shape"][0]),
        "pixel_arcsec": PIX_ARCSEC,
        "frames": out_frames,
        "diff": out_diff,
        "sources": _sources(imgs[-1], sky=sky),
        "target": {"positions": positions, "molecules": molecules, "molecules_snr": mol_snr},
        "meta": {"source": sky["source"], "n_images": sky["n_loaded"], "warnings": sky["warnings"],
                 "wavelength_um": [info[i]["wavelength_um"] for i, _ in items] if info else None,
                 "n_exposures": [info[i]["n_exposures"] for i, _ in items] if info else None,
                 "cadence": cadence,
                 # how the list of exposures was found when this field was built: "irsa" (NASA's archive),
                 # "aws-index" (our own index of NASA's AWS bucket; IRSA was not answering) or "saved"
                 # (a stored search or a saved field; no archive call). fresh: built by this very request.
                 "search": sky.get("search") or "saved", "fresh": bool(getattr(_BUILD, "fresh", False))},
    })


@app.get("/api/progress")
def progress(ra: RA, dec: DEC, fov: FOV = 0.1, detector: DETECTOR = 4, cadence: CADENCE = "pass"):
    """What the server is doing for a field that `/api/frames` is building right now. Never builds anything.

    `{"active": false}` when nothing is being built for this field (it is ready, or nobody asked). Otherwise
    `stage`: `search` (looking up which exposures cover the spot), `irsa` (asking NASA's IRSA archive),
    `index` (IRSA is not answering: our own index of the AWS bucket), `download` (fetching cutouts,
    `done`/`total`), `saved` (reading cutouts already on our disk, `done`/`total`), `build` (stacking frames);
    plus `elapsed_s`.
    """
    r4 = lambda v: round(v, 4)
    visit = (cadence, detector) if cadence == "visit" else ("pass", None)
    for f in (fov, 0.1):                       # fields under 0.1° are cut from the 0.1° one
        for c, d in (visit, ("pass", None)):   # a visit request also needs the per-pass set (chemistry)
            rec = _PROGRESS.get((r4(ra), r4(dec), r4(f), c, d))
            if rec:
                return {"active": True, "stage": rec["stage"], "done": rec["done"], "total": rec["total"],
                        "elapsed_s": round(time.time() - rec["t0"], 1)}
    return {"active": False}


LENS_RGB = {"water": (0x3D, 0x5B, 0xD9), "methane": (0xF0, 0x7A, 0x3A),
            "co2": (0xF6, 0xB5, 0x44), "co": (0xE0, 0x35, 0x2B)}   # цвета линз сайта


LENS_MIN_SPOT_PX = 8   # пятно звезды после сглаживания 3×3 ≈ 15 px; мельче — одиночные шумовые пиксели фона


def _lens_denoise(depth: np.ndarray) -> np.ndarray:
    """Убрать с карты линзы мелкие пятна (< LENS_MIN_SPOT_PX связанных пикселей): это шум фона, не объекты."""
    from scipy.ndimage import label
    d = np.nan_to_num(depth)
    lab, n = label(d >= 0.08, structure=np.ones((3, 3)))
    if not n:
        return d
    sizes = np.bincount(lab.ravel())
    small = sizes < LENS_MIN_SPOT_PX
    small[0] = False
    return np.where(small[lab], 0.0, d)


def _lens_png(depth: np.ndarray, rgb) -> str:
    """Карта глубины -> PNG цвета молекулы с прозрачностью: от 0.08 (едва) до 0.43+ (полностью)."""
    a = np.clip((np.nan_to_num(depth) - 0.08) / 0.35, 0, 1)
    img = np.zeros(depth.shape + (4,), np.uint8)
    img[..., :3] = rgb
    img[..., 3] = (a * 235).astype(np.uint8)
    return _png_uri(img)


def _lens_payload(sky: dict, maps: dict) -> dict:
    clean = {k: _lens_denoise(v) for k, v in maps.items()}
    return _clean({
        "size_px": int(sky["shape"][0]),
        "maps": {k: _lens_png(v, LENS_RGB[k]) for k, v in clean.items()},
        "peak": {k: round(float(v.max()), 3) if v.size else 0.0 for k, v in clean.items()},
        "lit": {k: int(np.sum(v >= 0.2)) for k, v in clean.items()},
    })


_LENS_LOCK = threading.Lock()


def _paint_from_spectrum(sky: dict, maps: dict, coord: SkyCoord) -> dict:
    """Пятно объекта у coord по индексам его спектра (те же числа, что в карточке /api/object).

    Для объектов, слишком тусклых для попиксельной карты; измеренные пиксели не перекрываются.
    """
    from lens import paint_object
    from spectrum import MIN_SNR
    c = _snap(sky, coord)
    a = _analysis(sky, c)
    snr = _mol_snr(a) or {}
    # only significant signs are painted: a noisy index (WISE 0855 methane, 1.4σ) would draw a false spot
    scores = {k: v for k, v in (a["scores"] or {}).items() if (snr.get(k) or 0) >= MIN_SNR}
    if not any(v > 0 for v in scores.values()):
        return maps
    x, y = (float(v) for v in sky["wcs"].world_to_pixel(c))
    ny, nx = sky["shape"]
    if not (0 <= x < nx and 0 <= y < ny):
        return maps
    return paint_object(maps, x, y, scores)


@app.get("/api/lens")
def lens(ra: RA, dec: DEC, fov: FOV = 0.1,
         ora: Annotated[float | None, Query(ge=0, le=360, description="RA of a clicked object, degrees (optional).")] = None,
         odec: Annotated[float | None, Query(ge=-90, le=90, description="Dec of a clicked object, degrees (optional).")] = None):
    """Molecule lens maps: where water, methane, CO₂ and CO absorption is stronger than in the surroundings.

    Same grid and orientation as `/api/frames` with the same ra, dec, fov.
    `maps` are transparent PNG overlays; `peak` is the deepest band in the field (0–1);
    `lit` is how many pixels light up noticeably.
    `ora`, `odec`: a clicked object. If it is too faint for the per-pixel map, its spot is painted from its
    spectrum. The paint is remembered for the field, so later responses already include it.
    """
    from lens import molecule_maps
    try:
        sky = _load_sky(ra, dec, fov)
    except SkyUnavailable as e:
        raise HTTPException(503, str(e))
    if "_lens_raw" not in sky:   # считается один раз на поле и живёт вместе с ним в кэше _SKIES
        maps = molecule_maps(sky)
        # объект в центре поля: если он слишком тусклый для попиксельной карты (WISE 0855),
        # его пятно закрашивается по индексам его спектра — тем же, что в карточке /api/object
        try:
            maps = _paint_from_spectrum(sky, maps, sky["coord"])
        except Exception:
            pass
        with _LENS_LOCK:
            if "_lens_raw" not in sky:
                sky["_lens_raw"], sky["_lens_painted"] = maps, set()
                sky["_lens"] = _lens_payload(sky, maps)
    if ora is not None and odec is not None:
        key = (round(ora, 5), round(odec, 5))
        if key not in sky["_lens_painted"]:
            try:
                painted = _paint_from_spectrum(sky, sky["_lens_raw"], SkyCoord(ora, odec, unit="deg"))
            except Exception:
                painted = None
            with _LENS_LOCK:
                sky["_lens_painted"].add(key)
                if painted is not None and painted is not sky["_lens_raw"]:
                    # поверх текущей карты: другой клик мог успеть закрасить своё пятно
                    cur = sky["_lens_raw"]
                    merged = {k: np.where(np.isfinite(cur[k]), cur[k], painted[k]) for k in cur}
                    sky["_lens_raw"] = merged
                    sky["_lens"] = _lens_payload(sky, merged)
    return sky["_lens"]


@app.get("/api/movers")
def movers(ra: RA, dec: DEC, fov: FOV = 0.1, detector: DETECTOR = 4):
    """Moving-object candidates (asteroids, comets) on the visit frames (`cadence=visit`).

    A candidate's `frames` are indices in the same order as `/api/frames?cadence=visit` with the same detector;
    `positions` are frontend pixel coordinates on those frames. `kind`: `track` means the spot moved along a
    line over 2+ frames (slow object); `transient` means it appears on one frame only (fast asteroid or
    artefact; lower confidence). `speed_arcsec_day` is given for tracks only.
    """
    from movers import find_movers
    try:
        sky = _load_sky(ra, dec, fov, "visit", detector)
    except SkyUnavailable as e:
        raise HTTPException(503, str(e))
    items = _det_items(sky, detector)
    info = sky.get("info") or []
    imgs = [f[2] for _, f in items]
    if len(info) == len(sky["frames"]):
        mjds = [(info[i]["mjd_min"] + info[i]["mjd_max"]) / 2 for i, _ in items]
        nexp = [info[i]["n_exposures"] for i, _ in items]
    else:   # без подписей кадров времени нет — связывать в треки нечем
        mjds, nexp = list(range(len(imgs))), None
    ny = sky["shape"][0]
    out = []
    for c in find_movers(imgs, mjds, nexp):
        w = sky["wcs"].pixel_to_world(c["x"][0], c["y"][0])
        out.append({
            "kind": c["kind"], "frames": c["frames"],
            "positions": [_to_display(x, y, ny) for x, y in zip(c["x"], c["y"])],
            "ra": round(float(w.ra.deg), 6), "dec": round(float(w.dec.deg), 6),
            "snr": round(c["snr"], 1), "confidence": round(c["confidence"], 2),
            "speed_arcsec_day": round(c["speed_px_day"] * PIX_ARCSEC, 1) if c["speed_px_day"] else None,
        })
    return _clean({"n_frames": len(imgs), "candidates": out})


# ---------- WISE/NEOWISE 2010–2024 (wise.py) ----------

def _wise(ra: float, dec: float, fov: float) -> dict:
    from wise import WiseUnavailable, get_wise
    try:
        return get_wise(ra, dec, fov)
    except WiseUnavailable as e:
        raise HTTPException(503, str(e))


def _wise_px(w: dict, coord: SkyCoord) -> tuple[float, float]:
    """Небо -> (столбец, строка сверху вниз) кадра WISE. Строки PNG идут сверху вниз, а у WiseView
    пиксели считаются с нуля: сверено по звёздам Gaia в трёх полях (остаток ≲ 0.5 пикселя)."""
    x, y = (float(v) for v in w["wcs"].world_to_pixel(coord))
    return x, w["shape"][0] - 2 - y


def _wise_sky(w: dict, col: float, row: float) -> SkyCoord:
    return w["wcs"].pixel_to_world(col, w["shape"][0] - 2 - row)


def _wise_known_path(w: dict, center: SkyCoord) -> list | None:
    """Если в центре поля тестовый объект (targets.py), его путь по кадрам WISE — по каталожному
    движению с уточнением по пятну рядом. Центр поля — положение «сейчас», а от последнего кадра WISE
    (2024) быстрые объекты успевают уйти на несколько пикселей, и ведение от центра срывается."""
    from targets import TARGETS, coord_at
    from wise import centroid
    for name in TARGETS:
        try:
            if coord_at(name).separation(center).arcsec > 30:
                continue
        except Exception:
            continue
        out = []
        for f in w["frames"]:
            px, py = _wise_px(w, coord_at(name, f["date"]))
            out.append(centroid(f["image"], px, py, search=2) or (px, py))
        return out
    return None


def _wise_frames(ra: float, dec: float, fov: float) -> dict:
    from wise import track as wtrack
    w = _wise(ra, dec, fov)
    imgs = [f["image"] for f in w["frames"]]
    ny = w["shape"][0]
    labels = [f"{f['date'][:4]} · {'Spring' if int(f['date'][5:7]) < 7 else 'Autumn'}" for f in w["frames"]]
    out_frames = [{"label": lab, "date": f["date"], "png": _png_uri(np.flipud(f["image"]))}
                  for lab, f in zip(labels, w["frames"])]
    ref = imgs[0].astype(np.float32)
    diffs = [im.astype(np.float32) - ref for im in imgs]
    finite = np.concatenate([np.abs(d).ravel() for d in diffs[1:]]) if len(diffs) > 1 else np.array([1.0])
    lim = float(np.percentile(finite, 99.5)) or 1.0
    out_diff = [{"label": lab, "png": _png_uri(np.flipud(_diff_rgb(d, lim)))} for lab, d in zip(labels, diffs)]
    center = SkyCoord(ra, dec, unit="deg")
    cx, cy = _wise_px(w, center)
    path = _wise_known_path(w, center)
    if path is None:                                                  # неизвестный объект: ведём от последнего кадра
        path = wtrack(w["frames"], cx, cy, start=len(imgs) - 1)
    from photutils.detection import find_peaks
    last = imgs[-1].astype(float)
    med = np.median(last); sig = 1.4826 * np.median(np.abs(last - med)) or 1.0
    pk = find_peaks(last - med, threshold=6 * sig, box_size=5, npeaks=40, border_width=2)
    sources = [] if pk is None else [{"x": round(float(r["x_peak"]) + 0.5, 2), "y": round(float(r["y_peak"]) + 0.5, 2),
                                      "id": f"WISE src {k + 1}"} for k, r in enumerate(pk)]
    return _clean({
        "size_px": int(ny), "pixel_arcsec": w["pixel_arcsec"],
        "center": {"x": round(cx + 0.5, 2), "y": round(cy + 0.5, 2)},
        "frames": out_frames, "diff": out_diff, "sources": sources,
        "target": {"positions": [{"x": round(px + 0.5, 2), "y": round(py + 0.5, 2)} for px, py in path],
                   "molecules": None},
        "meta": {"source": "wise", "survey": "WISE/NEOWISE W2 4.6 µm", "credit": w["credit"],
                 "n_images": len(imgs), "cadence": "pass", "warnings": []},
    })


def _wise_track(ra: float, dec: float, fra: float, fdec: float, fov: float, frame: int) -> dict:
    from wise import track as wtrack
    w = _wise(fra, fdec, fov)
    col, row = _wise_px(w, SkyCoord(ra, dec, unit="deg"))
    start = min(frame, len(w["frames"]) - 1)
    path = wtrack(w["frames"], col, row, start=start)
    c = _wise_sky(w, *path[start])
    return {"ra": round(float(c.ra.deg), 6), "dec": round(float(c.dec.deg), 6),
            "positions": [{"x": round(px + 0.5, 2), "y": round(py + 0.5, 2)} for px, py in path]}


@app.get("/api/track")
def track(ra: Annotated[float, Query(ge=0, le=360, description="RA of the selected object, degrees (0–360).")],
          dec: Annotated[float, Query(ge=-90, le=90, description="Dec of the selected object, degrees (−90…+90).")],
          fra: Annotated[float, Query(ge=0, le=360, description="RA of the field centre, degrees (0–360).")],
          fdec: Annotated[float, Query(ge=-90, le=90, description="Dec of the field centre, degrees (−90…+90).")],
          fov: FOV = 0.1, detector: DETECTOR = 4, cadence: CADENCE = "pass", source: SOURCE = "spherex",
          frame: Annotated[int, Query(ge=0, description="Frame the user clicked on (WISE only: the spot is followed from there).")] = 0,
          snap: Annotated[bool, Query(description="Snap to the nearest source (default). 0 = use the position as given (moving candidates).")] = True):
    """Path of a selected object across the field's frames (fra, fdec, fov, detector), for Trail and lenses.
    With `source=wise` the path is measured on the WISE/NEOWISE frames.

    `ra`, `dec`: the selected object's position (the site sends the dot's centre, not the click);
    it snaps to the nearest source unless `snap=0` (moving candidates).
    `positions` are in the same order as `frames` in `/api/frames` with the same parameters.
    """
    if source == "wise":
        return _wise_track(ra, dec, fra, fdec, fov, frame)
    try:
        sky = _load_sky(fra, fdec, fov, cadence, detector)
    except SkyUnavailable as e:
        raise HTTPException(503, str(e))
    coord = SkyCoord(ra, dec, unit="deg")
    if snap:
        coord = _snap(sky, coord)
    imgs = [f[2] for _, f in _det_items(sky, detector)]
    return {"ra": round(float(coord.ra.deg), 6), "dec": round(float(coord.dec.deg), 6),
            "positions": _track(sky, imgs, coord)}


_ANALYSES: "OrderedDict[tuple, dict]" = OrderedDict()
TARGET_NAMES_EN: dict[str, str] = {}   # названия тестовых объектов в targets.py уже английские


def _mol_snr(a: dict) -> dict | None:
    """Significance of each molecule index (index / error); None per molecule = not measured."""
    from spectrum import molecule_snr
    sp = a.get("spectrum") or {}
    try:
        return molecule_snr(sp.get("wavelength") or [], sp.get("flux") or [], sp.get("error") or [])
    except Exception:
        return None


def _analysis(sky: dict, coord: SkyCoord, online: bool = False) -> dict:
    key = (id(sky), round(coord.ra.deg, 5), round(coord.dec.deg, 5), online)
    a = _ANALYSES.get(key)
    if a is None:                             # requests and the precache thread both come here
        a = _ANALYSES[key] = analyze(sky, coord=coord, online=online)
        while len(_ANALYSES) > 64:
            try:
                _ANALYSES.popitem(last=False)
            except KeyError:
                break
    return a


def _snap(sky: dict, coord: SkyCoord, radius_px: int = 2) -> SkyCoord:
    """Притянуть клик к центру ближайшего источника (в кадре, где источник виден лучше всего)."""
    x, y = (float(v) for v in sky["wcs"].world_to_pixel(coord))
    best, best_peak = None, -np.inf
    for det, items in sky["by_det"].items():
        img = items[-1][1]
        ny, nx = img.shape
        ix, iy = int(round(x)), int(round(y))
        if not (radius_px + 3 <= ix < nx - radius_px - 3 and radius_px + 3 <= iy < ny - radius_px - 3):
            continue
        cut = img[iy - radius_px:iy + radius_px + 1, ix - radius_px:ix + radius_px + 1]
        v = img[np.isfinite(img)]
        noise = 1.4826 * np.median(np.abs(v - np.median(v))) if v.size else np.inf
        peak = (np.nanmax(fill_saturated(cut)) - np.median(v)) / noise if np.isfinite(cut).any() else -np.inf
        if peak > best_peak:
            best, best_peak = (det, img), peak
    if best is None or best_peak < 5:
        return coord
    c = _centroid(best[1], x, y, search=radius_px)
    return sky["wcs"].pixel_to_world(*c) if c is not None else coord


_OBJECTS: "OrderedDict[tuple, dict]" = OrderedDict()   # готовые карточки: (поле, x, y) -> ответ
_SAME_OBJECT_PX = 1.5   # клики рядом притягиваются к центру чуть по-разному; ближе 1.5 пкс (9″) SPHEREx не разделяет


def _cached_object(sky: dict, x: float, y: float):
    with _LOCK:
        for k, v in reversed(_OBJECTS.items()):
            if k[0] == id(sky) and (k[1] - x) ** 2 + (k[2] - y) ** 2 <= _SAME_OBJECT_PX ** 2:
                _OBJECTS.move_to_end(k)
                return v
    return None


# ---------- каталоги: вне запроса ----------
# SIMBAD (and Gaia through IRSA) can take seconds, or tens of seconds when a service is slow. The lookup of
# one object runs in a small pool, once (a second click joins the same lookup); the click waits for it
# CATALOG_WAIT_S and otherwise returns the card with the spectrum and catalog_status "pending".
CATALOG_WAIT_S = float(os.environ.get("SKYWATCH_CATALOG_WAIT", "6"))
CATALOG_RETRY_S = 60.0     # "SIMBAD did not answer" is asked again after this long
_CAT_POOL = ThreadPoolExecutor(max_workers=3, thread_name_prefix="catalog")
_CAT_LOCK = threading.Lock()
_CAT_JOBS: dict = {}                                   # (поле, x, y) -> Future, только пока идёт
_CAT_DONE: "OrderedDict[tuple, tuple]" = OrderedDict()  # (поле, x, y) -> (ответ каталогов, время)
_CAT_NOTE = {"pending": "catalogue lookup still running", "unavailable": "catalogues not answering right now"}


def _catalog_job(sky: dict, coord: SkyCoord, okey: tuple) -> dict:
    try:
        cat = None
        if _gaia_key(sky) in _GAIA_THREADS:           # Gaia of the field is on its way: it goes first, as before
            cat = match_gaia(coord, _gaia_for(sky, wait_s=25.0))
        if cat is None:
            cat = _check_catalogs(coord, online=True, gaia=not _gaia_down())
    except Exception:
        cat = dict(_check_catalogs(coord, online=False), unavailable=True)
    with _CAT_LOCK:
        _CAT_DONE[okey] = (cat, time.time())
        while len(_CAT_DONE) > 256:
            _CAT_DONE.popitem(last=False)
        _CAT_JOBS.pop(okey, None)
    return cat


def _catalog(sky: dict, coord: SkyCoord, okey: tuple, wait_s: float) -> tuple[dict, str]:
    """Catalogue answer for an object and its state: "ok" (final), "pending" (still running after wait_s),
    "unavailable" (SIMBAD did not answer; asked again in a minute)."""
    with _CAT_LOCK:
        done = _CAT_DONE.get(okey)
        if done is not None and done[0].get("unavailable") and time.time() - done[1] > CATALOG_RETRY_S:
            done = None
        fut = None
        if done is None:
            fut = _CAT_JOBS.get(okey)
            if fut is None:
                fut = _CAT_JOBS[okey] = _CAT_POOL.submit(_catalog_job, sky, coord, okey)
    if fut is not None:
        try:
            cat = fut.result(timeout=max(wait_s, 0.0))
        except FutureTimeout:
            return _check_catalogs(coord, online=False), "pending"
        except Exception:
            return _check_catalogs(coord, online=False), "unavailable"
    else:
        cat = done[0]
    return cat, ("unavailable" if cat.get("unavailable") else "ok")


# ---------- карточки заранее ----------
# As soon as /api/frames has a field, one background thread computes the cards students click first: the
# centre object and the brightest dots of the frame on screen. A click then finds the card ready; a click
# that comes earlier computes its own card and shares the catalogue lookup (_CAT_JOBS).
PRECACHE_N = int(os.environ.get("SKYWATCH_PRECACHE", "6"))    # brightest sources per field; 0 = off
PRECACHE_BUDGET_S = 60.0
_PRE_Q: deque = deque(maxlen=8)                               # fields waiting; when full the oldest is dropped
_PRE_SEEN: "OrderedDict[tuple, bool]" = OrderedDict()         # (поле, детектор, размер) уже в очереди или готово
_PRE_CV = threading.Condition()
_PRE_THREAD: threading.Thread | None = None


def _precache(shown: dict, img: np.ndarray, detector: int) -> None:
    """Queue the cards of a field just sent to the site. Never blocks and never raises."""
    global _PRE_THREAD
    if PRECACHE_N <= 0:
        return
    try:
        c = shown["coord"]
        sky = _sky_containing(c.ra.deg, c.dec.deg)    # the sky a click in this field will resolve to
        if sky is None:
            return
        key = (id(sky), detector, shown["shape"][0])
        with _PRE_CV:
            if key in _PRE_SEEN:
                return
            _PRE_SEEN[key] = True
            while len(_PRE_SEEN) > 256:
                _PRE_SEEN.popitem(last=False)
            _PRE_Q.append((sky, shown, img))
            if _PRE_THREAD is None or not _PRE_THREAD.is_alive():
                _PRE_THREAD = threading.Thread(target=_precache_worker, name="precache", daemon=True)
                _PRE_THREAD.start()
            _PRE_CV.notify()
    except Exception:
        pass


def _precache_worker() -> None:
    while True:
        with _PRE_CV:
            while not _PRE_Q:
                _PRE_CV.wait()
            sky, shown, img = _PRE_Q.pop()            # newest first: that is the field on someone's screen now
        try:
            t0 = time.time()
            ny = img.shape[0]
            pts = [shown["coord"]] + [shown["wcs"].pixel_to_world(s["x"] - 0.5, ny - s["y"] - 0.5)
                                      for s in _sources(img, n=PRECACHE_N)]
            for c in pts:
                with _LOCK:
                    alive = any(s is sky for s in _SKIES.values())
                if not alive or time.time() - t0 > PRECACHE_BUDGET_S:
                    break                             # field evicted, or the catalogues are crawling
                out = _object(float(c.ra.deg), float(c.dec.deg), wait_s=30.0, sky=sky)
                if out.get("catalog_status") == "unavailable":
                    break                             # SIMBAD is not answering: don't queue up behind it
        except Exception as e:
            print(f"[precache] skipped: {e!r}", flush=True)
        finally:
            sky = shown = img = None                  # don't pin an evicted field while idle


@app.get("/api/object")
def obj(ra: Annotated[float, Query(ge=0, le=360, description="RA of the selected object, degrees (0–360).")],
        dec: Annotated[float, Query(ge=-90, le=90, description="Dec of the selected object, degrees (−90…+90).")],
        snap: Annotated[bool, Query(description="Snap to the nearest source (default). 0 = use the position as given (moving candidates).")] = True):
    """Object card ("Who is it?") for a selected point: motion, brightness variability, spectrum, molecules,
    catalogue match (test targets, then Gaia DR3, then SIMBAD) and a plain-English verdict.

    The point snaps to the nearest source unless `snap=0`. The card comes back within a few seconds: if the
    catalogues are slower than that, `catalog_status` is `"pending"` (ask again, the lookup keeps running) or
    `"unavailable"` (SIMBAD did not answer); `"ok"` means the catalogue answer is final.
    """
    return _object(ra, dec, snap, CATALOG_WAIT_S)


def _object(ra: float, dec: float, snap: bool = True, wait_s: float = 6.0, sky: dict | None = None) -> dict:
    """The card itself. wait_s — how long to wait for the catalogue lookup (it runs in _CAT_POOL either way);
    sky — the field, when the caller already has it (background precache)."""
    coord = SkyCoord(ra, dec, unit="deg")
    if sky is None:
        sky = _sky_containing(ra, dec)
    if sky is None:
        try:
            sky = _load_sky(ra, dec, 0.1)
        except SkyUnavailable as e:
            raise HTTPException(503, str(e))

    if snap:                                          # snap=0: движущийся кандидат, не тянуть к звезде рядом
        coord = _snap(sky, coord)                     # клик -> центр ближайшего источника
    ox, oy = (float(v) for v in sky["wcs"].world_to_pixel(coord))
    okey = (id(sky), round(ox, 2), round(oy, 2))
    hit = _cached_object(sky, ox, oy) if snap else None   # повторный клик / прогретый объект: без SIMBAD
    if hit is not None:
        return hit
    a = _analysis(sky, coord, online=False)          # движение, спектр (быстро, без сети)
    card, m, v = a["card"], a["motion"], a["variability"]
    cat_status = "ok"
    if not card["catalog_found"]:
        # каталоги: Gaia поля (загружается в фоне при запросе кадров) -> SIMBAD (зеркало)
        cat = match_gaia(coord, _gaia_for(sky))
        if cat is None:
            # the network part runs in the background; a mover waits longer: "not in catalogues" would
            # call it a candidate, and that had better be true
            cat, cat_status = _catalog(sky, coord, okey, max(wait_s, 20.0) if card["moving"] else wait_s)
        card = who_is_it(coord, m, a["scores"], v, cat=cat)
        if cat_status != "ok":
            card["verdict"] = card["verdict"].replace(
                "not found in catalogues", _CAT_NOTE[cat_status]).replace(
                ": not in catalogues. CANDIDATE, needs checking", f": {_CAT_NOTE[cat_status]}")
    fl = [f for f in v.get("fluxes", []) if f > 0]
    ratio = round(max(fl) / min(fl), 2) if len(fl) >= 2 else 1.0

    found = card["catalog_found"]
    named = found and card.get("catalog_source") != "Gaia DR3"   # тестовый объект или SIMBAD
    if card["moving"]:
        status = "known" if found else "candidate"
    elif named and (card.get("catalog_source", "").startswith("targets") or v.get("is_variable")):
        status = "known"      # тестовые объекты и известные переменные
    else:
        status = "star"       # неподвижная фоновая звезда (Gaia / SIMBAD / без совпадения)
    catalog = None
    if found:
        src = card.get("catalog_source") or ""
        if src.startswith("targets"):
            from targets import TARGETS
            t = next((t for t in TARGETS.values() if t["title"] == card["name"]), {})
            catalog = t.get("catalog_en")
            name_en = t.get("simbad")
        if not catalog:
            parts = [src, card.get("catalog_id") or card["name"], card.get("catalog_type") or ""]
            catalog = " · ".join(p for p in parts if p)

    spd = card["speed_arcsec_yr"]
    out = _clean({
        "name": (TARGET_NAMES_EN.get(card["name"], card["name"])) if found else None,
        "status": status,
        "catalog": catalog,
        "motion": {
            "arcsec_per_yr": round(float(spd), 2),
            "arcsec_per_yr_err": card["speed_err_arcsec_yr"],
            "pa_deg": round(float(card["direction_deg"]), 1),
            "catalog_arcsec_per_yr": card.get("catalog_pm_arcsec_yr"),
            "significant": card["moving"],
        },
        "variability": {"ratio": ratio, "label": v.get("label"), "is_variable": v.get("is_variable", False)},
        "molecules": a["scores"] or {"water": 0.0, "methane": 0.0, "co2": 0.0, "co": 0.0},
        "molecules_snr": _mol_snr(a),
        "spectrum": a["spectrum"],
        "verdict": card["verdict"],
        "simbad_url": card["simbad_url"],
        "catalog_status": cat_status,
    })
    with _LOCK:
        if snap and cat_status == "ok":       # an unfinished lookup is not remembered: the next click completes it
            _OBJECTS[okey] = out
        while len(_OBJECTS) > 256:
            _OBJECTS.popitem(last=False)
    return out


# ---------- прогрев после запуска ----------
# Демо-объекты сайта (кейсы и кнопки «Try»): поле 0.1°, детектор 4, кадры раз в проход — ровно то, что
# запрашивает сайт. Они грузятся в память первыми; собранные поля лежат на томе (skycache), так что после
# деплоя это секунды, а не минуты. Потом в фоне на диск сохраняются поля «Surprise me» (без памяти).
# SKYWATCH_WARM=0 — не прогревать.
WARM_FIELDS = [
    ("Barnard's Star", 269.4463, 4.76783),
    ("WISE 0855-0714", 133.75977, -7.24218),
    ("Epsilon Indi B", 331.09639, -56.80057),
    ("RW Aurigae", 76.95609, 30.40113),
    ("R Draconis", 248.16723, 66.75502),
    ("Near Luhman 16", 162.328, -53.319),
    ("CRBR 2422.8-3423", 246.8525, -24.6842),
]
from famous import FAMOUS as FAMOUS_FIELDS   # знаменитые объекты: прогреваются и защищены от очистки диска
_WARM: dict = {"state": "idle"}


def _surprise_fields() -> list[tuple[str, float, float]]:
    """Поля «Surprise me» прямо из web/index.html (один список на сайт и сервер)."""
    import re
    try:
        html = (WEB / "index.html").read_text(encoding="utf-8")
        block = html[html.index("const SURPRISES"):]
        block = block[:block.index("];")]
    except (OSError, ValueError):
        return []
    out = []
    for m in re.finditer(r'name:"([^"]+)"[^}]*?ra:(-?[\d.]+), dec:(-?[\d.]+)', block):
        out.append((m.group(1), float(m.group(2)), float(m.group(3))))
    return out


def _warm() -> None:
    import time
    t_all = time.time()
    _WARM.update(state="running", demo_total=len(WARM_FIELDS), demo_done=0, started=round(t_all))
    print(f"[warm] preloading {len(WARM_FIELDS)} demo fields", flush=True)
    for name, ra, dec in WARM_FIELDS:
        t = time.time()
        try:
            frames(ra, dec, fov=0.1, detector=4, mode="blink", cadence="pass")
            obj(ra, dec)                                    # карточка центрального объекта (+ Gaia)
            lens(ra, dec, fov=0.1, ora=None, odec=None)     # карты молекул
            print(f"[warm] {name}: {time.time() - t:.1f} s", flush=True)
        except Exception as e:                              # нет кэша и архив недоступен — не страшно
            print(f"[warm] {name}: skipped ({e})", flush=True)
        _WARM["demo_done"] += 1
    _WARM.update(state="demo_ready", demo_seconds=round(time.time() - t_all))
    print(f"[warm] demo fields ready in {time.time() - t_all:.0f} s", flush=True)

    extra = [f for f in _surprise_fields() + list(FAMOUS_FIELDS)
             if all(abs(f[1] - w[1]) > 1e-3 or abs(f[2] - w[2]) > 1e-3 for w in WARM_FIELDS)]
    _WARM.update(surprise_total=len(extra), surprise_done=0, surprise_failed=0)
    todo, rnd = list(extra), 0
    while todo:                                         # IRSA can be down for hours: retry the rest every 10 min
        rnd += 1
        failed = []
        for name, ra, dec in todo:
            t = time.time()
            try:
                sky = _load_sky(ra, dec, 0.1, remember=False)    # на диск (и Gaia), память не трогаем
                _READY_SURPRISES.add(name)
                print(f"[warm] surprise {name}: {time.time() - t:.1f} s ({sky['n_loaded']} images)", flush=True)
            except Exception as e:
                failed.append((name, ra, dec))
                print(f"[warm] surprise {name}: skipped ({str(e)[:120]})", flush=True)
            if rnd == 1:
                _WARM["surprise_done"] += 1
        _WARM.update(surprise_failed=len(failed), surprise_ready=len(_READY_SURPRISES))
        if rnd == 1:
            _WARM.update(state="done", total_seconds=round(time.time() - t_all))
            print(f"[warm] done in {time.time() - t_all:.0f} s; {len(failed)} surprise fields will be retried", flush=True)
        todo = failed
        if todo:
            time.sleep(600)


_READY_SURPRISES: set = set()


@app.get("/api/surprises")
def surprises():
    """Names of the "Surprise me" fields the server can show right now (built and saved). The site picks
    only from these, so a dead archive never lands a visitor on a field that can't load."""
    return {"ready": sorted(_READY_SURPRISES), "warm": _WARM.get("state")}


def _ready_fields() -> list[tuple[float, float, float]]:
    """(ra, dec, fov) of every per-pass SPHEREx field this server can show without the archive:
    held in memory or saved on disk (skycache). Only looks at keys and file names, loads nothing."""
    import time
    with _LOCK:
        keys = [k for k in _SKIES if len(k) == 5 and k[3] == "pass"]
    out = [(float(k[0]), float(k[1]), float(k[2])) for k in keys]
    if skycache.ENABLED:
        try:
            now = time.time()
            for f in skycache.DIR.glob(f"*_pass_None_{skycache.CODE_HASH}.pkl"):
                if now - f.stat().st_mtime > skycache.MAX_AGE_S:
                    continue
                try:   # "133p7598_m7p2422_0p1_pass_None_<hash>" -> 133.7598, -7.2422, 0.1 (see skycache._path)
                    out.append(tuple(float(x.replace("m", "-").replace("p", ".")) for x in f.stem.split("_")[:3]))
                except ValueError:
                    pass
        except OSError:
            pass
    return out


def _sep_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    import math
    a1, d1, a2, d2 = (math.radians(v) for v in (ra1, dec1, ra2, dec2))
    h = math.sin((d2 - d1) / 2) ** 2 + math.cos(d1) * math.cos(d2) * math.sin((a2 - a1) / 2) ** 2
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(h))))


_NAMED: list = []          # (name, ra, dec) of the demo and "Surprise me" fields, read once


@app.get("/api/nearest")
def nearest(ra: RA, dec: DEC, n: Annotated[int, Query(ge=1, le=10, description="How many fields to return.")] = 3):
    """The nearest SPHEREx fields this server already has built (in memory or on disk), closest first.

    For when the archive is down and a new field can't be built: these open at once. Each item has `ra`,
    `dec`, `fov`, `distance_deg` and a `name` when the field is one of the site's named targets (else null).
    Nothing is built or downloaded here.
    """
    if not _NAMED:
        _NAMED.extend(WARM_FIELDS + _surprise_fields() + list(FAMOUS_FIELDS))
    best: dict = {}
    for fra, fdec, ffov in sorted(_ready_fields(), key=lambda f: abs(f[2] - 0.1)):   # one per position, 0.1° first
        best.setdefault((round(fra, 3), round(fdec, 3)), (fra, fdec, ffov))
    out = []
    for fra, fdec, ffov in best.values():
        name = next((nm for nm, nra, ndec in _NAMED if _sep_deg(fra, fdec, nra, ndec) < 0.01), None)
        out.append({"name": name, "ra": fra, "dec": fdec, "fov": ffov,
                    "distance_deg": round(_sep_deg(ra, dec, fra, fdec), 2)})
    out.sort(key=lambda f: f["distance_deg"])
    return {"fields": out[:n]}


# ---------- name search: "Vega", "M42", "Orion Nebula" -> coordinates ----------
def _norm_name(s: str) -> str:
    import re
    s = s.lower().replace("messier", "m").replace("−", "-")
    return re.sub(r"[^a-z0-9+\-]", "", s)


@app.get("/api/resolve")
def resolve(name: Annotated[str, Query(min_length=1, max_length=80, description="Object name, e.g. Vega, M42, Orion Nebula.")]):
    """Object name -> position. Our own lists first (famous objects, demo fields, "Surprise me"), then SIMBAD.

    Returns `name`, `ra`, `dec` (degrees, ICRS) and `source` ("lumen" or "simbad"); 404 if nobody knows it.
    """
    import re
    q = _norm_name(name)
    if not q:
        raise HTTPException(422, "Type an object name, e.g. Vega or M42.")
    lists = list(FAMOUS_FIELDS) + list(WARM_FIELDS) + _surprise_fields()
    for nm, ra, dec in lists:                                     # exact name or a name in brackets: "(M42)"
        parts = [nm] + re.findall(r"\(([^)]+)\)", nm) + [re.sub(r"\s*\(.*\)", "", nm)]
        if any(_norm_name(p) == q for p in parts):
            return {"name": nm, "ra": ra, "dec": dec, "source": "lumen"}
    if len(q) >= 4:
        for nm, ra, dec in lists:
            if q in _norm_name(nm):
                return {"name": nm, "ra": ra, "dec": dec, "source": "lumen"}
    try:                                                          # SIMBAD (Harvard mirror, as identify.py)
        import requests
        r = requests.get("https://simbad.harvard.edu/simbad/sim-id",
                         params={"Ident": name.strip(), "output.format": "ASCII"}, timeout=10)
        m = re.search(r"Coordinates\(ICRS[^)]*\):\s*(\d+ \d+ [\d.]+)\s+([+-]\d+ \d+ [\d.]+)", r.text)
        if m:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            c = SkyCoord(m.group(1), m.group(2), unit=(u.hourangle, u.deg))
            title = re.search(r"Object\s+(.+?)\s+---", r.text)
            return {"name": title.group(1).strip() if title else name.strip(),
                    "ra": round(c.ra.deg, 5), "dec": round(c.dec.deg, 5), "source": "simbad"}
    except Exception:
        raise HTTPException(503, "The SIMBAD catalogue is not answering. Type the coordinates instead.")
    raise HTTPException(404, f"No object called \"{name.strip()}\" was found. Check the spelling or type the coordinates.")


# ---------- IRSA watchdog: know that NASA's archive is down before a visitor has to wait for it ----------
_IRSA_WATCH: dict = {}


def _irsa_watch() -> None:
    import time
    import requests
    import data
    while True:
        try:                                                      # a real answer to a real query, not just any reply
            r = requests.get("https://irsa.ipac.caltech.edu/TAP/sync", timeout=10,
                             params={"QUERY": "SELECT TOP 1 planeid FROM spherex.plane", "FORMAT": "csv"})
            healthy = r.status_code == 200 and "planeid" in r.text[:200].lower()
        except Exception:
            healthy = False
        if not healthy:                                           # skip IRSA for the next 10 min
            data._tap_down_until = max(data._tap_down_until, time.time() + data.TAP_RETRY_AFTER_S)
        _IRSA_WATCH.update(checked=round(time.time()), healthy=healthy)
        time.sleep(120)


@app.on_event("startup")
def _start_irsa_watch() -> None:
    if os.environ.get("SKYWATCH_IRSA_WATCH", "1") != "0":
        threading.Thread(target=_irsa_watch, name="irsa-watch", daemon=True).start()


@app.on_event("startup")
def _start_warm() -> None:
    if os.environ.get("SKYWATCH_WARM", "1") != "0":
        threading.Thread(target=_warm, name="warm", daemon=True).start()


# ---------- cleanup of old visitor fields (cleanup.py) ----------
# Protected from deletion: every UPPERCASE list of (name, ra, dec) in this file (WARM_FIELDS and any list
# added later), the "Surprise me" fields, seed/queries. _load_sky is wrapped so each served field gets a
# "last opened" record. The job itself runs in its own thread and only acts when the disk is nearly full.
import sys as _sys      # noqa: E402
import cleanup          # noqa: E402

cleanup.protect("main.py lists", _sys.modules[__name__], required=True)
cleanup.protect("surprise", _surprise_fields, required=True)
_load_sky = cleanup.track(_load_sky)


@app.on_event("startup")
def _start_cleanup() -> None:
    cleanup.start()
# ---------- end of cleanup hook ----------
