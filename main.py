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
import threading
import warnings
from collections import OrderedDict

import numpy as np
from astropy.coordinates import SkyCoord
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from sky import SkyUnavailable, analyze, get_sky
from render import fill_saturated, stretch_limits, to_uint8
from identify import _centroid, _check_catalogs, gaia_cone, match_gaia, who_is_it
from data import CACHE
from astropy.table import Table

warnings.filterwarnings("ignore")

app = FastAPI(title="LUMEN Skywatch API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

PIX_ARCSEC = 6.15
DIFF_POS = np.array([0xF6, 0xB5, 0x44], float)   # ярче — янтарный
DIFF_NEG = np.array([0x3D, 0x5B, 0xD9], float)   # темнее — синий
DIFF_ZERO = np.array([0x07, 0x0A, 0x20], float)  # без изменений — фон


WEB = Path(__file__).resolve().parent / "web"


@app.get("/", include_in_schema=False)
def site():
    return FileResponse(WEB / "index.html")


# ---------- кэш наборов кадров ----------

_SKIES: "OrderedDict[tuple, dict]" = OrderedDict()
_LOCK = threading.Lock()
_MAX_SKIES = 12


def _load_sky(ra: float, dec: float, fov: float, cadence: str = "pass", det: int | None = None) -> dict:
    """Кадры для места: сначала полный набор из кэша на диске, иначе быстрый режим.

    cadence="visit" — проходы порезаны на сеансы съёмки (epochs.split_cadence); такие кадры собираются
    только для детектора det: сеансы у каждого детектора свои, и пересобирать все шесть незачем.
    """
    if cadence != "visit":
        cadence, det = "pass", None
    dets = [det] if det else None
    key = (round(ra, 4), round(dec, 4), round(fov, 4), cadence, det)
    with _LOCK:
        if key in _SKIES:
            _SKIES.move_to_end(key)
            return _SKIES[key]
    sky = None
    if fov < 0.1:
        try:
            big = _load_sky(ra, dec, 0.1, cadence, det)
            return _crop_sky(big, fov)
        except SkyUnavailable:
            pass
    try:
        full = get_sky(ra, dec, fov, quick=False, offline=True, detectors=dets, cadence=cadence)
        if full["n_loaded"] >= 0.8 * full["n_wanted"]:
            sky = full
    except SkyUnavailable:
        pass
    if sky is None:
        kw = dict(quick=True, detectors=dets, cadence=cadence)
        try:
            sky = get_sky(ra, dec, fov, **kw)
        except SkyUnavailable:
            sky = get_sky(ra, dec, fov, offline=True, **kw)  # архив лёг — что есть в кэше
    with _LOCK:
        _SKIES[key] = sky
        while len(_SKIES) > _MAX_SKIES:
            _SKIES.popitem(last=False)
    _start_gaia(sky)
    return sky


_GAIA: dict = {}          # ключ поля -> Table | None
_GAIA_THREADS: dict = {}


def _gaia_key(sky: dict) -> str:
    c = sky["coord"]
    return f"{c.ra.deg:.4f}_{c.dec.deg:+.4f}_{sky['size_deg']}"


def _start_gaia(sky: dict) -> None:
    """В фоне один раз загрузить Gaia DR3 для поля (диск-кэш cache/gaia/)."""
    key = _gaia_key(sky)
    if key in _GAIA or key in _GAIA_THREADS:
        return
    path = CACHE / "gaia" / f"{key}.ecsv"
    if path.exists():
        _GAIA[key] = Table.read(path)
        return

    def job():
        try:
            radius = sky["size_deg"] / 2 * 1.45 * 3600  # описанный круг квадрата
            t = gaia_cone(sky["coord"], radius, top=3000, gmax=20.5)
            path.parent.mkdir(parents=True, exist_ok=True)
            t.write(path, overwrite=True)
            _GAIA[key] = t
        except Exception:
            _GAIA[key] = None
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
    t = find_peaks(np.nan_to_num(filled - med), threshold=5 * sig, box_size=5, npeaks=n, border_width=2)
    if t is None:
        return []
    t.sort("peak_value", reverse=True)
    ny = img.shape[0]
    gaia = _gaia_for(sky) if sky is not None else None
    out = []
    for k, r in enumerate(t):
        sid = f"SPHEREx src {k + 1}"
        if gaia is not None:
            c = sky["wcs"].pixel_to_world(float(r["x_peak"]), float(r["y_peak"]))
            m = match_gaia(c, gaia, radius_arcsec=6.0)
            if m:
                sid = m["name"]
        out.append(dict(_to_display(float(r["x_peak"]), float(r["y_peak"]), ny), id=sid))
    return out


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
    return {"ok": True, "cached_fields": len(_SKIES)}


def _det_items(sky: dict, detector: int) -> list:
    """Кадры детектора в том порядке, в каком их показывает /api/frames (без шумных кадров)."""
    items = [(i, f) for i, f in enumerate(sky["frames"]) if f[1] == detector]
    if not items:
        raise HTTPException(404, f"Нет кадров детектора {detector}; есть: {sorted(sky['by_det'])}")
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
def frames(ra: float, dec: float, fov: float = Query(0.1, gt=0.01, le=0.5),
           detector: int = Query(4, ge=1, le=6), mode: str = "blink",
           cadence: str = Query("pass", pattern="^(pass|visit)$")):
    """cadence: "pass" — эпоха раз в полгода; "visit" — каждый проход SPHEREx порезан на сеансы съёмки (дни)."""
    try:
        sky = _load_sky(ra, dec, fov, cadence, detector)
    except SkyUnavailable as e:
        raise HTTPException(503, str(e))

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
    except Exception:
        molecules = None

    return _clean({
        "size_px": int(sky["shape"][0]),
        "pixel_arcsec": PIX_ARCSEC,
        "frames": out_frames,
        "diff": out_diff,
        "sources": _sources(imgs[-1], sky=sky),
        "target": {"positions": positions, "molecules": molecules},
        "meta": {"source": sky["source"], "n_images": sky["n_loaded"], "warnings": sky["warnings"],
                 "wavelength_um": [info[i]["wavelength_um"] for i, _ in items] if info else None,
                 "n_exposures": [info[i]["n_exposures"] for i, _ in items] if info else None,
                 "cadence": cadence},
    })


LENS_RGB = {"water": (0x3D, 0x5B, 0xD9), "methane": (0xF0, 0x7A, 0x3A),
            "co2": (0xF6, 0xB5, 0x44), "co": (0xE0, 0x35, 0x2B)}   # цвета линз сайта


def _lens_png(depth: np.ndarray, rgb) -> str:
    """Карта глубины -> PNG цвета молекулы с прозрачностью: от 0.08 (едва) до 0.43+ (полностью)."""
    a = np.clip((np.nan_to_num(depth) - 0.08) / 0.35, 0, 1)
    img = np.zeros(depth.shape + (4,), np.uint8)
    img[..., :3] = rgb
    img[..., 3] = (a * 235).astype(np.uint8)
    return _png_uri(img)


@app.get("/api/lens")
def lens(ra: float, dec: float, fov: float = Query(0.1, gt=0.01, le=0.5)):
    """Карты молекул поля: где на картинке поглощение воды, метана, CO₂, CO сильнее, чем у соседей.

    Та же сетка и ориентация, что у кадров /api/frames с теми же ra, dec, fov.
    maps — PNG с прозрачностью для наложения; peak — самая глубокая полоса в поле (0..1);
    lit — сколько пикселей заметно светится.
    """
    from lens import molecule_maps, paint_object
    try:
        sky = _load_sky(ra, dec, fov)
    except SkyUnavailable as e:
        raise HTTPException(503, str(e))
    if "_lens" not in sky:   # считается один раз на поле и живёт вместе с ним в кэше _SKIES
        maps = molecule_maps(sky)
        # объект в центре поля: если он слишком тусклый для попиксельной карты (WISE 0855),
        # его пятно закрашивается по индексам его спектра — тем же, что в карточке /api/object
        try:
            c = _snap(sky, sky["coord"])
            scores = _analysis(sky, c)["scores"] or {}
            if any(v > 0 for v in scores.values()):
                x, y = (float(v) for v in sky["wcs"].world_to_pixel(c))
                maps = paint_object(maps, x, y, scores)
        except Exception:
            pass
        sky["_lens"] = _clean({
            "size_px": int(sky["shape"][0]),
            "maps": {k: _lens_png(v, LENS_RGB[k]) for k, v in maps.items()},
            "peak": {k: round(float(np.nanmax(v)), 3) if np.isfinite(v).any() else 0.0 for k, v in maps.items()},
            "lit": {k: int(np.sum(np.nan_to_num(v) >= 0.2)) for k, v in maps.items()},
        })
    return sky["_lens"]


@app.get("/api/track")
def track(ra: float, dec: float, fra: float, fdec: float, fov: float = Query(0.1, gt=0.01, le=0.5),
          detector: int = Query(4, ge=1, le=6), cadence: str = Query("pass", pattern="^(pass|visit)$")):
    """Путь выбранного объекта по кадрам поля (fra, fdec, fov, detector) — для «Trail» и линз.

    ra, dec — куда кликнули; клик притягивается к ближайшему источнику.
    positions идут в том же порядке, что frames в /api/frames с теми же параметрами.
    """
    try:
        sky = _load_sky(fra, fdec, fov, cadence, detector)
    except SkyUnavailable as e:
        raise HTTPException(503, str(e))
    coord = _snap(sky, SkyCoord(ra, dec, unit="deg"))
    imgs = [f[2] for _, f in _det_items(sky, detector)]
    return {"ra": round(float(coord.ra.deg), 6), "dec": round(float(coord.dec.deg), 6),
            "positions": _track(sky, imgs, coord)}


_ANALYSES: "OrderedDict[tuple, dict]" = OrderedDict()
TARGET_NAMES_EN = {"Звезда Барнарда": "Barnard's Star", "Эпсилон Индейца B": "Epsilon Indi B",
                   "WISE 0855−0714": "WISE 0855−0714", "RW Возничего": "RW Aurigae", "R Дракона": "R Draconis"}


def _analysis(sky: dict, coord: SkyCoord, online: bool = False) -> dict:
    key = (id(sky), round(coord.ra.deg, 5), round(coord.dec.deg, 5), online)
    if key not in _ANALYSES:
        _ANALYSES[key] = analyze(sky, coord=coord, online=online)
        while len(_ANALYSES) > 64:
            _ANALYSES.popitem(last=False)
    return _ANALYSES[key]


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


@app.get("/api/object")
def obj(ra: float, dec: float):
    coord = SkyCoord(ra, dec, unit="deg")
    sky = _sky_containing(ra, dec)
    if sky is None:
        try:
            sky = _load_sky(ra, dec, 0.1)
        except SkyUnavailable as e:
            raise HTTPException(503, str(e))

    coord = _snap(sky, coord)                         # клик -> центр ближайшего источника
    a = _analysis(sky, coord, online=False)          # движение, спектр (быстро, без сети)
    card, m, v = a["card"], a["motion"], a["variability"]
    if not card["catalog_found"]:
        # каталоги: Gaia поля (загружается в фоне при запросе кадров) -> SIMBAD (зеркало)
        cat = match_gaia(coord, _gaia_for(sky, wait_s=25.0))
        if cat is None:
            cat = _check_catalogs(coord, online=True)
        card = who_is_it(coord, m, a["scores"], v, cat=cat)
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
    return _clean({
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
        "spectrum": a["spectrum"],
        "verdict": card["verdict"],
        "simbad_url": card["simbad_url"],
    })
