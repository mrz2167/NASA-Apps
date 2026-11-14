"""Кадры WISE/NEOWISE 2010–2024 для любой точки неба — для «Trail» на 14 лет.

    get_wise(ra, dec, fov) -> {"frames": [{date, mjd, image (uint8 2D, строки сверху вниз)}],
                               "wcs": WCS кадра, "shape": (ny, nx), "pixel_arcsec": 2.75, ...}

Источник — кадры unTimely (время-разрешённые коадды unWISE; Meisner, Caselden & Schlafly 2023),
как их режет и растягивает сервис WiseView (Backyard Worlds: Planet 9, byw.tools/wiseview):
сначала запрос к его API (список PNG и привязка кадра), потом сами PNG из открытого бакета AMNH.
Канал W2 (4.6 мкм) — ближе всего к детектору 6 SPHEREx (4.42–5.00 мкм).
window=0: одна эпоха на каждый полугодовой проход WISE, 2010.2 … 2024.2 (обычно 20–23 кадра).

Всё кэшируется в cache/wise/ (на Railway это постоянный диск), повторный запрос — без сети.
PNG уже растянуты по яркости, поэтому годятся для мигания и центроидов, но не для фотометрии.
"""
from __future__ import annotations

import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from astropy.time import Time
from astropy.wcs import WCS
from PIL import Image

from data import CACHE

API = "https://moo23y1au0.execute-api.us-west-2.amazonaws.com/default/png-animation"
BUCKET = "https://amnh-citsci-public.s3-us-west-2.amazonaws.com/"
PIX_ARCSEC = 2.75
MAX_PX = 260                       # WiseView режет до ~0.2°
UA = {"User-Agent": "lumen-skywatch/1.0"}
CREDIT = ("WISE/NEOWISE unTimely coadds (Meisner, Caselden & Schlafly 2023) "
          "via WiseView, Backyard Worlds: Planet 9")


class WiseUnavailable(RuntimeError):
    pass


# WiseView — сторонний сервис. Если он лёг, не ждём его на каждом запросе: после сбоя 5 минут сразу
# отвечаем «временно недоступен» (поля из кэша работают как обычно), потом пробуем снова.
import threading as _threading
import time as _time

API_TIMEOUT_S = 30
DOWN_FOR_S = 300
DOWN_MSG = ("WISE frames are temporarily unavailable: WiseView (a third-party service) isn't answering. "
            "SPHEREx frames still work, and WISE fields opened before are cached.")
_state = {"down_until": 0.0, "last_error": None, "last_error_at": None, "last_ok_at": None}
_state_lock = _threading.Lock()


def _mark_down(err: Exception) -> None:
    with _state_lock:
        _state.update(down_until=_time.time() + DOWN_FOR_S, last_error=str(err)[:200], last_error_at=_time.time())


def _mark_ok() -> None:
    with _state_lock:
        _state.update(down_until=0.0, last_ok_at=_time.time())


def status() -> dict:
    """Для /api/health: доступен ли WiseView по последним запросам."""
    with _state_lock:
        s = dict(_state)
    return {"ok": s["down_until"] <= _time.time(), "last_error": s["last_error"],
            "last_error_at": s["last_error_at"], "last_ok_at": s["last_ok_at"]}


def _key(ra: float, dec: float, size: int, band: int) -> str:
    return hashlib.sha1(f"{ra:.5f}_{dec:+.5f}_{size}_{band}".encode()).hexdigest()[:12]


def _params(ra: float, dec: float, size: int, band: int) -> dict:
    # тот же набор параметров, что шлёт сам WiseView; растяжка — его стандартная
    return dict(ra=f"{ra:.6f}", dec=f"{dec:.6f}", band=band, size=size, max_dyr=0,
                minbright=-50, maxbright=500, invert=0, stretch=1, diff=0, scandir=0, outer=0,
                neowise=0, window=0, diff_window=1, unique=1, smooth_scan=0, shift=0,
                pmx=0, pmy=0, synth_a=0, synth_b=0)


def _wcs(meta: dict) -> WCS:
    """Привязка кадра из ответа WiseView (FITS-пиксели считаются с 1, строки — снизу вверх)."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [meta["CRVAL1"], meta["CRVAL2"]]
    w.wcs.crpix = [meta["CRPIX1"], meta["CRPIX2"]]
    w.wcs.cdelt = [-PIX_ARCSEC / 3600, PIX_ARCSEC / 3600]
    return w


def _fetch_png(key: str, path) -> None:
    r = requests.get(BUCKET + key, headers=UA, timeout=API_TIMEOUT_S)
    r.raise_for_status()
    Image.open(io.BytesIO(r.content)).verify()       # не PNG -> исключение, файл не пишем
    path.write_bytes(r.content)


def _fill_cores(img: np.ndarray) -> np.ndarray:
    """Ядра пересвеченных звёзд WiseView отдаёт пустыми (0) посреди белого пятна: заливаем их белым,
    иначе на месте яркой звезды чёрный квадрат и центроид съезжает."""
    from scipy.ndimage import binary_fill_holes, label
    lab, n = label(img == 0)
    if n == 0:
        return img
    holes = binary_fill_holes(img >= 200) & (img == 0)
    out = img.copy()
    out[holes] = 255
    return out


def get_wise(ra: float, dec: float, fov: float = 0.1, band: int = 2, offline: bool = False) -> dict:
    size = int(min(MAX_PX, max(32, round(fov * 3600 / PIX_ARCSEC))))
    folder = CACHE / "wise" / _key(ra, dec, size, band)
    meta_file = folder / "meta.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
    else:
        if offline:
            raise WiseUnavailable("No cached WISE frames for this spot (offline).")
        if _time.time() < _state["down_until"]:
            raise WiseUnavailable(DOWN_MSG)
        try:
            r = requests.get(API, params=_params(ra, dec, size, band), headers=UA, timeout=API_TIMEOUT_S)
            r.raise_for_status()
            meta = r.json()
        except Exception as e:
            _mark_down(e)
            raise WiseUnavailable(DOWN_MSG) from e
        if not meta.get("ims"):
            raise WiseUnavailable("WiseView returned no WISE frames for this spot.")
        folder.mkdir(parents=True, exist_ok=True)

    paths = [folder / f"{i:02d}.png" for i in range(len(meta["ims"]))]
    todo = [(k, p) for k, p in zip(meta["ims"], paths) if not p.exists()]
    if todo:
        if offline:
            raise WiseUnavailable("Cached WISE frames for this spot are incomplete (offline).")
        if _time.time() < _state["down_until"]:
            raise WiseUnavailable(DOWN_MSG)
        with ThreadPoolExecutor(8) as pool:
            errs = [f.exception() for f in [pool.submit(_fetch_png, k, p) for k, p in todo]]
        if any(errs):
            _mark_down(next(e for e in errs if e))
            raise WiseUnavailable(DOWN_MSG)
        _mark_ok()
    if not meta_file.exists():
        meta_file.write_text(json.dumps(meta))

    frames = []
    for i, (ep, p) in enumerate(zip(meta["epochs"], paths)):
        mjd = float(np.mean([meta["all_mjds"][j] for j in ep]))
        img = _fill_cores(np.array(Image.open(p).convert("L")))
        frames.append(dict(mjd=mjd, date=Time(mjd, format="mjd").iso[:10], image=img))
    frames.sort(key=lambda f: f["mjd"])
    return dict(frames=frames, wcs=_wcs(meta), shape=frames[0]["image"].shape, size_deg=size * PIX_ARCSEC / 3600,
                pixel_arcsec=PIX_ARCSEC, band=band, credit=CREDIT, ra=ra, dec=dec)


def centroid(img: np.ndarray, x: float, y: float, search: int = 3) -> tuple[float, float] | None:
    """Центр яркого пятна около (x, y) (x — столбец, y — строка сверху вниз); None — пятна нет.

    Яркие звёзды в PNG WiseView упираются в 255 плоским пятном: тогда центр — середина всего
    плоского пятна, а не его первый пиксель.
    """
    from scipy.ndimage import label
    ny, nx = img.shape
    ix, iy = int(round(x)), int(round(y))
    x0, x1, y0, y1 = max(0, ix - search), min(nx, ix + search + 1), max(0, iy - search), min(ny, iy + search + 1)
    cut = img[y0:y1, x0:x1].astype(float)
    if cut.size == 0:
        return None
    bg = float(np.median(img))
    noise = 1.4826 * float(np.median(np.abs(img - bg))) or 1.0
    py, px = np.unravel_index(np.argmax(cut), cut.shape)
    if cut[py, px] - bg < 5 * noise:
        return None
    cy, cx = py + y0, px + x0
    if img[cy, cx] >= 250:                                   # пересвет: середина плоского пятна
        lab, _ = label(img >= 250)
        yy, xx = np.nonzero(lab == lab[cy, cx])
        return float(xx.mean()), float(yy.mean())
    a0, a1, b0, b1 = max(0, cy - 2), min(ny, cy + 3), max(0, cx - 2), min(nx, cx + 3)
    w = np.clip(img[a0:a1, b0:b1].astype(float) - bg, 0, None)
    if w.sum() <= 0:
        return float(cx), float(cy)
    yy, xx = np.mgrid[a0:a1, b0:b1]
    return float((xx * w).sum() / w.sum()), float((yy * w).sum() / w.sum())


def track(frames: list[dict], x: float, y: float, start: int = 0, step_px: int = 2) -> list[tuple[float, float]]:
    """Путь пятна по кадрам: от кадра start вперёд и назад.

    Каждое следующее положение ищется рядом с предсказанием по уже измеренной скорости (прямая по
    последним точкам), поэтому быстрые звёзды (Барнард ~3.6 пикселя WISE в год) не теряются и через
    перерыв 2011–2013, когда WISE не снимал, а соседние звёзды не перехватывают трек.
    Где пятна не нашлось — ставится предсказание.
    """
    n = len(frames)
    t = np.array([f["mjd"] for f in frames]) / 365.25
    start = min(max(start, 0), n - 1)
    out: list[tuple[float, float] | None] = [None] * n
    out[start] = centroid(frames[start]["image"], x, y, search=step_px + 1) or (x, y)
    for rng in (range(start + 1, n), range(start - 1, -1, -1)):
        done = [start]
        for i in rng:
            k = done[-6:]                                        # скорость по последним точкам
            if len(k) >= 2:
                tt = t[k]
                vx = np.polyfit(tt, [out[j][0] for j in k], 1)[0]
                vy = np.polyfit(tt, [out[j][1] for j in k], 1)[0]
            else:
                vx = vy = 0.0
            last = done[-1]
            dt = t[i] - t[last]
            px, py = out[last][0] + vx * dt, out[last][1] + vy * dt
            # пока скорость не измерена (первый шаг через большой пропуск), ищем шире
            s = step_px + (int(min(6, abs(dt) * 2)) if len(k) < 3 else 0)
            got = centroid(frames[i]["image"], px, py, search=s)
            out[i] = got if got is not None else (px, py)
            done.append(i)
    return [p for p in out]
