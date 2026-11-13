"""Кадры для мигания (Питонист 1).

    build_frames(cutouts, coord, size_deg) -> [(дата эпохи, детектор, 2D-кадр)]

Все кадры лежат на одной сетке make_target_wcs(coord, size_deg): север вверх, восток влево,
пиксель 6,15″, центр — coord. Другие модули могут вызвать make_target_wcs с теми же
аргументами и получить ту же WCS (например, для centroid_sky).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
from reproject import reproject_interp

from data import bad_pixels

PIX_ARCSEC = 6.15
EPOCH_GAP_DAYS = 30  # SPHEREx проходит небо раз в полгода; разрыв > 30 дней = новая эпоха


def split_epochs(times, gap_days: float = EPOCH_GAP_DAYS) -> np.ndarray:
    """MJD -> номер эпохи (0, 1, 2…). Новая эпоха — когда разрыв больше gap_days."""
    times = np.asarray(times, float)
    order = np.argsort(times)
    labels = np.empty(len(times), int)
    ep = 0
    for i, idx in enumerate(order):
        if i and times[idx] - times[order[i - 1]] > gap_days:
            ep += 1
        labels[idx] = ep
    return labels


def make_target_wcs(coord: SkyCoord, size_deg: float, pix_arcsec: float = PIX_ARCSEC):
    """Общая сетка вокруг coord. Возвращает (wcs, shape)."""
    n = int(round(size_deg * 3600 / pix_arcsec))
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [coord.icrs.ra.deg, coord.icrs.dec.deg]
    w.wcs.crpix = [(n + 1) / 2, (n + 1) / 2]  # центр массива (FITS считает с 1)
    w.wcs.cdelt = [-pix_arcsec / 3600, pix_arcsec / 3600]
    return w, (n, n)


def clean_image(cutout: dict) -> np.ndarray:
    """Снимок без зодиакального света и остаточного фона, плохие пиксели = NaN.

    Фон между эпохами разный (зодиакальный свет зависит от времени года), без вычитания
    кадры разных эпох будут отличаться яркостью целиком.
    """
    img = cutout["image"].astype(np.float32).copy()
    if cutout.get("zodi") is not None:
        img -= cutout["zodi"]
    img[bad_pixels(cutout)] = np.nan
    img -= np.nanmedian(img)
    return img


def stack(cutouts: list[dict], target_wcs: WCS, shape) -> np.ndarray:
    """Перепроецировать вырезки на общую сетку и взять медиану по пикселям."""
    layers = []
    for c in cutouts:
        arr, foot = reproject_interp((clean_image(c), c["wcs"]), target_wcs, shape_out=shape)
        arr[foot == 0] = np.nan
        layers.append(arr)
    if not layers:
        return np.full(shape, np.nan, np.float32)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # All-NaN slice
            return np.nanmedian(np.stack(layers), axis=0).astype(np.float32)


def build_frames(cutouts: list[dict], coord: SkyCoord, size_deg: float = 0.1,
                 detectors=None) -> list[tuple[str, int, np.ndarray]]:
    """Список вырезок -> кадры [(дата эпохи 'YYYY-MM-DD', детектор, 2D-массив)].

    Один кадр = медиана всех экспозиций данного детектора в данной эпохе.
    Мигать можно только кадрами ОДНОГО детектора (у разных — разные длины волн).
    Сортировка: по детектору, внутри — по дате.
    """
    if not cutouts:
        return []
    target_wcs, shape = make_target_wcs(coord, size_deg)
    labels = split_epochs([c["time"] for c in cutouts])

    groups = defaultdict(list)
    for c, ep in zip(cutouts, labels):
        if detectors is None or c["detector"] in detectors:
            groups[(ep, c["detector"])].append(c)

    epoch_date = {}
    for ep in set(labels):
        t = np.median([c["time"] for c, e in zip(cutouts, labels) if e == ep])
        epoch_date[ep] = Time(t, format="mjd").iso[:10]

    frames = []
    for (ep, det), group in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        frames.append((epoch_date[ep], det, stack(group, target_wcs, shape)))
    return frames


def frames_by_detector(frames) -> dict[int, list[tuple[str, np.ndarray]]]:
    """{детектор: [(дата, кадр), ...]} — удобно для интерфейса мигания."""
    out = defaultdict(list)
    for date, det, img in frames:
        out[det].append((date, img))
    return dict(out)
