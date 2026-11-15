# AI disclosure: parts of this file were written with AI assistance (Claude, Anthropic), then reviewed and tested by the LUMEN team. See README, "AI use".
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


MAX_EPOCH_SPAN_DAYS = 45  # в глубоких полях (у полюсов эклиптики) съёмка идёт непрерывно


def split_epochs(times, gap_days: float = EPOCH_GAP_DAYS,
                 max_span_days: float = MAX_EPOCH_SPAN_DAYS) -> np.ndarray:
    """MJD -> номер эпохи (0, 1, 2…).

    Новая эпоха — когда разрыв больше gap_days. Если съёмка идёт без разрывов
    (глубокие поля, например R Dra у северного полюса эклиптики), эпоха режется
    на куски не длиннее max_span_days.
    """
    times = np.asarray(times, float)
    order = np.argsort(times)
    labels = np.empty(len(times), int)
    ep, start = 0, None
    for i, idx in enumerate(order):
        t = times[idx]
        if i and (t - times[order[i - 1]] > gap_days or t - start > max_span_days):
            ep += 1
            start = None
        if start is None:
            start = t
        labels[idx] = ep
    return labels


VISIT_GAP_HOURS = 12  # cadence="visit": разрыв > 12 ч внутри прохода = новый сеанс съёмки


def split_cadence(times, cadence: str = "pass") -> np.ndarray:
    """MJD -> номер кадра.

    cadence="pass"  — эпоха = полугодовой проход (split_epochs).
    cadence="visit" — каждый проход дополнительно режется на сеансы съёмки: новый кадр, когда между
                      снимками больше VISIT_GAP_HOURS. Один детектор видит точку серией за 1–9 дней
                      (изредка до 40), так что вместо 1 эпохи выходит 3–8 кадров по ~4 снимка.
    Звать для снимков ОДНОГО детектора: у разных детекторов сеансы перемежаются.
    Сеансовые кадры шумнее (меньше снимков), и длина волны точки от сеанса к сеансу чуть разная
    (фильтр SPHEREx переменный). Сдвиг за дни заметен только у быстрых объектов (астероиды, кометы);
    для звёзд с собственным движением нужны проходы.
    """
    times = np.asarray(times, float)
    passes = split_epochs(times)
    if cadence != "visit" or len(times) == 0:
        return passes
    labels = np.empty(len(times), int)
    nxt = -1
    for ep in np.unique(passes):            # номера проходов растут со временем
        idx = np.where(passes == ep)[0]
        idx = idx[np.argsort(times[idx])]
        prev = None
        for i in idx:
            if prev is None or (times[i] - prev) * 24 > VISIT_GAP_HOURS:
                nxt += 1
            labels[i] = nxt
            prev = times[i]
    return labels


def select_exposures(table, per_group: int | None = None, cadence: str = "pass"):
    """Оставить не больше per_group экспозиций на каждую пару (эпоха, детектор),
    равномерно по времени. Быстрый режим: per_group=3 даёт ~50 вырезок вместо сотен.

    Принимает и возвращает таблицу find_images (с теми же столбцами).
    """
    if not per_group or len(table) == 0:
        return table
    labels = split_cadence(table["time"], cadence)
    keep = []
    for ep in np.unique(labels):
        for det in np.unique(table["detector"]):
            idx = np.where((labels == ep) & (table["detector"] == det))[0]
            if len(idx) > per_group:
                idx = idx[np.argsort(table["time"][idx])]
                idx = idx[np.linspace(0, len(idx) - 1, per_group).round().astype(int)]
            keep.extend(idx.tolist())
    out = table[sorted(keep)]
    return out


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
                 detectors=None, min_exposures: int = 2,
                 info: list | None = None, cadence: str = "pass") -> list[tuple[str, int, np.ndarray]]:
    """Список вырезок -> кадры [(дата эпохи 'YYYY-MM-DD', детектор, 2D-массив)].

    Один кадр = медиана всех экспозиций данного детектора в данной эпохе.
    Мигать можно только кадрами ОДНОГО детектора (у разных — разные длины волн).
    Кадры, собранные меньше чем из min_exposures экспозиций, пропускаются (шумные хвосты эпох).
    Сортировка: по детектору, внутри — по дате.

    info: если передать пустой список, он заполнится словарями (по одному на кадр, в том же
    порядке): date, detector, n_exposures, wavelength_um (длина волны в точке coord, медиана
    по экспозициям), mjd_min, mjd_max — для подписей в интерфейсе.
    """
    if not cutouts:
        return []
    target_wcs, shape = make_target_wcs(coord, size_deg)
    labels = split_cadence([c["time"] for c in cutouts], cadence)

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
        if len(group) < min_exposures:
            continue
        frames.append((epoch_date[ep], det, stack(group, target_wcs, shape)))
        if info is not None:
            info.append(dict(
                date=epoch_date[ep], detector=det, n_exposures=len(group),
                wavelength_um=_wavelength_at(group, coord),
                mjd_min=min(c["time"] for c in group), mjd_max=max(c["time"] for c in group),
            ))
    return frames


def _wavelength_at(cutouts: list[dict], coord: SkyCoord) -> float:
    """Медианная длина волны (мкм), на которой экспозиции видели точку coord."""
    wls = []
    for c in cutouts:
        try:
            x, y = c["wcs"].world_to_pixel(coord)
            ny, nx = c["image"].shape
            if 0 <= x < nx and 0 <= y < ny:
                wl, _ = c["spectral_wcs"].pixel_to_world(x, y)
                wls.append(wl.to_value("um"))
        except Exception:
            pass
    return float(np.median(wls)) if wls else float("nan")


def _flux_scale(img: np.ndarray, ref: np.ndarray) -> float:
    """Коэффициент k, при котором k·ref лучше всего совпадает с img на ярких пикселях."""
    ok = np.isfinite(img) & np.isfinite(ref)
    if ok.sum() < 20:
        return 1.0
    bright = ok & (ref > np.nanpercentile(ref[ok], 95))
    k = np.sum(img[bright] * ref[bright]) / np.sum(ref[bright] ** 2)
    return float(k) if np.isfinite(k) and k > 0 else 1.0


def diff_frames(frames, reference: str = "first", match_flux: bool = True) -> list[tuple[str, int, np.ndarray]]:
    """Режим «что изменилось»: кадр минус опорный кадр того же детектора.

    reference="first" — вычитаем самую раннюю эпоху; "previous" — предыдущую.
    Неподвижные звёзды исчезают; движущийся объект даёт пару «светлое пятно (где сейчас) +
    тёмное (где был)», переменная звезда — светлое или тёмное пятно на месте.
    match_flux: подогнать общую яркость опорного кадра (внутри детектора длина волны в разных
    эпохах чуть разная, и все звёзды целиком ярче/тусклее). Слабые остатки на ярких звёздах
    (красно-синие пары) — от сдвигов меньше пикселя, это нормально.
    Опорный кадр в результат не входит.
    """
    out = []
    for det, items in frames_by_detector(frames).items():
        for i in range(1, len(items)):
            ref = items[0][1] if reference == "first" else items[i - 1][1]
            date, img = items[i]
            k = _flux_scale(img, ref) if match_flux else 1.0
            out.append((date, det, (img - k * ref).astype(np.float32)))
    return out


def frames_by_detector(frames) -> dict[int, list[tuple[str, np.ndarray]]]:
    """{детектор: [(дата, кадр), ...]} — удобно для интерфейса мигания."""
    out = defaultdict(list)
    for date, det, img in frames:
        out[det].append((date, img))
    return dict(out)


def good_frames(frames, info, frac: float = 0.25):
    """Убрать шумные кадры: собранные из < frac от обычного (медианного по детектору) числа снимков."""
    if not info or len(info) != len(frames):
        return frames
    keep = []
    for det in sorted({f[1] for f in frames}):
        idx = [i for i, f in enumerate(frames) if f[1] == det]
        n = np.array([info[i]["n_exposures"] for i in idx])
        ok = [i for i, k in zip(idx, n >= frac * np.median(n)) if k]
        keep.extend(ok if len(ok) >= 2 else idx)
    return [frames[i] for i in sorted(keep)]
