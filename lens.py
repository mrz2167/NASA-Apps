# AI disclosure: parts of this file were written with AI assistance (Claude, Anthropic), then reviewed and tested by the LUMEN team. See README, "AI use".
"""Карты молекул для линз сайта: где на картинке видно поглощение воды, метана, CO₂, CO.

    maps = molecule_maps(sky)      # {"water": 2D, "methane": 2D, "co2": 2D, "co": 2D}

Идея: каждый снимок SPHEREx видит каждую точку неба на своей длине волны (фильтр переменный,
длина волны меняется поперёк кадра). Если собрать все снимки места (все детекторы, все проходы),
у каждого пикселя общей сетки получается свой грубый спектр — сотни точек (длина волны, поток).
Глубина полосы в пикселе = 1 − поток в полосе / континуум. Окна те же, что у индексов в карточке
объекта (spectrum.BANDS), но континуум — прямая между двумя окнами, а не общая медиана, и из карты
вычитается общий сдвиг поля (медиана по ярким пикселям). Поэтому карта показывает, ГДЕ поглощение
сильнее, чем у соседних звёзд; числа в карточке считаются по-своему и могут отличаться.

Значение есть только там, где пиксель заметно ярче фона (иначе отношение — шум), в полосе и
континууме хватает снимков и рядом нет пересвеченной звезды; в остальных пикселях NaN. Это признак,
а не доказательство: на краях ярких звёзд и у быстро движущихся объектов (снимки разных лет видят
их в разных местах) бывают ложные пятна; карта сглаживается окном 3×3 пикселя (≈ размер звезды).
Очень тусклые объекты (WISE 0855: континуум ~5σ) по пикселям не измерить; их пятно закрашивает
paint_object по индексам спектра всего объекта (те же числа, что в карточке).
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.ndimage import binary_dilation, map_coordinates, uniform_filter

from epochs import clean_image
from spectrum import BANDS

MIN_SAMPLES = 2      # снимков в полосе и в континууме на пиксель
SNR_MIN = 8.0        # континуум ярче фона во столько σ, иначе пиксель пропускаем
MIN_FIELD_PX = 15    # столько ярких пикселей нужно, чтобы оценить общий сдвиг поля
SAT_HALO_PX = 5      # вокруг пересвеченных ядер карту не строим: края дают ложное кольцо


def _samples(cutouts: list[dict], wcs, shape) -> tuple[np.ndarray, np.ndarray]:
    """(поток, длина волны) каждого снимка в каждом пикселе общей сетки: два массива (n, ny, nx)."""
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    world = wcs.pixel_to_world(xx.ravel(), yy.ravel())
    F, W = [], []
    for c in cutouts:
        try:
            cx, cy = c["wcs"].world_to_pixel(world)
            img = clean_image(c)
            h, w = img.shape
            inside = (cx >= 0) & (cx <= w - 1) & (cy >= 0) & (cy <= h - 1)
            if not inside.any():
                continue
            # билинейно; NaN-пиксели (плохие) не размазываем: маска отдельно
            bad = ~np.isfinite(img)
            f = map_coordinates(np.nan_to_num(img), [cy, cx], order=1, mode="nearest")
            b = map_coordinates(bad.astype(np.float32), [cy, cx], order=1, mode="nearest") > 0.01
            wl, _ = c["spectral_wcs"].pixel_to_world_values(cx, cy)
            wl = np.asarray(wl, float) * 1e6 if np.nanmedian(wl) < 1e-3 else np.asarray(wl, float)
            f = np.where(inside & ~b & np.isfinite(wl), f, np.nan)
            F.append(f.reshape(ny, nx).astype(np.float32))
            W.append(wl.reshape(ny, nx).astype(np.float32))
        except Exception:
            continue
    if not F:
        return np.empty((0, ny, nx), np.float32), np.empty((0, ny, nx), np.float32)
    return np.stack(F), np.stack(W)


def _smooth(stack: np.ndarray) -> np.ndarray:
    """Сумма по окну 3×3 в каждом снимке (как апертура), NaN не размазываются."""
    ok = np.isfinite(stack)
    num = uniform_filter(np.where(ok, stack, 0.0), size=(1, 3, 3), mode="nearest")
    den = uniform_filter(ok.astype(np.float32), size=(1, 3, 3), mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0.5, num / den, np.nan).astype(np.float32)


def _median_in(F: np.ndarray, W: np.ndarray, windows) -> tuple[np.ndarray, np.ndarray]:
    """Медиана потока по снимкам, чья длина волны в пикселе попала в любое из окон; и число таких снимков."""
    m = np.zeros(F.shape, bool)
    for lo, hi in windows:
        m |= (W > lo) & (W < hi)
    m &= np.isfinite(F)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.where(m, F, np.nan), axis=0), m.sum(axis=0)


def molecule_maps(sky: dict) -> dict[str, np.ndarray]:
    """{молекула: карта глубины полосы 0..1 на сетке sky (NaN — нет данных или слишком тускло)}."""
    shape = tuple(sky["shape"])
    F, W = _samples(sky["cutouts"], sky["wcs"], shape)
    out = {}
    if len(F) == 0:
        return {k: np.full(shape, np.nan, np.float32) for k in BANDS}
    # пересвеченные звёзды: ядро помечено плохим в большинстве снимков -> вместе с окрестностью не трогаем
    with np.errstate(invalid="ignore"):
        sat = np.isfinite(F).mean(axis=0) < 0.5
    inside = np.isfinite(F).any(axis=0)
    sat = binary_dilation(sat & inside, iterations=SAT_HALO_PX)
    F = _smooth(F)
    for name, b in BANDS.items():
        fb, nb = _median_in(F, W, [b["band"]])
        # континуум: медиана в каждом окне отдельно и прямая между ними к центру полосы. Общая
        # медиана двух окон смещается к окну, где больше снимков, и на обычных звёздах с наклонным
        # спектром даёт ложную «полосу» (проверено: «вода» на всех ярких звёздах поля).
        (l1, h1), (l2, h2) = b["cont"]
        f1, n1 = _median_in(F, W, [(l1, h1)])
        f2, n2 = _median_in(F, W, [(l2, h2)])
        x1, x2, xb = (l1 + h1) / 2, (l2 + h2) / 2, sum(b["band"]) / 2
        if b.get("powerlaw"):   # далёкие окна (метан): степенной закон, как в карточке (spectrum.BANDS)
            with np.errstate(invalid="ignore", divide="ignore"):
                t = np.log(xb / x1) / np.log(x2 / x1)
                fc = np.where((f1 > 0) & (f2 > 0), np.exp((1 - t) * np.log(f1) + t * np.log(f2)), np.nan)
        else:
            fc = f1 + (f2 - f1) * (xb - x1) / (x2 - x1)
        nc = np.minimum(n1, n2)
        v = fc[np.isfinite(fc)]
        if v.size < 20:
            out[name] = np.full(shape, np.nan, np.float32)
            continue
        med = np.median(v)
        sig = 1.4826 * np.median(np.abs(v - med)) or 1e-9
        bright = (fc - med) > SNR_MIN * sig
        with np.errstate(invalid="ignore", divide="ignore"):
            depth = 1.0 - fb / fc
        ok = bright & ~sat & (nb >= MIN_SAMPLES) & (nc >= MIN_SAMPLES) & np.isfinite(depth)
        # Поправка на поле: размер звезды растёт с длиной волны, калибровка детекторов чуть разная —
        # у всех звёзд поля появляется общий сдвиг глубины. Большинство звёзд без полосы, поэтому
        # вычитаем медиану по ярким пикселям: светится то, что поглощает сильнее соседей.
        if ok.sum() >= MIN_FIELD_PX:
            depth = depth - np.median(depth[ok])
        out[name] = np.where(ok, np.clip(depth, 0.0, 1.0), np.nan).astype(np.float32)
    return out


def paint_object(maps: dict[str, np.ndarray], x: float, y: float, scores: dict[str, float],
                 radius_px: float = 2.2) -> dict[str, np.ndarray]:
    """Закрасить пятно объекта (кружок radius_px вокруг пикселя x, y сетки) по индексам его спектра.

    Для тусклых объектов (WISE 0855: континуум ~5σ) глубина по отдельному пикселю — шум, а спектр
    по апертуре вокруг всего объекта (spectrum.get_spectrum -> карточка) надёжнее. Закрашиваются
    только пиксели, где своей карты нет (NaN): измеренное не перекрываем.
    """
    shape = next(iter(maps.values())).shape
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    disc = (xx - x) ** 2 + (yy - y) ** 2 <= radius_px ** 2
    out = {}
    for k, m in maps.items():
        v = float(scores.get(k) or 0.0)
        out[k] = np.where(disc & ~np.isfinite(m), np.float32(np.clip(v, 0, 1)), m) if v > 0 else m
    return out

