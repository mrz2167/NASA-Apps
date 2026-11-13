"""Одна функция для сайта: координаты -> всё, что нужно показать.

    from sky import get_sky, get_target
    sky = get_sky(133.76006, -7.24221)            # быстрый режим, ~1–2 минуты (или мгновенно из кэша)
    sky = get_target("barnard")                   # один из 5 тестовых объектов

    sky["frames"]   [(дата, детектор, 2D)]        кадры для мигания
    sky["info"]     [{date, detector, n_exposures, wavelength_um, ...}]  подписи, в том же порядке
    sky["diffs"]    [(дата, детектор, 2D)]        «что изменилось» (минус первая эпоха)
    sky["by_det"]   {детектор: [(дата, кадр)]}
    sky["wcs"], sky["shape"]                      общая сетка кадров
    sky["coord"]    SkyCoord центра
    sky["cutouts"]  исходные вырезки (для спектра и фотометрии)
    sky["source"]   "archive" | "cache" — откуда данные
    sky["warnings"] список строк для показа пользователю

Если архив IRSA не отвечает, берётся то, что есть в cache/. Если нет и кэша — SkyUnavailable.
"""
from __future__ import annotations

from astropy.coordinates import SkyCoord

from data import find_images, load_cutouts
from epochs import build_frames, diff_frames, frames_by_detector, make_target_wcs, select_exposures

DETECTOR_BANDS_UM = {1: (0.75, 1.09), 2: (1.10, 1.62), 3: (1.63, 2.42),
                     4: (2.43, 3.82), 5: (3.83, 4.41), 6: (4.42, 5.00)}


class SkyUnavailable(RuntimeError):
    """Нет ни ответа архива, ни кэша для этого места."""


def get_sky(ra: float, dec: float, size_deg: float = 0.1, quick: bool = True, per_group: int = 3,
            detectors=None, offline: bool = False, on_progress=None, verbose: bool = False) -> dict:
    """Координаты (градусы) -> кадры, подписи, разности.

    quick=True    — не больше per_group снимков на (эпоху, детектор): ~50 вырезок, 1–2 минуты.
    quick=False   — все снимки (для спектра; первый раз 10+ минут).
    detectors     — например [6] или [5, 6], чтобы качать только нужные детекторы.
    offline=True  — не ходить в сеть, только кэш.
    on_progress(i, n) — колбэк прогресса загрузки вырезок.
    """
    coord = SkyCoord(ra, dec, unit="deg")
    warnings: list[str] = []
    source = "archive"

    try:
        table = find_images(coord.ra.deg, coord.dec.deg, size_deg, offline=offline)  # кэш, потом архив
    except FileNotFoundError as e:
        raise SkyUnavailable("Для этого места нет сохранённых данных (офлайн-режим).") from e
    except Exception as e:
        raise SkyUnavailable(f"Архив IRSA не отвечает, а кэша для этого места нет: {e}") from e

    n_found = len(table)
    if n_found == 0:
        raise SkyUnavailable("SPHEREx ещё не снимал это место (или его данных пока нет в архиве).")
    if detectors is not None:
        table = table[[d in detectors for d in table["detector"]]]
    if quick:
        table = select_exposures(table, per_group)

    cutouts = load_cutouts(table, progress=verbose, on_progress=on_progress, offline=offline)
    if offline:
        source = "cache"
    if len(cutouts) < len(table):
        missing = len(table) - len(cutouts)
        if not offline and missing == len(table):
            # архив лёг посреди загрузки: пробуем кэш
            cutouts = load_cutouts(table, progress=verbose, offline=True)
            source = "cache"
        if len(cutouts) < len(table):
            warnings.append(f"Загружено {len(cutouts)} из {len(table)} снимков "
                            f"(архив не отдал часть данных).")
    if not cutouts:
        raise SkyUnavailable("Не удалось загрузить ни одного снимка: архив недоступен, кэша нет.")

    info: list[dict] = []
    frames = build_frames(cutouts, coord, size_deg, info=info)
    if not frames:
        raise SkyUnavailable("Снимков слишком мало, чтобы собрать кадры.")
    by_det = frames_by_detector(frames)
    if all(len(v) < 2 for v in by_det.values()):
        warnings.append("Есть только одна эпоха — мигать пока нечем.")

    wcs, shape = make_target_wcs(coord, size_deg)
    return dict(
        coord=coord, size_deg=size_deg, wcs=wcs, shape=shape,
        frames=frames, info=info, diffs=diff_frames(frames), by_det=by_det,
        cutouts=cutouts, n_found=n_found, n_loaded=len(cutouts),
        source=source, quick=quick, warnings=warnings,
    )


def get_target(name: str, **kw) -> dict:
    """Один из тестовых объектов (barnard, eps_ind_b, wise0855, rw_aur, r_dra)."""
    from targets import TARGETS, coord_at
    c = coord_at(name)
    kw.setdefault("quick", False)  # для тестовых объектов в кэше лежат все снимки
    if name == "r_dra":
        kw.setdefault("per_group", 3)
        kw["quick"] = True        # глубокое поле: 2493 снимка, в кэше быстрый режим
    sky = get_sky(c.ra.deg, c.dec.deg, **kw)
    sky["target"] = dict(TARGETS[name], key=name)
    return sky
