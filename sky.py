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
            detectors=None, offline: bool = False, on_progress=None, verbose: bool = False,
            cadence: str = "pass") -> dict:
    """Координаты (градусы) -> кадры, подписи, разности.

    quick=True    — не больше per_group снимков на (эпоху, детектор): ~50 вырезок, 1–2 минуты.
    quick=False   — все снимки (для спектра; первый раз 10+ минут).
    detectors     — например [6] или [5, 6], чтобы качать только нужные детекторы.
    offline=True  — не ходить в сеть, только кэш.
    on_progress(i, n) — колбэк прогресса загрузки вырезок.
    cadence       — "pass": эпоха = полугодовой проход; "visit": проход режется на сеансы съёмки
                    (epochs.split_cadence; передавайте один детектор), быстрый режим тогда берёт
                    per_group снимков на сеанс.
    """
    coord = SkyCoord(ra, dec, unit="deg")
    warnings: list[str] = []
    source = "archive"

    try:
        table = find_images(coord.ra.deg, coord.dec.deg, size_deg, offline=offline)  # кэш, потом архив
    except FileNotFoundError as e:
        raise SkyUnavailable("No saved data for this sky position (offline mode).") from e
    except Exception as e:
        raise SkyUnavailable(f"The IRSA archive is not responding and there is no cached data for this position: {e}") from e

    n_found = len(table)
    if n_found == 0:
        raise SkyUnavailable("SPHEREx has not observed this position yet (or its data are not in the archive yet).")
    if detectors is not None:
        table = table[[d in detectors for d in table["detector"]]]
    if quick:
        table = select_exposures(table, per_group, cadence)
    n_wanted = len(table)

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
            warnings.append(f"Loaded {len(cutouts)} of {len(table)} images "
                            f"(the archive did not return some of the data).")
    if not cutouts:
        raise SkyUnavailable("Could not load any images: the archive is unavailable and there is no cache.")

    info: list[dict] = []
    frames = build_frames(cutouts, coord, size_deg, info=info, cadence=cadence)
    if not frames:
        raise SkyUnavailable("Too few images to build frames.")
    by_det = frames_by_detector(frames)
    if all(len(v) < 2 for v in by_det.values()):
        warnings.append("Only one epoch is available, so there is nothing to blink yet.")

    wcs, shape = make_target_wcs(coord, size_deg)
    return dict(
        coord=coord, size_deg=size_deg, wcs=wcs, shape=shape,
        frames=frames, info=info, diffs=diff_frames(frames), by_det=by_det,
        cutouts=cutouts, n_found=n_found, n_wanted=n_wanted, n_loaded=len(cutouts),
        source=source, quick=quick, warnings=warnings, cadence=cadence, detectors=detectors,
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


def _clip_spectrum(spec: dict, window: int = 15, k: float = 6.0) -> dict:
    """Убрать одиночные выбросы (космические частицы, соседние звёзды): точка дальше k·σ
    от медианы соседей по длине волны."""
    import numpy as np
    wl, fl, er = (np.asarray(spec[c], float) for c in ("wavelength", "flux", "error"))
    if len(wl) < window:
        return spec
    keep = np.ones(len(wl), bool)
    h = window // 2
    for i in range(len(wl)):
        nb = np.r_[fl[max(0, i - h):i], fl[i + 1:i + 1 + h]]
        med = np.median(nb)
        sig = 1.4826 * np.median(np.abs(nb - med)) or np.median(er) or 1e-9
        keep[i] = abs(fl[i] - med) <= k * sig
    return {"wavelength": wl[keep].round(4).tolist(), "flux": fl[keep].round(6).tolist(),
            "error": er[keep].round(6).tolist()}


def analyze(sky: dict, online: bool = False, valid_detectors=None, coord=None) -> dict:
    """Полный разбор объекта в центре: движение, переменность, спектр, молекулы, карточка.

    online=False — сверка с каталогами только по тестовым объектам (быстро, без сети);
    online=True  — ещё SIMBAD (зеркало) и Gaia.
    coord — разобрать не центр, а другую точку кадра (например, куда кликнул пользователь).
    Возвращает {"motion", "variability", "spectrum", "scores", "card"}.
    """
    from identify import measure_motion, measure_variability, who_is_it
    from spectrum import get_spectrum, molecule_scores
    import numpy as np

    from epochs import good_frames
    coord, wcs = (coord if coord is not None else sky["coord"]), sky["wcs"]
    frames = good_frames(sky["frames"], sky.get("info"))   # без шумных кадров из 2–4 снимков
    motion = measure_motion(frames, coord, wcs)
    variability = measure_variability(frames, coord, wcs)
    spec = _clip_spectrum(get_spectrum(sky["cutouts"], coord.ra.deg, coord.dec.deg, valid_detectors=valid_detectors))
    fl, er = np.array(spec["flux"]), np.array(spec["error"])
    snr = float(np.median(fl / np.where(er > 0, er, np.inf))) if len(fl) else 0.0
    spec_ok = len(fl) >= 20 and snr >= 3.0          # на шуме индексы молекул дают мусор
    scores = (molecule_scores(np.array(spec["wavelength"]), fl) if spec_ok
              else {"water": 0.0, "methane": 0.0, "co2": 0.0, "co": 0.0})
    card = who_is_it(coord, motion, scores, variability, online=online)
    if len(spec["wavelength"]) < 20:
        card["molecule_badges"] = (["Star too bright: saturated core, spectrum unavailable"]
                                   if variability.get("saturated") else ["Not enough spectrum points"])
    elif not spec_ok:
        card["molecule_badges"] = [f"Signal too weak for chemistry (S/N≈{snr:.1f})"]
    card["spectrum_snr"] = round(snr, 1)
    return dict(motion=motion, variability=variability, spectrum=spec, scores=scores, card=card)
