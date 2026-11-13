"""Координаты, движение, переменность и карточка «Кто это?» (Питонист 3).

    now_coord(name, when)                         -> SkyCoord на дату
    measure_motion(frames, coord, target_wcs)     -> скорость ″/год, направление, ошибка, положения
    measure_variability(frames, coord, target_wcs)-> размах яркости, метка
    who_is_it(coord, motion, scores, variability) -> карточка для интерфейса

frames — формат epochs.build_frames: [(дата, детектор, 2D)], target_wcs — epochs.make_target_wcs.
"""
from __future__ import annotations

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

from targets import TARGETS, coord_at, simbad, simbad_coord

PIX_ARCSEC = 6.15
SIMBAD_MIRROR = "https://simbad.harvard.edu/simbad/sim-coo"

# Имена тестовых объектов в любом написании -> ключ targets.TARGETS
ALIASES = {
    "WISE 0855": "wise0855", "WISE0855": "wise0855", "WISE 0855-0714": "wise0855",
    "WISE J085510.83-071442.5": "wise0855",
    "BARNARD'S STAR": "barnard", "BARNARD": "barnard", "BARNARDS STAR": "barnard",
    "EPS IND B": "eps_ind_b", "EPSILON INDI B": "eps_ind_b", "ε IND B": "eps_ind_b",
    "RW AUR": "rw_aur", "RW AURIGAE": "rw_aur",
    "R DRA": "r_dra", "R DRACONIS": "r_dra",
}
# Для совместимости со старым кодом: имена тестовых объектов
FALLBACK_OBJECTS = {k: TARGETS[v] for k, v in ALIASES.items()}


def _target_key(name: str) -> str | None:
    n = name.strip().upper()
    if n.lower() in TARGETS:
        return n.lower()
    if n in ALIASES:
        return ALIASES[n]
    if "0855" in n:
        return "wise0855"
    return None


def now_coord(name: str, when: str = "2026-01-01") -> SkyCoord:
    """Координаты объекта на дату when с учётом собственного движения.

    Тестовые объекты — из targets.py (Gaia DR3 / литература, работает без сети);
    остальные — из SIMBAD (зеркало Гарварда); в крайнем случае — Sesame без движения.
    """
    key = _target_key(name)
    if key is not None:
        return coord_at(key, when)
    try:
        return simbad_coord(name, when)
    except Exception as e:
        try:
            return SkyCoord.from_name(name)  # без собственного движения
        except Exception:
            raise RuntimeError(f"Не удалось получить координаты для '{name}': {e}")


def _build_default_wcs(coord: SkyCoord, shape: tuple, pix_arcsec: float = PIX_ARCSEC) -> WCS:
    """Та же сетка, что epochs.make_target_wcs (на случай, если WCS не передали)."""
    ny, nx = shape
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [coord.ra.deg, coord.dec.deg]
    w.wcs.crpix = [(nx + 1) / 2.0, (ny + 1) / 2.0]
    w.wcs.cdelt = [-pix_arcsec / 3600.0, pix_arcsec / 3600.0]
    return w


def _group(frames, target_wcs):
    by_det = {}
    for item in frames:
        date, det, img = item[0], item[1], item[2]
        w = item[3] if len(item) > 3 else target_wcs
        by_det.setdefault(det, []).append((date, img, w))
    return by_det


def _fill_nan_cores(cut: np.ndarray) -> np.ndarray:
    """Пересвеченное ядро (NaN внутри звезды) -> максимум окна; прочие NaN -> медиана."""
    out = np.array(cut, dtype=float)
    bad = ~np.isfinite(out)
    if not bad.any():
        return out
    from scipy import ndimage
    lab, n = ndimage.label(bad)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    vmax, vmed = np.nanmax(out), np.nanmedian(out)
    for k in range(1, n + 1):
        out[lab == k] = vmed if k in edge else vmax
    return out


def _saturated_core(img: np.ndarray, x: float, y: float, max_r: int = 10):
    """Если в точке (x, y) пересвеченное ядро (закрытая NaN-область), вернуть (cx, cy, площадь)
    — центр «дыры». Дыра симметрична вокруг звезды, так что это точный центр яркой звезды."""
    from scipy import ndimage
    ix, iy = int(round(x)), int(round(y))
    ny, nx = img.shape
    if not (0 <= ix < nx and 0 <= iy < ny):
        return None
    y0, y1, x0, x1 = max(0, iy - max_r), min(ny, iy + max_r + 1), max(0, ix - max_r), min(nx, ix + max_r + 1)
    sub_ = ~np.isfinite(img[y0:y1, x0:x1])
    lab, n = ndimage.label(sub_)
    if n == 0:
        return None
    best, best_d = None, 2.5
    for k in range(1, n + 1):
        yy, xx = np.nonzero(lab == k)
        d = np.min(np.hypot(xx + x0 - x, yy + y0 - y))
        if d < best_d:
            best, best_d = k, d
    if best is None:
        return None
    yy, xx = np.nonzero(lab == best)
    h, w = sub_.shape
    touches_edge = ((yy.min() == 0 and y0 == 0) or (xx.min() == 0 and x0 == 0)
                    or (yy.max() == h - 1 and y1 == ny) or (xx.max() == w - 1 and x1 == nx))
    if touches_edge or len(yy) > h * w * 0.8:
        return None  # это край кадра / нет данных, а не звезда
    return float(xx.mean() + x0), float(yy.mean() + y0), int(len(yy))


def _noise(img: np.ndarray) -> float:
    v = img[np.isfinite(img)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if v.size else np.nan


def _prominence(img: np.ndarray, w: WCS, coord: SkyCoord, r_px: int = 2) -> float:
    """Насколько объект в точке coord выделяется на кадре: пик / 99.5-й перцентиль кадра.

    Холодный карлик на длинных волнах — один из самых ярких источников кадра (>1),
    а на коротких — теряется среди звёзд. Так выбирается детектор, где виден именно он,
    а не соседняя звезда.
    """
    x, y = (float(v) for v in w.world_to_pixel(coord))
    ix, iy = int(round(x)), int(round(y))
    ny, nx = img.shape
    if not (r_px <= ix < nx - r_px and r_px <= iy < ny - r_px):
        return -np.inf
    if _saturated_core(img, x, y) is not None:
        return 10.0  # пересвеченная звезда — заведомо самый яркий источник
    cut = _fill_nan_cores(img[iy - r_px:iy + r_px + 1, ix - r_px:ix + r_px + 1])
    bg = np.nanmedian(img)
    ref = np.nanpercentile(img, 99.5) - bg
    peak = np.nanmax(cut) - bg
    snr = peak / _noise(img) if _noise(img) > 0 else 0
    if not np.isfinite(peak) or ref <= 0 or snr < 5:
        return -np.inf
    return float(peak / ref)


def _select_detectors(by_det: dict, coord: SkyCoord, top: int = 2) -> list[int]:
    """Детекторы (≥2 эпох), где объект виднее всего, по убыванию заметности."""
    scored = []
    for det, items in by_det.items():
        if len(items) < 2:
            continue
        p = np.nanmedian([_prominence(img, w if w is not None else _build_default_wcs(coord, img.shape), coord)
                          for _, img, w in items])
        if np.isfinite(p):
            scored.append((p, det))
    scored.sort(reverse=True)
    if not scored:
        return []
    best = scored[0][0]
    return [d for p, d in scored[:top] if p >= 0.5 * best]


def _centroid(img: np.ndarray, x0: float, y0: float, search: int = 2, r: int = 2):
    """Центр объекта около (x0, y0): ищем самый яркий пиксель в радиусе search,
    затем центр масс в окне ±r вокруг него (пересвеченное ядро залито)."""
    sat = _saturated_core(img, x0, y0)
    if sat is not None:
        return sat[0], sat[1]
    ny, nx = img.shape
    ix, iy = int(round(x0)), int(round(y0))
    b = search + r + 1
    if ix - b < 0 or iy - b < 0 or ix + b >= nx or iy + b >= ny:
        return None
    big = _fill_nan_cores(img[iy - b:iy + b + 1, ix - b:ix + b + 1])
    big = big - np.nanmedian(img)
    inner = big[b - search:b + search + 1, b - search:b + search + 1]
    py, px = np.unravel_index(np.nanargmax(inner), inner.shape)
    py, px = py + b - search, px + b - search
    win = big[py - r:py + r + 1, px - r:px + r + 1].copy()
    win[win < 0] = 0
    if win.sum() <= 0:
        return None
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    cx = px + (win * xx).sum() / win.sum()
    cy = py + (win * yy).sum() / win.sum()
    return ix - b + cx, iy - b + cy


def _date_to_year(d) -> float:
    if isinstance(d, (int, float, np.floating)):
        return float(Time(d, format="mjd").jyear)
    return float(Time(str(d).strip()).jyear)


def _fit_motion(times, ras, decs):
    t = np.asarray(times) - np.mean(times)
    cosd = np.cos(np.deg2rad(np.mean(decs)))
    x = (np.asarray(ras) - np.mean(ras)) * cosd * 3600
    y = (np.asarray(decs) - np.mean(decs)) * 3600
    if len(t) >= 3:
        (px, _), cx = np.polyfit(t, x, 1, cov=True)
        (py, _), cy = np.polyfit(t, y, 1, cov=True)
        ex, ey = np.sqrt(cx[0, 0]), np.sqrt(cy[0, 0])
    else:
        px, py = np.polyfit(t, x, 1)[0], np.polyfit(t, y, 1)[0]
        ex = ey = np.nan
    return float(px), float(py), float(ex), float(ey)


def measure_motion(frames: list, coord: SkyCoord, target_wcs: WCS = None, box: int = 2,
                   detectors: list[int] | None = None) -> dict:
    """Собственное движение по кадрам эпох.

    В каждом из 1–2 детекторов, где объект виден лучше всего, находим центр объекта
    на каждой эпохе и проводим прямую по времени; детекторы усредняются.
    Возвращает speed_arcsec_yr, direction_deg (от севера через восток), pm_ra/pm_dec,
    speed_err_arcsec_yr (ошибка по разбросу точек; NaN при 2 эпохах), detectors, positions.
    """
    empty = {"speed_arcsec_yr": 0.0, "direction_deg": 0.0, "pm_ra_arcsec_yr": 0.0, "pm_dec_arcsec_yr": 0.0,
             "speed_err_arcsec_yr": float("nan"), "n_epochs": 0, "detectors": [], "positions": []}
    if not frames or len(frames) < 2:
        return empty
    by_det = _group(frames, target_wcs)
    dets = detectors or _select_detectors(by_det, coord)
    if not dets:
        return empty

    fits, positions, n_ep = [], [], 0
    for det in dets:
        times, ras, decs = [], [], []
        for date, img, w in by_det[det]:
            w = w if w is not None else _build_default_wcs(coord, img.shape)
            x0, y0 = (float(v) for v in w.world_to_pixel(coord))
            c = _centroid(img, x0, y0, search=box)
            if c is None:
                continue
            pos = w.pixel_to_world(*c)
            try:
                times.append(_date_to_year(date))
            except Exception:
                continue
            ras.append(float(pos.ra.deg)); decs.append(float(pos.dec.deg))
            positions.append({"epoch": str(date), "detector": int(det),
                              "ra": round(float(pos.ra.deg), 6), "dec": round(float(pos.dec.deg), 6)})
        if len(times) >= 2 and max(times) - min(times) >= 0.25:
            fits.append(_fit_motion(times, ras, decs))
            n_ep = max(n_ep, len(times))
    if not fits:
        return dict(empty, positions=positions)

    f = np.array(fits)
    if len(f) > 1:
        # детекторы противоречат друг другу (например, на коротких волнах меряется соседняя звезда)
        # -> верим тому, где объект заметнее (он первый в списке)
        e = np.nanmax(np.hypot(f[:, 2], f[:, 3])) if np.isfinite(f[:, 2:]).any() else 1.0
        if np.hypot(*(f[0, :2] - f[1, :2])) > max(3.0, 3 * e):
            f, dets = f[:1], dets[:1]
    pmra, pmdec = float(np.mean(f[:, 0])), float(np.mean(f[:, 1]))
    # ошибка: по разбросу точек внутри детектора и между детекторами
    errs = np.nanmean(np.hypot(f[:, 2], f[:, 3])) if np.isfinite(f[:, 2:]).any() else np.nan
    if len(f) > 1:
        between = float(np.hypot(*np.std(f[:, :2], axis=0)))
        errs = np.nanmax([errs, between]) if np.isfinite(errs) else between
    speed = float(np.hypot(pmra, pmdec))
    return {
        "speed_arcsec_yr": round(speed, 2),
        "direction_deg": round(float(np.degrees(np.arctan2(pmra, pmdec)) % 360), 1),
        "pm_ra_arcsec_yr": round(pmra, 2),
        "pm_dec_arcsec_yr": round(pmdec, 2),
        "speed_err_arcsec_yr": round(float(errs), 2) if np.isfinite(errs) else float("nan"),
        "n_epochs": n_ep,
        "detectors": [int(d) for d in dets],
        "positions": positions,
    }


def _ring_mean(img: np.ndarray, x: float, y: float, r_in: float, r_out: float) -> float:
    """Средняя яркость в кольце по хорошим пикселям (не зависит от размера пересвеченной дыры)."""
    v = CircularAnnulus((x, y), r_in=r_in, r_out=r_out).to_mask("center").get_values(img)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


def _flux(img: np.ndarray, x: float, y: float, r: float, r_in: float, r_out: float) -> float:
    """Апертурный поток с фоном из кольца; NaN-пиксели в апертуре игнорируются."""
    mask = ~np.isfinite(img)
    ap = CircularAperture((x, y), r=r)
    an = CircularAnnulus((x, y), r_in=r_in, r_out=r_out)
    bkg = np.nanmedian(an.to_mask("center").get_values(img))
    t = aperture_photometry(np.nan_to_num(img), ap, mask=mask)["aperture_sum"][0]
    # площадь только по хорошим пикселям
    area = aperture_photometry((~mask).astype(float), ap)["aperture_sum"][0]
    return float(t - bkg * area)


def measure_variability(frames: list, coord: SkyCoord, target_wcs: WCS = None) -> dict:
    """Переменность: размах потока объекта между эпохами в детекторе, где он виден лучше всего.

    Пересвеченные звёзды (NaN в ядре) меряются по крыльям — кольцо 2–5 пикселей.
    Оговорка: внутри детектора длина волны в разных эпохах чуть разная, у очень красных
    звёзд это добавляет ~10–20% к размаху; порог «переменная» — 30%.
    """
    if not frames or len(frames) < 2:
        return {"is_variable": False, "rel_amp": 0.0, "fluxes": [], "label": "Недостаточно данных"}
    by_det = _group(frames, target_wcs)
    dets = _select_detectors(by_det, coord, top=1)
    if not dets:
        return {"is_variable": False, "rel_amp": 0.0, "fluxes": [], "label": "Объект слишком тусклый"}

    items = by_det[dets[0]]
    wings, r_sat = False, 2.0
    for date, img, w in items:
        w = w if w is not None else _build_default_wcs(coord, img.shape)
        x, y = (float(v) for v in w.world_to_pixel(coord))
        sat = _saturated_core(img, x, y)
        if sat is not None:
            wings = True
            r_sat = max(r_sat, float(np.sqrt(sat[2] / np.pi)))
    fluxes, dates, errs = [], [], []
    for date, img, w in items:
        w = w if w is not None else _build_default_wcs(coord, img.shape)
        x, y = (float(v) for v in w.world_to_pixel(coord))
        c = _centroid(img, x, y)
        if c is not None:
            x, y = c
        ny, nx = img.shape
        if x < 12 or x > nx - 13 or y < 12 or y > ny - 13:
            continue
        noise = _noise(img)
        if wings:
            # яркость крыльев: кольцо сразу за самой большой пересвеченной областью минус фон
            r1 = r_sat + 1.5
            f = _ring_mean(img, x, y, r1, r1 + 2.5) - _ring_mean(img, x, y, r1 + 5, r1 + 7)
            e = noise / np.sqrt(max(1.0, np.pi * ((r1 + 2.5) ** 2 - r1 ** 2)))
        else:
            f = _flux(img, x, y, r=2, r_in=4, r_out=6)
            e = noise * np.sqrt(np.pi * 4)
        if np.isfinite(f) and f > 0:
            fluxes.append(f); dates.append(str(date)); errs.append(e)

    if len(fluxes) < 2:
        return {"is_variable": False, "rel_amp": 0.0, "fluxes": fluxes, "label": "Недостаточно данных"}
    med = float(np.median(fluxes))
    rel = float((max(fluxes) - min(fluxes)) / med)
    signif = (max(fluxes) - min(fluxes)) / (np.sqrt(2) * np.median(errs)) if np.median(errs) > 0 else np.inf
    is_var = rel > 0.30 and signif > 5
    how = " (по крыльям: ядро пересвечено)" if wings else ""
    if rel > 0.30 and not is_var:
        label = f"Разброс {rel * 100:.0f}% в пределах шума — переменность не доказана"
    elif rel > 0.6:
        label = f"Сильная переменность (размах {rel * 100:.0f}%){how}"
    elif is_var:
        label = f"Умеренная переменность (размах {rel * 100:.0f}%){how}"
    else:
        label = f"Стабильный блеск (разброс {rel * 100:.0f}%){how}"
    return {"is_variable": is_var, "rel_amp": round(rel, 2), "fluxes": [round(f, 2) for f in fluxes],
            "dates": dates, "detector": int(dets[0]), "saturated": wings, "label": label}


def _simbad_link(coord: SkyCoord, radius_arcsec: float) -> str:
    from urllib.parse import quote
    return (f"{SIMBAD_MIRROR}?Coord={quote(f'{coord.ra.deg:.6f} {coord.dec.deg:+.6f}')}"
            f"&Radius={radius_arcsec:g}&Radius.unit=arcsec")


def gaia_cone(coord: SkyCoord, radius_arcsec: float, top: int = 1000, gmax: float = 20.5):
    """Gaia DR3 в круге (через архив IRSA — он у нас работает стабильно; ~20 с на запрос)."""
    from astroquery.ipac.irsa import Irsa
    q = (f"SELECT TOP {top} source_id, ra, dec, phot_g_mean_mag, pmra, pmdec FROM gaia_dr3_source "
         f"WHERE 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{coord.ra.deg},{coord.dec.deg},{radius_arcsec / 3600}))"
         f" AND phot_g_mean_mag < {gmax} ORDER BY phot_g_mean_mag")
    return Irsa.query_tap(q).to_table()


def match_gaia(coord: SkyCoord, table, radius_arcsec: float = 8.0) -> dict | None:
    """Самая яркая звезда Gaia в радиусе из заранее загруженной таблицы поля."""
    if table is None or len(table) == 0:
        return None
    cat = SkyCoord(np.asarray(table["ra"], float), np.asarray(table["dec"], float), unit="deg")
    sep = coord.separation(cat).arcsec
    idx = np.where(sep < radius_arcsec)[0]
    if not len(idx):
        return None
    i = idx[np.argmin(np.asarray(table["phot_g_mean_mag"], float)[idx])]
    r = table[i]
    pmra, pmdec = float(r["pmra"]), float(r["pmdec"])
    pm = float(np.hypot(pmra, pmdec)) / 1000 if np.isfinite(pmra) and np.isfinite(pmdec) else None
    sid = int(r["source_id"])
    return {"found": True, "name": f"Gaia DR3 {sid}", "catalog_id": f"Gaia DR3 {sid}",
            "otype": f"звезда, G={float(r['phot_g_mean_mag']):.1f}", "source": "Gaia DR3",
            "pm_arcsec_yr": round(pm, 3) if pm else None, "simbad_url": _simbad_link(coord, 15.0)}


def _check_catalogs(coord: SkyCoord, radius_arcsec: float = 15.0, online: bool = True) -> dict:
    """Известен ли объект: тестовые объекты -> SIMBAD (зеркало) -> Gaia DR3."""
    info = {"found": False, "name": "Неизвестный объект", "otype": "", "source": None,
            "simbad_url": _simbad_link(coord, radius_arcsec), "pm_arcsec_yr": None, "catalog_id": None}

    for key, t in TARGETS.items():
        if coord.separation(coord_at(key, "2026-01-01")).arcsec < 30.0:
            info.update(found=True, name=t["title"], otype=t["kind"], source="targets (Gaia DR3/SIMBAD)",
                        catalog_id=t["simbad"], pm_arcsec_yr=round(float(np.hypot(t["pmra"], t["pmdec"])) / 1000, 2))
            return info
    if not online:
        return info

    try:
        s = simbad()
        s.add_votable_fields("otype")
        res = s.query_region(coord, radius=radius_arcsec * u.arcsec)
        if res is not None and len(res) > 0:
            cols = {c.lower(): c for c in res.colnames}
            r0 = res[0]
            name = str(r0[cols.get("main_id", res.colnames[0])]).strip()
            pm = None
            try:
                pm = round(float(np.hypot(float(r0[cols["pmra"]]), float(r0[cols["pmdec"]]))) / 1000, 2)
                pm = pm if np.isfinite(pm) else None
            except Exception:
                pass
            info.update(found=True, name=name, catalog_id=name, pm_arcsec_yr=pm,
                        otype=str(r0[cols["otype"]]).strip() if "otype" in cols else "", source="SIMBAD")
            return info
    except Exception:
        pass

    try:
        t = gaia_cone(coord, radius_arcsec, top=1)
        if t is not None and len(t):
            m = match_gaia(coord, t, radius_arcsec)
            if m:
                info.update(m)
    except Exception:
        pass
    return info


def who_is_it(coord: SkyCoord, motion: dict, scores: dict, variability: dict, online: bool = True,
              cat: dict | None = None) -> dict:
    """Карточка «Кто это?» по простым правилам (см. план). cat — готовый результат сверки с каталогами."""
    if cat is None:
        cat = _check_catalogs(coord, online=online)
    cat = dict({"found": False, "name": "Неизвестный объект", "otype": "", "source": None,
                "simbad_url": _simbad_link(coord, 15.0), "pm_arcsec_yr": None, "catalog_id": None}, **cat)
    motion = motion or {}
    speed = motion.get("speed_arcsec_yr", 0.0) or 0.0
    err = motion.get("speed_err_arcsec_yr", float("nan"))
    n_ep = motion.get("n_epochs", 0)
    variability = variability or {}
    is_var = variability.get("is_variable", False)
    var_label = variability.get("label", "Нет данных")
    scores = scores or {}

    # движение считаем значимым, если оно больше 1″/год и больше 2 ошибок (эпох всего 3–5)
    noise = 2 * err if np.isfinite(err) else 2.0
    moving = speed >= max(1.0, noise)
    has_water = (scores.get("water") or 0) > 0.15
    has_methane = (scores.get("methane") or 0) > 0.15
    has_mol = has_water or has_methane

    if n_ep <= 1 or speed > 60.0:
        verdict = "Возможно, астероид или артефакт (виден на одном кадре или слишком быстрый)"
    elif moving and has_mol:
        verdict = "Похож на коричневый карлик (движется + признаки метана/воды)"
    elif moving:
        verdict = "Близкая звезда с большим собственным движением"
    elif is_var:
        verdict = "Переменная звезда (меняет яркость, но стоит на месте)"
    elif not cat["found"]:
        verdict = "Неподвижный источник, в каталогах не найден"
    else:
        verdict = f"Известный неподвижный источник ({cat['otype'] or 'звезда/галактика'})"
    if moving and not cat["found"] and n_ep > 1 and speed <= 60.0:
        verdict += " — в каталогах нет: КАНДИДАТ, нужна проверка"

    names = {"water": "Вода (H₂O)", "methane": "Метан (CH₄)", "co2": "CO₂", "co": "CO"}
    badges = [f"{lab}: {scores[k]:.2f}" for k, lab in names.items()
              if scores.get(k) is not None and np.isfinite(scores[k]) and scores[k] > 0.15]
    speed_txt = (f"{speed:.1f} ± {err:.1f}″/год" if np.isfinite(err) else f"{speed:.1f}″/год")
    return {
        "name": cat["name"], "catalog_found": cat["found"], "catalog_type": cat["otype"],
        "catalog_source": cat["source"], "catalog_id": cat["catalog_id"],
        "catalog_pm_arcsec_yr": cat["pm_arcsec_yr"],
        "speed_arcsec_yr": speed, "speed_err_arcsec_yr": err, "speed_text": speed_txt,
        "direction_deg": motion.get("direction_deg", 0.0), "moving": bool(moving),
        "variability": var_label,
        "molecule_badges": badges or ["Явных молекулярных полос не выявлено"],
        "scores": scores, "verdict": verdict, "simbad_url": cat["simbad_url"],
    }
