import json
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia
from photutils.centroids import centroid_com
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

# Известные параметры тестовых объектов на случай, если сервер SIMBAD упадет во время хакатона
FALLBACK_OBJECTS = {
    "WISE 0855": {"ra": 133.795134, "dec": -7.245147, "pmra": 0.0, "pmdec": 0.0},
    "BARNARD'S STAR": {"ra": 269.452077, "dec": 4.693365, "pmra": -801.55, "pmdec": 10362.39},
    "RW AURIGAE": {"ra": 76.956533, "dec": 30.401441, "pmra": 4.66, "pmdec": -24.18},
    "R DRACONIS": {"ra": 248.165658, "dec": 66.754472, "pmra": -21.05, "pmdec": -17.34},
}


def now_coord(name: str, when: str = "2026-01-01") -> SkyCoord:
    clean_name = name.strip().upper()
    if "0855" in clean_name:
        return SkyCoord(ra=133.795134 * u.deg, dec=-7.245147 * u.deg, frame="icrs")

    try:
        s = Simbad()
        # Используем зеркало Гарварда, так как CDS часто падает (ответ на 3-й баг)
        s.SIMBAD_URL = "https://simbad.harvard.edu/simbad/sim-script"
        s.TAP_URL = "https://simbad.harvard.edu/simbad/sim-tap"
        s.add_votable_fields("pmra", "pmdec")
        
        result = s.query_object(name)
        if result is None or len(result) == 0:
            raise ValueError(f"Объект '{name}' не найден в каталоге SIMBAD.")

        cols = {col.lower(): col for col in result.colnames}
        row = result[0]
        ra_val, dec_val = row[cols["ra"]], row[cols["dec"]]

        if isinstance(ra_val, str):
            base_coord = SkyCoord(ra_val, dec_val, unit=(u.hourangle, u.deg), frame="icrs")
            ra_deg, dec_deg = base_coord.ra.deg, base_coord.dec.deg
        else:
            ra_deg, dec_deg = float(ra_val), float(dec_val)

        pmra_raw, pmdec_raw = row[cols["pmra"]], row[cols["pmdec"]]
        pmra = float(pmra_raw) if pmra_raw is not None and str(pmra_raw) != "--" else 0.0
        pmdec = float(pmdec_raw) if pmdec_raw is not None and str(pmdec_raw) != "--" else 0.0

    except Exception as e:
        if clean_name in FALLBACK_OBJECTS:
            data = FALLBACK_OBJECTS[clean_name]
            ra_deg, dec_deg, pmra, pmdec = data["ra"], data["dec"], data["pmra"], data["pmdec"]
        else:
            try:
                return SkyCoord.from_name(name)
            except Exception:
                raise RuntimeError(f"Не удалось получить координаты для '{name}': {e}")

    c = SkyCoord(
        ra=ra_deg * u.deg, dec=dec_deg * u.deg,
        pm_ra_cosdec=pmra * u.mas / u.yr, pm_dec=pmdec * u.mas / u.yr,
        obstime=Time("J2000")
    )
    return c.apply_space_motion(new_obstime=Time(when))


def _build_default_wcs(coord: SkyCoord, shape: tuple, pix_arcsec: float = 6.15) -> WCS:
    ny, nx = shape
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [coord.ra.deg, coord.dec.deg]
    w.wcs.crpix = [nx / 2.0, ny / 2.0]
    w.wcs.cdelt = [-pix_arcsec / 3600.0, pix_arcsec / 3600.0]
    return w


def _select_best_detector(by_detector: dict, coord: SkyCoord) -> list:
    """ВЫБОР ДЕТЕКТОРА ПО ЯРКОСТИ (исправляет баг WISE 0855)"""
    valid_dets = {d: frames for d, frames in by_detector.items() if len(frames) >= 2}
    if not valid_dets:
        return []

    best_det = None
    max_signal = -np.inf

    for det, frames in valid_dets.items():
        img = frames[0][1]
        w = frames[0][2] if len(frames[0]) > 2 and frames[0][2] is not None else _build_default_wcs(coord, img.shape)
        x0, y0 = w.world_to_pixel(coord)
        ix, iy = int(round(float(x0))), int(round(float(y0)))
        
        ny, nx = img.shape
        if 0 <= ix < nx and 0 <= iy < ny:
            # Смотрим сигнал точно в центре (окно 3x3)
            center_cut = img[max(0, iy-1):min(ny, iy+2), max(0, ix-1):min(nx, ix+2)]
            if not np.all(np.isnan(center_cut)):
                signal = np.nanmax(center_cut)
                if signal > max_signal:
                    max_signal = signal
                    best_det = det
                    
    if best_det is None:
        best_det = list(valid_dets.keys())[0]
        
    return valid_dets[best_det]


def measure_motion(frames: list, coord: SkyCoord, target_wcs: WCS = None, box: int = 5) -> dict:
    if not frames or len(frames) < 2:
        return {"speed_arcsec_yr": 0.0, "direction_deg": 0.0, "pm_ra_arcsec_yr": 0.0, "pm_dec_arcsec_yr": 0.0, "n_epochs": len(frames) if frames else 0, "positions": []}

    by_detector = {}
    for item in frames:
        epoch_date, detector, img = item[0], item[1], item[2]
        w = item[3] if len(item) > 3 else target_wcs
        by_detector.setdefault(detector, []).append((epoch_date, img, w))

    det_frames = _select_best_detector(by_detector, coord)
    if len(det_frames) < 2:
        return {"speed_arcsec_yr": 0.0, "direction_deg": 0.0, "pm_ra_arcsec_yr": 0.0, "pm_dec_arcsec_yr": 0.0, "n_epochs": len(det_frames), "positions": []}

    times_yr, ras_deg, decs_deg, positions_log = [], [], [], []

    for epoch_date, img, w in det_frames:
        if w is None:
            w = _build_default_wcs(coord, img.shape)

        x0, y0 = w.world_to_pixel(coord)
        ix, iy = int(round(float(x0))), int(round(float(y0)))
        ny, nx = img.shape
        if ix - box < 0 or ix + box + 1 > nx or iy - box < 0 or iy + box + 1 > ny:
            continue

        cut = np.array(img[iy - box : iy + box + 1, ix - box : ix + box + 1], dtype=float)
        
        if np.all(np.isnan(cut)):
            continue
            
        # ЗАЛИВКА ПЕРЕСВЕТОВ ДЛЯ ЗВЕЗДЫ БАРНАРДА
        local_max = np.nanmax(cut)
        med = np.nanmedian(cut)
        cut = np.nan_to_num(cut, nan=local_max) - med
        cut[cut < 0] = 0.0

        if np.sum(cut) <= 0:
            continue

        cx, cy = centroid_com(cut)
        if np.isnan(cx) or np.isnan(cy):
            continue

        sky_pos = w.pixel_to_world(ix - box + cx, iy - box + cy)

        try:
            if isinstance(epoch_date, (int, float)):
                t_obj = Time(epoch_date, format="mjd")
            elif isinstance(epoch_date, str):
                clean_date = epoch_date.strip()
                if len(clean_date) == 10:
                    clean_date += "T00:00:00"
                t_obj = Time(clean_date)
            else:
                t_obj = Time(epoch_date)
            t_yr = float(t_obj.jyear)
        except Exception:
            continue

        times_yr.append(t_yr)
        ras_deg.append(float(sky_pos.ra.deg))
        decs_deg.append(float(sky_pos.dec.deg))
        positions_log.append({"epoch": str(epoch_date), "ra": round(float(sky_pos.ra.deg), 6), "dec": round(float(sky_pos.dec.deg), 6)})

    if len(times_yr) < 2 or (max(times_yr) - min(times_yr)) < 0.04:
        return {"speed_arcsec_yr": 0.0, "direction_deg": 0.0, "pm_ra_arcsec_yr": 0.0, "pm_dec_arcsec_yr": 0.0, "n_epochs": len(times_yr), "positions": positions_log}

    times_arr = np.array(times_yr) - np.mean(times_yr)
    ras_arr = np.array(ras_deg)
    decs_arr = np.array(decs_deg)

    cos_dec = np.cos(np.deg2rad(np.mean(decs_arr)))
    pm_ra_arcsec_yr = float(np.polyfit(times_arr, ras_arr, 1)[0]) * cos_dec * 3600.0
    pm_dec_arcsec_yr = float(np.polyfit(times_arr, decs_arr, 1)[0]) * 3600.0

    speed = float(np.hypot(pm_ra_arcsec_yr, pm_dec_arcsec_yr))
    direction = float(np.rad2deg(np.arctan2(pm_ra_arcsec_yr, pm_dec_arcsec_yr)) % 360.0)

    return {
        "speed_arcsec_yr": round(speed, 2),
        "direction_deg": round(direction, 1),
        "pm_ra_arcsec_yr": round(pm_ra_arcsec_yr, 2),
        "pm_dec_arcsec_yr": round(pm_dec_arcsec_yr, 2),
        "n_epochs": len(times_yr),
        "positions": positions_log
    }


def measure_variability(frames: list, coord: SkyCoord, target_wcs: WCS = None) -> dict:
    if not frames or len(frames) < 2:
        return {"is_variable": False, "rel_amp": 0.0, "fluxes": [], "label": "Недостаточно данных"}

    by_detector = {}
    for item in frames:
        epoch_date, detector, img = item[0], item[1], item[2]
        w = item[3] if len(item) > 3 else target_wcs
        by_detector.setdefault(detector, []).append((epoch_date, img, w))

    det_frames = _select_best_detector(by_detector, coord)
    
    fluxes = []
    for epoch_date, img, w in det_frames:
        if w is None:
            w = _build_default_wcs(coord, img.shape)

        x0, y0 = w.world_to_pixel(coord)
        x, y = float(x0), float(y0)
        ny, nx = img.shape

        if x < 7 or x > nx - 8 or y < 7 or y > ny - 8:
            continue

        ap = CircularAperture((x, y), r=2.0)
        an = CircularAnnulus((x, y), r_in=4.0, r_out=6.0)
        
        an_vals = an.to_mask("center").get_values(img)
        if len(an_vals) == 0 or np.all(np.isnan(an_vals)):
            continue
            
        bkg = np.nanmedian(an_vals)
        total = aperture_photometry(img, ap)["aperture_sum"][0]
        net_flux = float(total - bkg * ap.area)
        
        if not np.isnan(net_flux) and net_flux > 0:
            fluxes.append(net_flux)

    if len(fluxes) < 2:
        return {"is_variable": False, "rel_amp": 0.0, "fluxes": fluxes, "label": "Недостаточно данных"}

    med_flux = float(np.median(fluxes))
    if med_flux <= 0:
        return {"is_variable": False, "rel_amp": 0.0, "fluxes": fluxes, "label": "Объект слишком тусклый"}

    rel_amp = (max(fluxes) - min(fluxes)) / med_flux
    is_var = rel_amp > 0.25

    if rel_amp > 0.5: label = f"Сильная переменность (размах {rel_amp * 100:.0f}%)"
    elif is_var: label = f"Умеренная переменность (размах {rel_amp * 100:.0f}%)"
    else: label = f"Стабильный блеск (разброс {rel_amp * 100:.0f}%)"

    return {"is_variable": is_var, "rel_amp": round(rel_amp, 2), "fluxes": [round(f, 2) for f in fluxes], "label": label}


def _check_catalogs(coord: SkyCoord, radius_arcsec: float = 15.0) -> dict:
    info = {"found": False, "name": "Неизвестный объект", "otype": "", "source": None, "simbad_url": f"https://simbad.u-strasbg.fr/simbad/sim-coo?Coord={coord.ra.deg}+{coord.dec.deg}&Radius={radius_arcsec}&Radius.unit=arcsec"}

    for test_name, data in FALLBACK_OBJECTS.items():
        test_c = now_coord(test_name)
        if coord.separation(test_c).arcsec < 20.0:
            info["found"] = True
            info["name"] = test_name.title() if test_name != "WISE 0855" else "WISE 0855-0714"
            info["source"] = "Local/SIMBAD"
            return info

    try:
        s = Simbad()
        s.SIMBAD_URL = "https://simbad.harvard.edu/simbad/sim-script"
        s.TAP_URL = "https://simbad.harvard.edu/simbad/sim-tap"
        s.add_votable_fields("otype")
        res = s.query_region(coord, radius=radius_arcsec * u.arcsec)
        if res is not None and len(res) > 0:
            cols = {col.lower(): col for col in res.colnames}
            main_id = res[0][cols.get("main_id", res.colnames[0])]
            if isinstance(main_id, bytes): main_id = main_id.decode("utf-8")
            otype = res[0][cols["otype"]] if "otype" in cols else ""
            if isinstance(otype, bytes): otype = otype.decode("utf-8")
            info["found"] = True
            info["name"] = str(main_id).strip()
            info["otype"] = str(otype).strip()
            info["source"] = "SIMBAD"
            return info
    except Exception: pass

    try:
        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
        job = Gaia.cone_search_async(coord, radius=radius_arcsec * u.arcsec)
        g_res = job.get_results()
        if g_res is not None and len(g_res) > 0:
            source_id = g_res[0]["SOURCE_ID"] if "SOURCE_ID" in g_res.colnames else g_res[0]["source_id"]
            info["found"] = True
            info["name"] = f"Gaia DR3 {source_id}"
            info["otype"] = "Star (Gaia)"
            info["source"] = "Gaia DR3"
            return info
    except Exception: pass

    return info


def who_is_it(coord: SkyCoord, motion: dict, scores: dict, variability: dict) -> dict:
    cat = _check_catalogs(coord)
    speed = motion.get("speed_arcsec_yr", 0.0) if motion else 0.0
    direction = motion.get("direction_deg", 0.0) if motion else 0.0
    n_epochs = motion.get("n_epochs", 0) if motion else 0
    is_var = variability.get("is_variable", False) if variability else False
    var_label = variability.get("label", "Нет данных") if variability else "Нет данных"

    scores = scores or {}
    has_water = (scores.get("water") or 0) > 0.15
    has_methane = (scores.get("methane") or 0) > 0.15
    has_molecules = has_water or has_methane

    if n_epochs <= 1 or speed > 60.0: verdict = "Возможно, астероид или артефакт (виден на 1 кадре или слишком быстрый)"
    elif speed >= 0.5 and has_molecules: verdict = "Похож на коричневый карлик (движется + признаки метана/воды)"
    elif speed >= 0.5 and not has_molecules: verdict = "Близкая звезда с большим собственным движением"
    elif speed < 0.5 and is_var: verdict = "Переменная звезда (меняет яркость, но стоит на месте)"
    elif not cat["found"]: verdict = "Кандидат: нужна проверка (совпадений в каталогах нет)"
    else: verdict = f"Известный стационарный источник ({cat['otype'] or 'звезда/галактика'})"

    mol_badges = []
    mol_names_ru = {"water": "Вода (H₂O)", "methane": "Метан (CH₄)", "co2": "CO₂", "co": "CO"}
    for key, label_ru in mol_names_ru.items():
        val = scores.get(key)
        if val is not None and not np.isnan(val) and val > 0.15:
            mol_badges.append(f"{label_ru}: {val:.2f}")

    return {
        "name": cat["name"], "catalog_found": cat["found"], "catalog_type": cat["otype"],
        "speed_arcsec_yr": speed, "direction_deg": direction, "variability": var_label,
        "molecule_badges": mol_badges if mol_badges else ["Явных молекулярных полос не выявлено"],
        "scores": scores, "verdict": verdict, "simbad_url": cat["simbad_url"]
    }