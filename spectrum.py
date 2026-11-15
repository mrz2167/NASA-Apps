from __future__ import annotations

import numpy as np
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

from data import bad_pixels

# Окна поглощения и континуума в микронах
BANDS = {
    "water":   {"band": (2.8, 3.1),  "cont": [(2.2, 2.4), (3.6, 3.8)]},
    # метан: опорные окна вне полосы воды (2.7–3.1 мкм), иначе континуум занижен и индекс тоже;
    # окна далеко друг от друга, поэтому континуум — степенной закон между ними (у обычной звезды ≈ 0)
    "methane": {"band": (3.2, 3.4),  "cont": [(2.05, 2.2), (3.85, 4.05)], "powerlaw": True},
    "co2":     {"band": (4.2, 4.35), "cont": [(4.0, 4.1), (4.45, 4.55)]},
    "co":      {"band": (4.55, 4.8), "cont": [(4.45, 4.5), (4.85, 4.95)]},
}


def measure_flux(
    image: np.ndarray,
    x: float,
    y: float,
    variance: np.ndarray | None = None,
    r: float = 2.0,
    r_in: float = 4.0,
    r_out: float = 6.0,
) -> tuple[float, float]:
    """Апертурная фотометрия: возвращает (поток, погрешность) в кружке радиусом r с вычетом фона кольца."""
    ap = CircularAperture((x, y), r=r)
    an = CircularAnnulus((x, y), r_in=r_in, r_out=r_out)
    bkg_vals = an.to_mask(method="center").get_values(image)
    bkg = float(np.nanmedian(bkg_vals))
    total = float(aperture_photometry(image, ap)["aperture_sum"][0])
    flux = total - bkg * ap.area

    if variance is not None and np.any(np.isfinite(variance)):
        var_sum = float(aperture_photometry(np.nan_to_num(variance, nan=0.0), ap)["aperture_sum"][0])
        err = float(np.sqrt(max(var_sum, 0.0)))
    else:
        bkg_std = float(np.nanstd(bkg_vals))
        err = float(bkg_std * np.sqrt(ap.area)) if np.isfinite(bkg_std) else 0.01

    return float(flux), float(err)


def extract_spectrum_with_errors(
    cutouts: list[dict],
    ra: float,
    dec: float,
    valid_detectors: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Собирает спектр (длина волны в мкм, поток, погрешность) по списку вырезок из data.load_cutouts().
    При передаче valid_detectors (например, [5, 6]) фильтрует только нужные детекторы.
    """
    wavelengths: list[float] = []
    fluxes: list[float] = []
    errors: list[float] = []

    for c in cutouts:
        if valid_detectors is not None and c.get("detector") not in valid_detectors:
            continue

        img = c["image"].astype(np.float64)
        if c.get("zodi") is not None:
            img = img - c["zodi"].astype(np.float64)

        var = c.get("variance")
        if var is not None:
            var = var.astype(np.float64)

        x_raw, y_raw = c["wcs"].world_to_pixel_values(ra, dec)
        x, y = float(x_raw), float(y_raw)
        h, w = img.shape

        if not (6 <= x < w - 6 and 6 <= y < h - 6):
            continue

        ix, iy = int(round(x)), int(round(y))
        if not np.isfinite(img[iy, ix]):
            continue

        if "flags" in c and c["flags"] is not None:
            mask = bad_pixels(c)
            if mask[iy, ix]:
                continue

        flux, err = measure_flux(img, x, y, variance=var)
        wl_raw, _ = c["spectral_wcs"].pixel_to_world(x, y)
        wl = float(wl_raw.to_value("um")) if hasattr(wl_raw, "to_value") else float(wl_raw)

        if np.isfinite(flux) and np.isfinite(wl):
            wavelengths.append(wl)
            fluxes.append(flux)
            errors.append(err if np.isfinite(err) else 0.01)

    if not wavelengths:
        return np.array([]), np.array([]), np.array([])

    order = np.argsort(wavelengths)
    return (
        np.array(wavelengths)[order],
        np.array(fluxes)[order],
        np.array(errors)[order],
    )


def extract_spectrum(
    cutouts: list[dict],
    ra: float,
    dec: float,
    valid_detectors: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (wavelength, flux) с поддержкой фильтрации valid_detectors."""
    wl, flux, _ = extract_spectrum_with_errors(cutouts, ra, dec, valid_detectors=valid_detectors)
    return wl, flux


def get_spectrum(
    cutouts: list[dict],
    ra: float,
    dec: float,
    valid_detectors: list[int] | None = None,
) -> dict[str, list[float]]:
    """
    Формат строго по контракту API.md для эндпоинта GET /api/object:
    {"wavelength": [...], "flux": [...], "error": [...]}
    """
    wl, flux, err = extract_spectrum_with_errors(cutouts, ra, dec, valid_detectors=valid_detectors)
    return {
        "wavelength": [round(float(v), 4) for v in wl],
        "flux": [round(float(v), 6) for v in flux],
        "error": [round(float(v), 6) for v in err],
    }


def bin_spectrum(wl: np.ndarray, flux: np.ndarray, bin_width: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Усредняет (медианой) точки спектра в узких интервалах длин волн для чистого графика."""
    if len(wl) == 0:
        return np.array([]), np.array([])
    edges = np.arange(np.floor(wl.min() * 10) / 10, np.ceil(wl.max() * 10) / 10 + bin_width, bin_width)
    b_wl, b_flux = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (wl >= lo) & (wl < hi)
        if np.any(m):
            b_wl.append(float(np.nanmedian(wl[m])))
            b_flux.append(float(np.nanmedian(flux[m])))
    return np.array(b_wl), np.array(b_flux)


def molecule_scores(wl: np.ndarray, flux: np.ndarray) -> dict[str, float]:
    """
    Считает индекс поглощения 0..1 по контракту API.md (без NaN, чтобы не ломать JSON):
    {"water": ..., "methane": ..., "co2": ..., "co": ...}
    """
    wl = np.asarray(wl)
    flux = np.asarray(flux)
    scores: dict[str, float] = {}

    for name, b in BANDS.items():
        r = _band_index(wl, flux, None, b)
        scores[name] = round(float(np.clip(r[0], 0.0, 1.0)), 3) if r else 0.0

    return scores


def _median_err(flux, err, m):
    """Медиана потока в маске и её погрешность (≈ 1.25·σ/√n); без ошибок — (медиана, nan)."""
    f = float(np.nanmedian(flux[m]))
    if err is None:
        return f, np.nan
    return f, float(1.2533 * np.sqrt(np.nanmean(err[m] ** 2) / m.sum()))


def _band_index(wl, flux, err, b) -> tuple[float, float] | None:
    """(индекс полосы 1 − F_полосы / F_континуума, его погрешность) или None, если нет точек.

    Континуум: обычно — общая медиана опорных окон; для "powerlaw" — степенной закон между медианами
    двух окон в центре полосы (нужны оба окна)."""
    mb = (wl > b["band"][0]) & (wl < b["band"][1])
    masks = [(wl > lo) & (wl < hi) for lo, hi in b["cont"]]
    if not mb.any():
        return None
    fb, eb = _median_err(flux, err, mb)
    if b.get("powerlaw"):
        if not all(m.any() for m in masks):
            return None
        (f1, e1), (f2, e2) = (_median_err(flux, err, m) for m in masks)
        if not (f1 > 0 and f2 > 0):
            return None
        x1, x2 = (sum(w) / 2 for w in b["cont"])
        t = np.log(sum(b["band"]) / 2 / x1) / np.log(x2 / x1)
        fc = float(np.exp((1 - t) * np.log(f1) + t * np.log(f2)))
        ec = fc * float(np.hypot((1 - t) * e1 / f1, t * e2 / f2))
    else:
        mc = np.any(masks, axis=0)
        if not mc.any():
            return None
        fc, ec = _median_err(flux, err, mc)
    if not (np.isfinite(fc) and fc > 0 and np.isfinite(fb)):
        return None
    return 1.0 - fb / fc, float(np.hypot(eb / fc, fb * ec / fc ** 2))


MIN_SNR = 3.0   # индекс считается признаком, только если он больше 3σ своей погрешности


def molecule_snr(wl, flux, err) -> dict[str, float | None]:
    """Значимость индекса каждой молекулы: индекс / его погрешность (по ошибкам точек спектра).

    None — в полосе или континууме нет ни одной точки (не измерено). Погрешность медианы ≈ 1.25·σ/√n.
    """
    wl, flux, err = (np.asarray(v, float) for v in (wl, flux, err))
    out: dict[str, float | None] = {}
    for name, b in BANDS.items():
        r = _band_index(wl, flux, err, b) if len(wl) else None
        out[name] = round(r[0] / r[1], 1) if r and r[1] > 0 and np.isfinite(r[1]) else None
    return out


if __name__ == "__main__":
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    from data import find_images, load_cutouts

    RA, DEC = 133.768, -7.239
    tbl = find_images(RA, DEC, size_deg=0.05)
    cutouts = load_cutouts(tbl[:80])

    spec_all = get_spectrum(cutouts, RA, DEC)
    spec_d56 = get_spectrum(cutouts, RA, DEC, valid_detectors=[5, 6])

    print("Points (all detectors):", len(spec_all["wavelength"]))
    print("Points (D5, D6 only):", len(spec_d56["wavelength"]))