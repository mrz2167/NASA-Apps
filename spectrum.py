from __future__ import annotations

import numpy as np
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

from data import bad_pixels

# Окна поглощения и континуума в микронах
BANDS = {
    "water":   {"band": (2.8, 3.1),  "cont": [(2.2, 2.4), (3.6, 3.8)]},
    "methane": {"band": (3.2, 3.4),  "cont": [(2.9, 3.0), (3.9, 4.0)]},
    "co2":     {"band": (4.2, 4.35), "cont": [(4.0, 4.1), (4.45, 4.55)]},
    "co":      {"band": (4.55, 4.8), "cont": [(4.45, 4.5), (4.85, 4.95)]},
}


def measure_flux(image: np.ndarray, x: float, y: float, r: float = 2.0, r_in: float = 4.0, r_out: float = 6.0) -> float:
    """Апертурная фотометрия: сумма в кружке радиусом r минус медианный фон из кольца [r_in, r_out]."""
    ap = CircularAperture((x, y), r=r)
    an = CircularAnnulus((x, y), r_in=r_in, r_out=r_out)
    bkg_vals = an.to_mask(method="center").get_values(image)
    bkg = np.nanmedian(bkg_vals)
    total = aperture_photometry(image, ap)["aperture_sum"][0]
    return float(total - bkg * ap.area)


def extract_spectrum(cutouts: list[dict], ra: float, dec: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Собирает спектр (длина волны в мкм, поток) по списку вырезок из data.load_cutouts().
    """
    wavelengths: list[float] = []
    fluxes: list[float] = []

    for c in cutouts:
        img = c["image"].astype(np.float64)
        if c.get("zodi") is not None:
            img = img - c["zodi"].astype(np.float64)

        x_raw, y_raw = c["wcs"].world_to_pixel_values(ra, dec)
        x, y = float(x_raw), float(y_raw)
        h, w = img.shape

        # Пропускаем, если объект вне кадра или слишком близко к краю (нужен отступ 6 пикс. для кольца фона)
        if not (6 <= x < w - 6 and 6 <= y < h - 6):
            continue

        ix, iy = int(round(x)), int(round(y))
        if not np.isfinite(img[iy, ix]):
            continue

        # Проверяем центральный пиксель на критические ошибки (NaN / сбойные биты)
        if "flags" in c and c["flags"] is not None:
            mask = bad_pixels(c)
            if mask[iy, ix]:
                continue

        flux = measure_flux(img, x, y)
        wl_raw, _ = c["spectral_wcs"].pixel_to_world(x, y)
        wl = float(wl_raw.to_value("um")) if hasattr(wl_raw, "to_value") else float(wl_raw)

        if np.isfinite(flux) and np.isfinite(wl):
            wavelengths.append(wl)
            fluxes.append(flux)

    if not wavelengths:
        return np.array([]), np.array([])

    order = np.argsort(wavelengths)
    return np.array(wavelengths)[order], np.array(fluxes)[order]


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
    """Считает глубину провалов поглощения: 1 - (поток в окне / поток континуума)."""
    wl = np.asarray(wl)
    flux = np.asarray(flux)
    scores: dict[str, float] = {}

    for name, b in BANDS.items():
        in_mask = (wl > b["band"][0]) & (wl < b["band"][1])
        inband = np.nanmedian(flux[in_mask]) if np.any(in_mask) else np.nan

        cont_parts = [flux[(wl > lo) & (wl < hi)] for lo, hi in b["cont"]]
        cont_parts = [p for p in cont_parts if len(p) > 0]
        cont = np.nanmedian(np.concatenate(cont_parts)) if cont_parts else np.nan

        if np.isfinite(cont) and cont > 0 and np.isfinite(inband):
            scores[name] = float(1.0 - inband / cont)
        else:
            scores[name] = np.nan

    return scores


if __name__ == "__main__":
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    import matplotlib.pyplot as plt
    from data import find_images, load_cutouts

    # Главный целевой объект проекта: коричневый карлик WISE 0855-0714 (эпоха 2025-2026)
    RA, DEC = 133.768, -7.239
    print("Ищем снимки SPHEREx для WISE 0855...")
    tbl = find_images(RA, DEC, size_deg=0.05)
    print(f"Найдено экспозиций: {len(tbl)}")

    cutouts = load_cutouts(tbl[:80])
    wl, flux = extract_spectrum(cutouts, RA, DEC)
    scores = molecule_scores(wl, flux)

    print(f"Собрано валидных точек спектра: {len(wl)}")
    print("Индексы молекул:", scores)

    if len(wl) > 0:
        bw, bf = bin_spectrum(wl, flux, bin_width=0.1)
        plt.figure(figsize=(9, 4))
        plt.scatter(wl, flux, s=14, alpha=0.5, label="Отдельные вырезки")
        plt.plot(bw, bf, color="#1f4e78", linewidth=2, label="Усреднённый спектр")
        plt.axvspan(2.8, 3.1, color="cyan", alpha=0.2, label="Вода (3.0 мкм)")
        plt.axvspan(3.2, 3.4, color="orange", alpha=0.2, label="Метан (3.3 мкм)")
        plt.xlabel("Длина волны (мкм)")
        plt.ylabel("Поток (МЯн/ср · пикс²)")
        plt.title("Реальный спектр SPHEREx: WISE 0855-0714")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()