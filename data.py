"""Доступ к снимкам SPHEREx в архиве IRSA (Питонист 1).

    find_images(ra, dec, size_deg)  -> astropy Table: url, obs_id, release, time, detector, wl_min_um, wl_max_um
    load_cutout(url, time=None)     -> dict: image, flags, variance, zodi, wcs, spectral_wcs, time, detector, header
    load_cutouts(table)             -> list[dict] — параллельно, с кэшем на диске

Всё кэшируется в cache/: повторный запуск работает без сети.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time as _time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import conf
from astropy.wcs import WCS, FITSFixedWarning
import astropy.units as u

conf.remote_timeout = 120
warnings.simplefilter("ignore", FITSFixedWarning)
logging.getLogger("astropy").setLevel(logging.ERROR)  # шум про SIP у спектральной сетки

IRSA = "https://irsa.ipac.caltech.edu/"
CACHE = Path(__file__).resolve().parent / "cache"
# Предпочтение при дублях одной экспозиции в разных релизах.
RELEASE_PRIORITY = {"dr1": 0, "qr3": 1, "qr2": 2}

# Биты FLAGS, при которых пиксель считаем плохим (описание — в заголовке FLAGS, ключи MSKN/MSKD).
# OUTLIER и SOURCE специально НЕ включены: движущийся объект может попасть в «выбросы».
BAD_BITS = {
    "TRANSIENT": 0, "OVERFLOW": 1, "SUR_ERROR": 2, "NONFUNC": 6,
    "MISSING_DATA": 9, "HOT": 10, "COLD": 11, "NONLINEAR": 15,
}
BAD_MASK = sum(1 << b for b in BAD_BITS.values())


def _key(*parts) -> str:
    return hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:12]


def _tap(query: str, retries: int = 3) -> Table:
    from astroquery.ipac.irsa import Irsa
    for attempt in range(retries):
        try:
            return Irsa.query_tap(query).to_table()
        except Exception:
            if attempt == retries - 1:
                raise
            _time.sleep(10 * (attempt + 1))


def find_images(ra: float, dec: float, size_deg: float = 0.1, use_cache: bool = True) -> Table:
    """Все снимки SPHEREx, накрывающие точку (ra, dec), отсортированные по времени.

    Столбцы: url (ссылка на вырезку size_deg вокруг точки), obs_id, release (dr1/qr3),
    time (MJD начала экспозиции), detector (1..6), wl_min_um / wl_max_um (диапазон детектора).
    Одна экспозиция из разных релизов остаётся одна (приоритет dr1 > qr3).
    """
    coord = SkyCoord(ra, dec, unit="deg")
    ra, dec = round(coord.ra.deg, 6), round(coord.dec.deg, 6)
    cache_file = CACHE / "queries" / f"{_key(ra, dec, size_deg)}.ecsv"
    if use_cache and cache_file.exists():
        return Table.read(cache_file)

    query = f"""
    SELECT a.uri, p.time_bounds_lower, p.energy_bandpassname,
           p.energy_bounds_lower, p.energy_bounds_upper
    FROM spherex.artifact a
    JOIN spherex.plane p ON a.planeid = p.planeid
    WHERE 1 = CONTAINS(POINT('ICRS', {ra}, {dec}), p.poly)
      AND a.producttype = 'science'
    ORDER BY p.time_bounds_lower
    """
    raw = _tap(query)

    rows = {}
    for r in raw:
        uri = str(r["uri"])
        m = re.search(r"level2_(.+?)_spx", uri)
        obs_id = m.group(1) if m else uri
        release = uri.split("/")[3] if uri.count("/") > 3 else "?"
        row = dict(
            url=f"{IRSA}{uri}?center={ra},{dec}d&size={size_deg}",
            obs_id=obs_id,
            release=release,
            time=float(r["time_bounds_lower"]),
            detector=int(str(r["energy_bandpassname"])[-1]),
            wl_min_um=float(r["energy_bounds_lower"]) * 1e6,
            wl_max_um=float(r["energy_bounds_upper"]) * 1e6,
        )
        old = rows.get(obs_id)
        if old is None or RELEASE_PRIORITY.get(release, 9) < RELEASE_PRIORITY.get(old["release"], 9):
            rows[obs_id] = row

    names = ["url", "obs_id", "release", "time", "detector", "wl_min_um", "wl_max_um"]
    t = Table(rows=[[r[n] for n in names] for r in rows.values()], names=names) if rows else Table(names=names)
    if len(t):
        t.sort("time")
    t.meta.update(ra=ra, dec=dec, size_deg=size_deg)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    t.write(cache_file, overwrite=True)
    return t


def _download(url: str, dest: Path, retries: int = 3) -> None:
    """Скачать вырезку и сохранить облегчённую копию (без тяжёлой таблицы EPSF)."""
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=conf.remote_timeout) as resp:
                raw = resp.read()
            break
        except (HTTPError, URLError, TimeoutError, OSError):
            if attempt == retries - 1:
                raise
            _time.sleep(10 * (attempt + 1))
    import io
    with fits.open(io.BytesIO(raw)) as hdul:
        keep = [hdul[0]] + [h for h in hdul[1:] if h.name in ("IMAGE", "FLAGS", "VARIANCE", "ZODI", "WCS-WAVE")]
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        fits.HDUList([h.copy() for h in keep]).writeto(tmp, overwrite=True)
        tmp.replace(dest)


def cutout_path(url: str) -> Path:
    m = re.search(r"level2_(.+?)_spx", url)
    name = m.group(1) if m else "cutout"
    return CACHE / "cutouts" / f"{name}_{_key(url)}.fits"


def load_cutout(url: str, time: float | None = None, retries: int = 3) -> dict:
    """Ссылка на вырезку -> словарь с данными.

    image    — 2D, MJy/sr; зодиакальный свет НЕ вычтен (он лежит в zodi)
    flags    — 2D int, биты описаны в BAD_BITS; mask = (flags & BAD_MASK) != 0
    variance — 2D дисперсия
    zodi     — 2D модель зодиакального света
    wcs      — небесная WCS
    spectral_wcs — pixel_to_world(x, y) -> (длина волны, ширина полосы)
    time     — MJD (из таблицы поиска, иначе MJD-AVG из заголовка)
    detector — 1..6
    """
    path = cutout_path(url)
    if not path.exists():
        _download(url, path, retries)
    with fits.open(path) as hdul:
        hdr = hdul["IMAGE"].header
        swcs = WCS(header=hdr, fobj=hdul, key="W")
        swcs.sip = None  # для спектральной сетки искажения отключаются
        return {
            "image": hdul["IMAGE"].data.astype(np.float32),
            "flags": hdul["FLAGS"].data,
            "variance": hdul["VARIANCE"].data if "VARIANCE" in hdul else None,
            "zodi": hdul["ZODI"].data if "ZODI" in hdul else None,
            "wcs": WCS(hdr),
            "spectral_wcs": swcs,
            "time": float(time) if time is not None else float(hdr.get("MJD-AVG", hdr.get("MJD-OBS"))),
            "detector": int(hdr["DETECTOR"]),
            "header": hdr,
            "url": url,
        }


def load_cutouts(table: Table, max_workers: int = 8, progress: bool = True) -> list[dict]:
    """Загрузить все вырезки из таблицы find_images (параллельно, через кэш).

    Вырезки, которые не скачались после повторов, пропускаются с предупреждением.
    """
    def one(row):
        try:
            return load_cutout(row["url"], time=row["time"])
        except Exception as e:  # сеть/архив: не роняем всю загрузку
            print(f"  ! пропуск {row['obs_id']}: {e}")
            return None

    out, n = [], len(table)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i, c in enumerate(pool.map(one, table), 1):
            if c is not None:
                out.append(c)
            if progress and (i % 25 == 0 or i == n):
                print(f"  вырезки: {i}/{n}")
    return out


def bad_pixels(cutout: dict) -> np.ndarray:
    """Маска плохих пикселей (True = плохой) по флагам и NaN."""
    return ((cutout["flags"] & BAD_MASK) != 0) | ~np.isfinite(cutout["image"])
