"""Пять тестовых объектов NASA с позициями и собственным движением.

    coord_at(name, when="2025-11-15") -> SkyCoord на нужную дату

Источник: SIMBAD (зеркало simbad.harvard.edu — основной сервер CDS бывает недоступен).
Barnard, eps Ind B, R Dra — Gaia DR3 через SIMBAD; WISE 0855 — Kirkpatrick et al. 2021 через SIMBAD;
RW Aur — Gaia DR3 напрямую (в SIMBAD для него устаревшие данные Hipparcos).
Значения записаны в файл, чтобы всё работало без сети; simbad_coord() берёт свежие из SIMBAD.
Середина данных SPHEREx (май 2025 – май 2026) ≈ ноябрь 2025, поэтому дата по умолчанию такая:
объект попадает почти в центр вырезки во всех эпохах.
"""
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

DEFAULT_WHEN = "2025-11-15"

TARGETS = {
    "barnard": dict(
        title="Звезда Барнарда", simbad="Barnard's Star", catalog_en="SIMBAD · Barnard's Star · M4V red dwarf", kind="близкая звезда, самое быстрое собственное движение",
        gaia_dr3=4472832130942575872, ra=269.44850252543836, dec=4.739420051112412,
        pmra=-801.5509783684709, pmdec=10362.394206546573, parallax=546.975939730948, epoch=2016.0,
    ),
    "eps_ind_b": dict(
        title="Эпсилон Индейца B", simbad="eps Ind B", catalog_en="SIMBAD · eps Ind Ba + Bb · T1 + T6 brown dwarfs", kind="пара коричневых карликов у звезды eps Ind A",
        gaia_dr3=6412596012146801152, ra=331.07645259656397, dec=-56.79381206688221,
        pmra=3981.97666296105, pmdec=-2466.8318147663504, parallax=270.6580324694526, epoch=2016.0,
    ),
    "wise0855": dict(
        title="WISE 0855−0714", simbad="WISE J085510.83-071442.5", catalog_en="SIMBAD · WISE J085510.83−071442.5 · Y4 brown dwarf", kind="самый холодный известный коричневый карлик",
        gaia_dr3=None, ra=133.81890825, dec=-7.247048,
        pmra=-8123.7, pmdec=673.2, parallax=439.0, epoch=2000.0,
        when="2026-01-01",  # под эту дату скачан кэш (центр вырезки)
    ),
    "rw_aur": dict(
        title="RW Возничего", simbad="RW Aur", catalog_en="SIMBAD · RW Aur · T Tauri variable", kind="молодая переменная звезда (тип T Тельца)",
        gaia_dr3=156430822114424576, ra=76.95608119552952, dec=30.401204589097233,
        pmra=2.6406431770317083, pmdec=-27.397582540455325, parallax=6.406249329219477, epoch=2016.0,
    ),
    "r_dra": dict(
        title="R Дракона", simbad="R Dra", catalog_en="SIMBAD · R Dra · Mira variable", kind="переменная звезда типа Миры",
        gaia_dr3=1648328016354496384, ra=248.1673783672889, dec=66.75500674547219,
        pmra=-20.642811719160797, pmdec=6.148818852711324, parallax=1.2509180143740992, epoch=2016.0,
    ),
}


SIMBAD_SERVER = "simbad.harvard.edu"


def simbad():
    """Клиент SIMBAD, настроенный на зеркало Гарварда."""
    from astroquery.simbad import Simbad, conf
    conf.server = SIMBAD_SERVER
    s = Simbad()
    s.add_votable_fields("pmra", "pmdec", "plx_value")
    return s


def simbad_coord(name: str, when: str = DEFAULT_WHEN) -> SkyCoord:
    """Любой объект по имени из SIMBAD -> координаты на дату when (позиции SIMBAD даны на J2000)."""
    r = simbad().query_object(name)
    if r is None or len(r) == 0:
        raise ValueError(f"SIMBAD не знает объект {name!r}")
    r = r[0]
    pmra = float(r["pmra"]) if r["pmra"] is not None and np.isfinite(r["pmra"]) else 0.0
    pmdec = float(r["pmdec"]) if r["pmdec"] is not None and np.isfinite(r["pmdec"]) else 0.0
    plx = float(r["plx_value"]) if r["plx_value"] is not None and np.isfinite(r["plx_value"]) and r["plx_value"] > 0 else None
    c = SkyCoord(ra=float(r["ra"]) * u.deg, dec=float(r["dec"]) * u.deg,
                 pm_ra_cosdec=pmra * u.mas / u.yr, pm_dec=pmdec * u.mas / u.yr,
                 distance=(1000 / plx) * u.pc if plx else 1 * u.kpc, obstime=Time("J2000"))
    return c.apply_space_motion(new_obstime=Time(when))


def coord_at(name: str, when: str | None = None) -> SkyCoord:
    """Координаты объекта на дату when с учётом собственного движения.
    По умолчанию — дата, под которую скачан кэш (центр вырезки)."""
    t = TARGETS[name]
    when = when or t.get("when", DEFAULT_WHEN)
    c = SkyCoord(ra=t["ra"] * u.deg, dec=t["dec"] * u.deg,
                 pm_ra_cosdec=t["pmra"] * u.mas / u.yr, pm_dec=t["pmdec"] * u.mas / u.yr,
                 distance=(1000 / t["parallax"]) * u.pc, obstime=Time(t["epoch"], format="jyear"))
    return c.apply_space_motion(new_obstime=Time(when))


def predicted_track(name: str, dates=None):
    """Где объект должен быть по каталогу (для следа на анимации).
    dates=None -> функция date -> SkyCoord (её можно передать прямо в render: track=predicted_track("barnard"))."""
    if dates is None:
        return lambda d: coord_at(name, d)
    return [coord_at(name, d) for d in dates]
