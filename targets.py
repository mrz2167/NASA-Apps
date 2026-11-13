"""Пять тестовых объектов NASA с позициями и собственным движением.

    coord_at(name, when="2025-11-15") -> SkyCoord на нужную дату

Источник: Gaia DR3 (эпоха 2016.0), для WISE 0855 — позиция из обозначения WISE (эпоха ~2010.5)
и собственное движение из Luhman & Esplin 2016 (в Gaia его нет — слишком холодный).
Середина данных SPHEREx (май 2025 – май 2026) ≈ ноябрь 2025, поэтому дата по умолчанию такая:
объект попадает почти в центр вырезки во всех эпохах.
"""
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

DEFAULT_WHEN = "2025-11-15"

TARGETS = {
    "barnard": dict(
        title="Звезда Барнарда", simbad="Barnard's Star", kind="близкая звезда, самое быстрое собственное движение",
        gaia_dr3=4472832130942575872, ra=269.44850252543836, dec=4.739420051112412,
        pmra=-801.5509783684709, pmdec=10362.394206546573, parallax=546.975939730948, epoch=2016.0,
    ),
    "eps_ind_b": dict(
        title="Эпсилон Индейца B", simbad="eps Ind B", kind="пара коричневых карликов у звезды eps Ind A",
        gaia_dr3=6412596012146801152, ra=331.07645259656397, dec=-56.79381206688221,
        pmra=3981.97666296105, pmdec=-2466.8318147663504, parallax=270.6580324694526, epoch=2016.0,
    ),
    "wise0855": dict(
        title="WISE 0855−0714", simbad="WISE J085510.83-071442.5", kind="самый холодный известный коричневый карлик",
        gaia_dr3=None, ra=133.795125, dec=-7.245139,
        pmra=-8118.0, pmdec=680.0, parallax=438.0, epoch=2010.5,
    ),
    "rw_aur": dict(
        title="RW Возничего", simbad="RW Aur", kind="молодая переменная звезда (тип T Тельца)",
        gaia_dr3=156430822114424576, ra=76.95608119552952, dec=30.401204589097233,
        pmra=2.6406431770317083, pmdec=-27.397582540455325, parallax=6.406249329219477, epoch=2016.0,
    ),
    "r_dra": dict(
        title="R Дракона", simbad="R Dra", kind="переменная звезда типа Миры",
        gaia_dr3=1648328016354496384, ra=248.1673783672889, dec=66.75500674547219,
        pmra=-20.642811719160797, pmdec=6.148818852711324, parallax=1.2509180143740992, epoch=2016.0,
    ),
}


def coord_at(name: str, when: str = DEFAULT_WHEN) -> SkyCoord:
    """Координаты объекта на дату when с учётом собственного движения."""
    t = TARGETS[name]
    c = SkyCoord(ra=t["ra"] * u.deg, dec=t["dec"] * u.deg,
                 pm_ra_cosdec=t["pmra"] * u.mas / u.yr, pm_dec=t["pmdec"] * u.mas / u.yr,
                 distance=(1000 / t["parallax"]) * u.pc, obstime=Time(t["epoch"], format="jyear"))
    return c.apply_space_motion(new_obstime=Time(when))
