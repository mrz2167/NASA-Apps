# AI disclosure: parts of this file were written with AI assistance (Claude, Anthropic), then reviewed and tested by the LUMEN team. See README, "AI use".
"""Built skies on disk, so a restart (every deploy) doesn't rebuild the demo fields from hundreds of FITS files.

A "sky" (main._load_sky / sky.get_sky) holds the reprojected frames plus the raw cutouts the spectrum and
lens code still need. Everything pickles except each cutout's spectral WCS: SPHEREx stores wavelengths as a
-TAB lookup table that astropy can't rebuild from a pickle. Wavelength is linear in pixel position over a
cutout, so we sample it once on the pixel grid and interpolate (exact to ~1e-16, checked on R Dra).

Files: cache/skies/<key>_<code hash>.pkl. The hash covers the modules that build skies, so changing how
frames are made invalidates old files on its own. SKYWATCH_SKY_DISK=0 turns the disk cache off.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import threading
import time
from pathlib import Path

import numpy as np
import astropy.units as u
from scipy.ndimage import map_coordinates

from data import CACHE

DIR = CACHE / "skies"
MAX_AGE_S = 7 * 24 * 3600          # new SPHEREx passes arrive slowly; a week-old field is still fine
ENABLED = os.environ.get("SKYWATCH_SKY_DISK", "1") != "0"
MAX_BYTES = float(os.environ.get("SKYWATCH_SKY_DISK_GB", "2.5")) * 1e9   # ~40 demo-sized fields


def _code_hash() -> str:
    h = hashlib.sha1(b"sky-disk-v1")
    for name in ("data.py", "epochs.py", "sky.py", "skycache.py"):
        p = Path(__file__).resolve().parent / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:10]


CODE_HASH = _code_hash()


class GridSpectralWCS:
    """Stand-in for a cutout's spectral WCS: wavelength and bandwidth (µm) sampled on the pixel grid."""

    def __init__(self, wl: np.ndarray, bw: np.ndarray):
        self.wl = wl.astype(np.float64)
        self.bw = bw.astype(np.float64)

    @classmethod
    def from_wcs(cls, swcs, shape) -> "GridSpectralWCS":
        ny, nx = shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        wl, bw = swcs.pixel_to_world_values(xx.astype(float), yy.astype(float))
        units = [str(x) for x in swcs.world_axis_units]
        k = 1e6 if units and units[0] == "m" else 1.0          # keep everything in µm
        return cls(np.asarray(wl) * k, np.asarray(bw) * k)

    def _at(self, grid, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        v = map_coordinates(grid, [y.ravel(), x.ravel()], order=1, mode="nearest").reshape(x.shape)
        return v if v.ndim else v[()]

    def pixel_to_world_values(self, x, y):
        return self._at(self.wl, x, y), self._at(self.bw, x, y)

    def pixel_to_world(self, x, y):
        wl, bw = self.pixel_to_world_values(x, y)
        return wl * u.um, bw * u.um


def _path(key: tuple) -> Path:
    name = "_".join(str(k) for k in key).replace("-", "m").replace(".", "p")
    return DIR / f"{name}_{CODE_HASH}.pkl"


def get(key: tuple) -> dict | None:
    if not ENABLED:
        return None
    p = _path(key)
    try:
        if not p.exists() or time.time() - p.stat().st_mtime > MAX_AGE_S:
            return None
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception as e:                       # half-written or from another numpy: rebuild instead
        print(f"[skycache] {p.name}: unreadable ({e!r}), rebuilding", flush=True)
        return None


def _slim(sky: dict) -> dict:
    out = dict(sky)
    cuts = []
    for c in sky.get("cutouts", []):
        c2 = {k: v for k, v in c.items() if k not in ("spectral_wcs", "header")}
        sw = c.get("spectral_wcs")
        c2["spectral_wcs"] = sw if isinstance(sw, GridSpectralWCS) else GridSpectralWCS.from_wcs(sw, c["image"].shape)
        cuts.append(c2)
    out["cutouts"] = cuts
    return out


def _prune(keep: Path) -> None:
    """Keep the folder under MAX_BYTES: drop files with an outdated code hash first, then the oldest."""
    files = [f for f in DIR.glob("*.pkl") if f != keep]
    stale = [f for f in files if not f.stem.endswith(CODE_HASH)]
    for f in stale:
        f.unlink(missing_ok=True)
    files = sorted((f for f in files if f not in stale and f.exists()), key=lambda f: f.stat().st_mtime)
    total = sum(f.stat().st_size for f in files) + (keep.stat().st_size if keep.exists() else 0)
    while files and total > MAX_BYTES:
        f = files.pop(0)
        total -= f.stat().st_size
        f.unlink(missing_ok=True)


def _write(key: tuple, slim: dict) -> None:
    try:
        DIR.mkdir(parents=True, exist_ok=True)
        p = _path(key)
        tmp = p.with_suffix(f".tmp{threading.get_ident()}")
        with open(tmp, "wb") as f:
            pickle.dump(slim, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, p)
        for old in DIR.glob(p.name.rsplit("_", 1)[0] + "_*.pkl"):   # older code hashes of this field
            if old != p:
                old.unlink(missing_ok=True)
        _prune(keep=p)
    except Exception as e:
        print(f"[skycache] could not save {key}: {e!r}", flush=True)


def put(key: tuple, sky: dict, background: bool = True) -> dict:
    """Save a freshly built sky and return its slim form, which the caller should keep in memory instead.

    The spectral WCS grids are sampled here, in the caller's thread (astropy WCS objects aren't safe to share
    between threads); only the file write goes to the background, so the request that built it doesn't wait.
    """
    try:
        slim = _slim(sky)
    except Exception as e:
        print(f"[skycache] could not prepare {key}: {e!r}", flush=True)
        return sky
    if ENABLED:
        if background:
            threading.Thread(target=_write, args=(key, slim), name="skycache", daemon=True).start()
        else:
            _write(key, slim)
    return slim
