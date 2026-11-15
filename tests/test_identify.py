"""Source finding on awkward images (no network)."""
from __future__ import annotations

import numpy as np

from identify import _centroid


def test_centroid_survives_all_nan_core():
    """Vega, Sirius, Orion…: the core around the star is all NaN; this used to be a 500 on /api/frames."""
    img = np.random.default_rng(0).normal(0, 1, (40, 40))
    img[14:27, 14:27] = np.nan
    c = _centroid(img, 20.0, 20.0)
    assert c is None or all(np.isfinite(c))


def test_molecule_snr_flags_noise_and_missing_bands():
    import numpy as np
    from spectrum import molecule_snr
    wl = np.array([2.3, 2.35, 2.95, 3.25, 3.3, 3.35, 3.7, 3.95, 3.97])
    clean = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0])
    s = molecule_snr(wl, clean, np.full(len(wl), 0.01))
    assert s["methane"] > 3                      # deep, precise dip: significant
    assert s["co2"] is None and s["co"] is None  # no points at 4-5 µm: not measured
    noisy = molecule_snr(wl, clean, np.full(len(wl), 0.5))
    assert noisy["methane"] < 3                  # same dip, huge errors: within noise
