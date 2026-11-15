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
