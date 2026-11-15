# LUMEN Skywatch — API contract for the Python team

The frontend (`index.html`) runs on a simulated sky until you connect a backend.
To switch to real SPHEREx data, set one line at the top of the script in `index.html`:

```js
const CONFIG = { API_BASE: "https://your-api-host", PIX_ARCSEC: 6.15 };
```

The backend is a small FastAPI app with **two endpoints**. All coordinates are degrees (ICRS).
Enable CORS for the site's domain (or `*` during the hackathon).

---

## 1. `GET /api/frames`

Returns the blink frames for one detector, already stacked per epoch and reprojected onto one grid.

**Query:** `ra` (0–360), `dec` (−90…+90), `fov` (degrees, more than 0.01 and at most 0.5; the site uses 0.05/0.1/0.2), `detector` (1–6), `mode` (only `blink` for now)

`cadence` (optional): `pass` (default) — one frame per six-month sky pass; `visit` — each pass split into
observing visits (a new frame after a > 12 h gap in that detector's exposures). SPHEREx sees a spot in
bursts a few days apart, so `visit` gives 7–12 frames instead of 2–4 (deep fields near the ecliptic poles:
capped at 36), each stacked from ~2–6 exposures, so noisier. Labels become `"2025-10-31 · Frame 2"`.
`meta.n_exposures` lists the exposures per frame. `/api/track` takes the same `cadence` and returns
positions in the same frame order.

**Response:**

```json
{
  "size_px": 58,
  "pixel_arcsec": 6.15,
  "frames": [
    {"label": "2025 · Survey 1", "date": "2025-06-14", "png": "data:image/png;base64,..."},
    {"label": "2025–26 · Survey 2", "date": "2025-12-20", "png": "data:image/png;base64,..."}
  ],
  "diff": [
    {"label": "2025 · Survey 1", "png": "data:image/png;base64,..."},
    {"label": "2025–26 · Survey 2", "png": "data:image/png;base64,..."}
  ],
  "sources": [
    {"x": 12.4, "y": 30.1, "id": "Gaia DR3 1234567890"}
  ],
  "target": {
    "positions": [{"x": 29.0, "y": 29.6}, {"x": 29.1, "y": 28.8}],
    "molecules": {"water": 0.42, "methane": 0.61, "co2": 0.08, "co": 0.22}
  }
}
```

Rules:

- **`frames`**: one PNG per epoch, all the same size (`size_px` × `size_px`), north up, east left. Stretch the brightness yourself (e.g. astropy `ZScaleInterval` or asinh) so the frontend can draw it as is. A PNG URL also works instead of a data URI.
- **`diff`** (optional): frame minus the first frame, coloured however you like (suggested: positive = amber `#F6B544`, negative = blue `#3D5BD9`, zero = `#070A20`). If omitted, "What changed" shows the normal frames.
- **`sources`** (optional): bright sources in pixel coordinates of the grid, so users can click them. Peaks on the flat
  core of a saturated star are merged into one source, and each Gaia DR3 ID is given to at most one source (the
  nearest); the rest are labelled `SPHEREx src N`.
- **`target`** (optional): the tracked object. `positions` has one entry per frame, pixel coordinates on the grid (x right, y down, 0 at the top-left corner). It drives the Trail line and the molecule lens glow.

---

## 2. `GET /api/object`

Everything about the object at a sky position. The frontend converts the clicked pixel to RA/Dec and calls this.

**Query:** `ra`, `dec`

**Response:**

```json
{
  "name": "WISE 0855−0714",
  "status": "known",
  "catalog": "SIMBAD · WISE J085510.83−071442.5 · Y4",
  "motion": {"arcsec_per_yr": 8.0, "pa_deg": 275, "catalog_arcsec_per_yr": 8.1},
  "variability": {"ratio": 1.0},
  "molecules": {"water": 0.477, "methane": 0.205, "co2": 0.0, "co": 0.0},
  "molecules_snr": {"water": 12.0, "methane": 1.4, "co2": -1.8, "co": -1.3},
  "spectrum": {
    "wavelength": [0.75, 0.79, 0.83],
    "flux": [0.001, 0.002, 0.002],
    "error": [0.01, 0.01, 0.01]
  }
}
```

Field by field:

| Field | Meaning | Who computes it |
| --- | --- | --- |
| `name` | Catalogue name, or `null` if nothing matches | identify.py (SIMBAD / Gaia) |
| `status` | `known`, `candidate` (moves, no catalogue match) or `star` (static background star) | identify.py |
| `catalog` | One-line catalogue description, or `null` | identify.py |
| `motion.arcsec_per_yr` | Measured proper motion | identify.py `measure_motion` |
| `motion.pa_deg` | Direction, degrees east of north (0 = N, 90 = E) | identify.py |
| `motion.catalog_arcsec_per_yr` | Catalogue value for comparison, or `null` | identify.py |
| `variability.ratio` | max / min flux across epochs (1.0 = steady) | identify.py |
| `molecules` | Absorption index 0–1 per molecule (0 also when not measured; see `molecules_snr`) | spectrum.py |
| `molecules_snr` | Index divided by its error, per molecule; `null` = no spectrum points in the band or its continuum. The site shows "SIGN" only for index ≥ 0.2 (methane 0.15) **and** ≥ 3σ; above the threshold but below 3σ is "UNSURE" | spectrum.py |
| `spectrum` | One point per SPHEREx exposure, wavelength in µm, any flux unit | spectrum.py |

The frontend writes the verdict itself from these numbers (moving + methane → "Looks like a cold brown dwarf", varies but still → "Variable star", etc.).

---

## Where the data came from, and progress of a new field

`/api/frames` `meta.search` says how the list of exposures was found when the field was built: `irsa`
(NASA's IRSA archive), `aws-index` (our own index of NASA's AWS bucket; IRSA was not answering) or `saved`
(a stored search or a saved field: no archive call). `meta.fresh` is true when this very request built the field.

`GET /api/progress?ra&dec&fov&detector&cadence` (same values as the pending `/api/frames` call) never builds
anything and answers at once: `{"active": false}` when nothing is being built for the field, otherwise
`{"active": true, "stage", "done", "total", "elapsed_s"}` with `stage` one of `search`, `irsa`, `index`,
`download` (cutouts fetched, `done`/`total`), `saved` (cutouts read from our disk, `done`/`total`), `build`.
The site polls it once a second while a frames request is pending.

---

## WISE/NEOWISE frames: `source=wise`

`/api/frames?ra&dec&fov&source=wise` returns 20–23 frames of WISE/NEOWISE W2 (4.6 µm), one per six-month
WISE pass from 2010 to 2024 (unTimely coadds via WiseView; `wise.py`), in the same response shape:
`pixel_arcsec` = 2.75, `center` = pixel position of (ra, dec) for click→sky conversion, `target.positions`
= the centre object's path (known targets follow their catalogue motion, refined on the image), `meta.credit`.
`detector`/`cadence` are ignored; there are no molecule maps. `/api/track?…&source=wise&frame=N` follows a
clicked spot from frame N forwards and backwards. Cached in `cache/wise/`.

## 3. `GET /api/lens`

Per-pixel molecule maps for the molecule lenses: where on the picture the light dips in the water, methane,
CO₂ or CO band more than for the neighbouring stars.

**Query:** `ra`, `dec`, `fov` — same as `/api/frames`; the maps lie on the same grid and orientation.

Optional `ora`, `odec`: an object the user clicked. If it is too faint for the per-pixel map, its disc (~2 px) is
painted from its own spectrum indices (the same numbers as its `/api/object` card), only where the map is empty.
The paint is kept for the field, so later responses include it.

**Response:**

```json
{
  "size_px": 59,
  "maps": {"water": "data:image/png;base64,...", "methane": "...", "co2": "...", "co": "..."},
  "peak": {"water": 0.51, "methane": 0.77, "co2": 0.26, "co": 0.31},
  "lit":  {"water": 16, "methane": 15, "co2": 9, "co": 7}
}
```

- `maps`: transparent PNGs in the lens colour; opacity grows with band depth (from 0.08, full at ≥ 0.43).
  The site dims the frame and draws the map on top.
- `peak`: deepest band in the field (0–1); `lit`: pixels with depth ≥ 0.2.
- How (`lens.py`): every SPHEREx exposure sees each pixel at its own wavelength, so all exposures of the field
  give each pixel a coarse spectrum. Depth = 1 − band / continuum (straight line between the two continuum
  windows of `spectrum.BANDS`), minus the field's median depth (PSF and calibration offsets shared by all stars).
  Only pixels well above the background, away from saturated stars. A sign, not proof; numbers differ from the
  object card's indices, which use a different continuum.

## Minimal FastAPI skeleton

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

@app.get("/api/frames")
def frames(ra: float, dec: float, fov: float = 0.1, detector: int = 4, mode: str = "blink"):
    # find_images -> load_cutout -> build_frames -> PNG data URIs
    ...

@app.get("/api/object")
def obj(ra: float, dec: float):
    # get_spectrum -> molecule_scores -> measure_motion -> who_is_it
    ...
```

Run locally: `uvicorn main:app --reload`, then set `API_BASE: "http://localhost:8000"` and open `index.html`.

## Errors

- Invalid parameters (out of range, wrong type, unknown `mode`/`cadence`) return **422** with a readable English
  `detail` string, e.g. `{"detail": "Invalid parameter 'ra': Input should be less than or equal to 360 (got '400').",
  "errors": [{"param": "ra", "message": "..."}]}`. The same ranges are listed in `/docs`.
- No SPHEREx data / archive down: **503** with `detail`; detector without frames in the field: **404**.
- All response text (labels, verdicts, warnings, errors) is English.

## Notes

- Cache everything (`functools.lru_cache` or files on disk): the archive search takes 30–40 s.
- If the API fails, the frontend shows a toast and falls back to the simulated sky, so the demo never goes blank.
- Coordinates in the link work out of the box: `index.html?ra=133.795134&dec=-7.245147&fov=0.1`.
