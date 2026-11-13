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

**Query:** `ra`, `dec`, `fov` (degrees, e.g. 0.1), `detector` (1–6), `mode` (always `blink` for now)

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
- **`sources`** (optional): bright sources in pixel coordinates of the grid, so users can click them.
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
  "molecules": {"water": 0.42, "methane": 0.61, "co2": 0.08, "co": 0.22},
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
| `molecules` | Absorption index 0–1 per molecule; ≥ 0.2 shows as "SIGN" | molecules.py |
| `spectrum` | One point per SPHEREx exposure, wavelength in µm, any flux unit | spectrum.py |

The frontend writes the verdict itself from these numbers (moving + methane → "Looks like a cold brown dwarf", varies but still → "Variable star", etc.).

---

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

## Notes

- Cache everything (`functools.lru_cache` or files on disk): the archive search takes 30–40 s.
- If the API fails, the frontend shows a toast and falls back to the simulated sky, so the demo never goes blank.
- Coordinates in the link work out of the box: `index.html?ra=133.795134&dec=-7.245147&fov=0.1`.
