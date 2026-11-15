# LUMEN Skywatch

**NASA Space Apps Challenge 2026 · Planet X and SPHEREx challenge**

**Live site: https://nasa-skywatch-production.up.railway.app/**

LUMEN Skywatch lets anyone open NASA's **SPHEREx** images of any part of the sky and watch them as a short movie, the way astronomers "blink" photos to find things that move. SPHEREx photographs the whole sky in 102 colours of infrared light, so the site can answer two questions: **what is moving** and **what is it made of**.

Students learn the method step by step through 10 levels. Then they explore the real sky on their own and send in what they find.

---

## What you can do on the site

| Feature | What it does |
|---|---|
| **Blink, Trail, What changed** | Real SPHEREx frames from different months, played as a movie. *Trail* draws the measured path of the object you clicked. *What changed* subtracts the first visit, so moving and variable objects stand out. |
| **Molecule lenses (H₂O, CO₂, CO, CH₄)** | Each pixel gets a rough spectrum from every SPHEREx exposure that covers it. The lens colours the image where that spectrum dips inside a molecule's absorption band. |
| **Six detectors (0.75–5.0 µm)** | Look at the same field through each SPHEREx detector. Cold brown dwarfs are nearly invisible at 1 µm and bright at 4–5 µm. |
| **Object card** | Click any dot. The card shows its Gaia DR3 / SIMBAD name or *candidate* status, motion with error bars against the catalogue value, brightness variation, spectrum, molecule signs and a plain-language verdict. |
| **Second survey: WISE / NEOWISE 2010–2024** | The same patch of sky over 14 years, so slow nearby stars visibly move. |
| **Any place in the sky** | Type coordinates or a name (Orion Nebula, M31, Vega). There are also *Famous* buttons and **Surprise me**, which jumps to a random ready field. |
| **Field reports** | A find from free exploration is sent to our server with one click (position, catalogue match, motion, molecules, detector). The team reviews reports in a password-protected table. |
| **Three languages** | English, Russian and Uzbek interface, for schools in Uzbekistan. |

## Learning path: 10 levels

| # | Level | Skill | Real object |
|---|---|---|---|
| 1 | The Power of the Blink | blink comparison | WISE 0855−0714 |
| 2 | False Alarm | real mover vs. glitch | training field |
| 3 | Through the Years | motion over years | Luhman 16 |
| 4 | Ingredients of Space | reading ices from a spectrum | CRBR 2422.8-3423 (CO₂ ice) |
| 5 | Did I Find You First? | catalogue check and report (final exam on real WISE frames) | Gliese 570 D |
| 6 | The Runaway Star | measuring proper motion | Barnard's Star |
| 7 | The Coldest Ghost | detectors and the water band | WISE 0855−0714 |
| 8 | Twins Too Close To Split | chemistry when you can't resolve a pair | Epsilon Indi Ba/Bb |
| 9 | The Breathing Star | variability without motion | RW Aurigae |
| 10 | The Slow Pulse | long-period variability | R Draconis |

Every level ends with a short history fact (Tombaugh and Pluto, helium on the Sun, Backyard Worlds' "The Accident") and the real story of the object the student just found.

## Data sources

All data are public NASA or open archive data. They are fetched when someone opens a field; the site shows no pre-made pictures.

| Data | Used for | Source |
|---|---|---|
| SPHEREx Level 2 spectral images (DR1 and quick releases, 2025–26) | blink frames, detectors, per-pixel spectra, molecule lenses | NASA/IPAC IRSA, read directly from NASA's open AWS bucket [`s3://nasa-irsa-spherex`](https://nasa-irsa-spherex.s3.amazonaws.com/) |
| SPHEREx exposure search | which exposures cover a sky position | [IRSA TAP](https://irsa.ipac.caltech.edu/) (`spherex.plane`, `spherex.artifact`), with our own coverage index as a fallback |
| WISE / NEOWISE unTimely frames, 2010–2024 | long-baseline motion | unTimely coadds via [WiseView](http://byw.tools/wiseview) (Backyard Worlds) |
| Gaia DR3 | star names, catalogue proper motions | IRSA |
| SIMBAD | object names and types | Harvard mirror (`simbad.harvard.edu`) |

## How it works

1. **Find exposures.** The server asks IRSA which SPHEREx exposures cover the requested point. If IRSA is down, it uses our own coverage index instead. We built that index from the headers of about 820,000 exposures in the AWS bucket; it returns 97% of what IRSA returns.
2. **Read only the pixels needed.** SPHEREx files are 2040×2040 images of about 67 MB each. The server uses HTTP range requests to read only the rows around the target (about 1.8 MB) and rebuilds the image, flags, variance, zodiacal light and WCS. The result is pixel-identical to IRSA's own cutout service.
3. **Build epochs.** Exposures are grouped into visits. The server masks bad pixels (overflow, hot, cold, non-linear, transient), subtracts zodiacal light and background, and median-combines each detector per visit onto a common north-up grid (6.15″ pixels).
4. **Measure.**
   - *Motion:* the centroid of the object is fitted in every epoch, and a straight line through those positions gives speed and direction with an error estimate. Catalogue motions are propagated from the Gaia epoch (2016.0) to SPHEREx's epoch (2025.9) so they can be compared directly.
   - *Variability:* the brightness range is compared with the noise level. For saturated stars it is measured in the star's wings.
   - *Molecules:* band depth against a local continuum in each pixel for water, methane, CO₂ and CO. A sign is shown only above 3σ.
   - *Movers:* single-visit frames minus the median of the other visits reveal transients. Detections in successive visits that line up are joined into tracks (slow asteroids and comets).
5. **Explain.** All of this becomes an object card with a plain-language verdict ("Nearby star on the move", "Looks like a cold brown dwarf", "Variable star", "Candidate") that also states the uncertainty.

## Validation on NASA's test objects

| Object | We measured | Catalogue | Verdict |
|---|---|---|---|
| Barnard's Star | 10.3 ± 3.7″/yr, 352° | 10.4″/yr, 356° | nearby star on the move |
| Epsilon Indi B | 4.4 ± 0.6″/yr, 125°; methane sign | 4.7″/yr, 122° | cold brown dwarf |
| WISE 0855−0714 | 7.6″/yr, 286° | 8.2″/yr, 275° | cold brown dwarf |
| RW Aurigae | not moving; 78% brightness range | known variable | variable star |
| R Draconis | not moving; 53% brightness range | known variable (Mira type) | variable star |

All five verdicts match the known nature of the objects.

## Honest limitations

- **Signs, not proof.** A dip inside a molecule band is evidence, not a detection. Edges of bright stars and fast movers can create false spots, and the site says so on screen.
- **Pixel size.** SPHEREx pixels are 6.15″, so close pairs (Epsilon Indi Ba/Bb) appear as one dot. Very bright stars saturate, and their spectra are not measured.
- **Fast asteroids.** A main-belt asteroid crosses a 0.1° field in well under a day, so it appears as a single transient rather than a track. Only slow, distant objects give tracks.
- **"New to you" is not "new to science."** An unnamed faint dot that does not move is reported as a *candidate*, not a discovery.

## Reliability

- **Three paths to the data:** IRSA → our AWS coverage index → the server's disk cache. When IRSA was down for over a day during the hackathon, the site kept working through the index.
- The demo and famous fields are prepared in advance, so the first click is fast.
- Automated tests run on every change (API validation, reports, click-selection maths, behaviour when an outside service is down). A health endpoint reports archive state, disk space and the WISE service.

## Built with

Python (FastAPI, Astropy, Astroquery, NumPy, SciPy, Reproject, Photutils) on the server. A single-page HTML/JavaScript canvas app in the browser. SQLite for field reports.

## Repository map

| Path | What it is |
|---|---|
| `web/index.html` | the whole site: viewer, levels, cases, lenses, object card, reports |
| `main.py` | web server and API (`/api/frames`, `/api/object`, `/api/track`, `/api/lens`, `/api/movers`) |
| `data.py`, `cloud.py`, `s3index.py` | finding SPHEREx exposures and reading them from AWS |
| `epochs.py` | bad-pixel masks, background, epochs, common grid |
| `identify.py`, `spectrum.py`, `lens.py`, `movers.py` | motion, catalogue match, spectra, molecule maps, moving-object search |
| `wise.py` | WISE / NEOWISE frames |
| `reports.py` | field reports |
| `API.md` | API reference |
| `tests/` | automated tests |

## Acknowledgements

This project uses data from **SPHEREx**, a NASA mission, served by **NASA/IPAC IRSA** and the NASA Open Data on AWS programme. It also uses data from **WISE/NEOWISE**, the unTimely catalogue and the **Backyard Worlds: Planet 9** project's WiseView service, from ESA **Gaia DR3**, and from the **SIMBAD** database (CDS, Strasbourg, with the Harvard mirror).
