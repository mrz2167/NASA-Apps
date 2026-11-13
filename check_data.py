"""Проверка конвейера данных: поиск -> вырезки (в кэш) -> кадры -> картинки и GIF.

    python check_data.py                       # WISE 0855
    python check_data.py --ra 269.45 --dec 4.74 --name barnard --size 0.1

Результат: out/<name>/frames.png (сетка детектор × эпоха), out/<name>/blink_D<n>.gif,
out/<name>/frames.npz (кадры для других модулей без сети).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from data import find_images, load_cutouts
from epochs import build_frames, frames_by_detector

# WISE J085510.83-071442.5: позиция из обозначения (эпоха WISE ~2010.5), собственное движение
# из литературы (Luhman & Esplin 2016). Точные «координаты на сегодня» — задача identify.now_coord.
W0855 = SkyCoord("08h55m10.83s -07d14m42.5s", pm_ra_cosdec=-8118 * u.mas / u.yr,
                 pm_dec=680 * u.mas / u.yr, obstime=Time(2010.5, format="jyear"), distance=2.28 * u.pc)


def save_outputs(frames, out: Path, name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    out.mkdir(parents=True, exist_ok=True)
    by_det = frames_by_detector(frames)
    dets = sorted(by_det)
    ncol = max(len(v) for v in by_det.values())
    fig, axes = plt.subplots(len(dets), ncol, figsize=(2.6 * ncol, 2.6 * len(dets)), squeeze=False)
    for i, d in enumerate(dets):
        imgs = [im for _, im in by_det[d]]
        lo, hi = np.nanpercentile(np.stack(imgs), [5, 99.5])
        for j in range(ncol):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            if j < len(by_det[d]):
                date, im = by_det[d][j]
                ax.imshow(im, origin="lower", cmap="gray", vmin=lo, vmax=hi)
                ax.plot(im.shape[1] / 2 - 0.5, im.shape[0] / 2 - 0.5, "+", color="red", ms=8)
                ax.set_title(f"D{d}  {date}", fontsize=9)
        # GIF мигания для детектора
        gif = []
        for date, im in by_det[d]:
            norm = np.clip((np.nan_to_num(im, nan=lo) - lo) / (hi - lo), 0, 1)
            big = np.kron((norm * 255).astype(np.uint8), np.ones((6, 6), np.uint8))
            gif.append(big)
        imageio.mimsave(out / f"blink_D{d}.gif", gif, duration=0.7, loop=0)
    fig.suptitle(f"{name}: кадры по детекторам и эпохам (+ = центр запроса)")
    fig.tight_layout()
    fig.savefig(out / "frames.png", dpi=110)
    np.savez_compressed(out / "frames.npz",
                        dates=np.array([f[0] for f in frames]),
                        detectors=np.array([f[1] for f in frames]),
                        images=np.stack([f[2] for f in frames]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ra", type=float)
    ap.add_argument("--dec", type=float)
    ap.add_argument("--name", default="wise0855")
    ap.add_argument("--size", type=float, default=0.1)
    ap.add_argument("--when", default="2026-01-01", help="дата для сдвига WISE 0855 на собственное движение")
    a = ap.parse_args()

    if a.ra is None:
        coord = W0855.apply_space_motion(new_obstime=Time(a.when))
    else:
        coord = SkyCoord(a.ra, a.dec, unit="deg")
    print(f"{a.name}: RA={coord.ra.deg:.5f} Dec={coord.dec.deg:.5f} size={a.size}°")

    t0 = time.time()
    table = find_images(coord.ra.deg, coord.dec.deg, a.size)
    print(f"снимков: {len(table)} ({time.time() - t0:.0f} c); релизы: {sorted(set(table['release']))}")

    t0 = time.time()
    cutouts = load_cutouts(table)
    print(f"загружено {len(cutouts)} вырезок ({time.time() - t0:.0f} c)")

    frames = build_frames(cutouts, coord, a.size)
    for date, det, im in frames:
        print(f"  D{det} {date}: {im.shape}, валидных пикселей {np.isfinite(im).mean():.0%}")
    save_outputs(frames, Path("out") / a.name, a.name)
    print(f"готово: out/{a.name}/")


if __name__ == "__main__":
    main()
