"""Проверка конвейера данных: поиск -> вырезки (в кэш) -> кадры -> картинки и GIF.

    python check_data.py                       # WISE 0855
    python check_data.py --target barnard      # любой объект из targets.py
    python check_data.py --ra 269.45 --dec 4.74 --name mysky --size 0.1

Результат: out/<name>/frames.png (сетка детектор × эпоха), out/<name>/blink_D<n>.gif,
out/<name>/frames.npz (кадры для других модулей без сети).
"""
import argparse
import time
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord

from data import find_images, load_cutouts
from epochs import build_frames, frames_by_detector, select_exposures, diff_frames
from targets import TARGETS, coord_at, DEFAULT_WHEN


def save_diff(frames, out: Path, name: str):
    """Сетка разностей «эпоха минус первая эпоха»: красное — стало ярче, синее — темнее."""
    import matplotlib.pyplot as plt
    by_det = frames_by_detector(diff_frames(frames))
    if not by_det:
        return
    dets = sorted(by_det)
    ncol = max(len(v) for v in by_det.values())
    fig, axes = plt.subplots(len(dets), ncol, figsize=(2.6 * ncol, 2.6 * len(dets)), squeeze=False)
    for i, d in enumerate(dets):
        lim = np.nanpercentile(np.abs(np.stack([im for _, im in by_det[d]])), 99.5)
        for j in range(ncol):
            ax = axes[i, j]; ax.set_xticks([]); ax.set_yticks([])
            if j < len(by_det[d]):
                date, im = by_det[d][j]
                ax.imshow(im, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
                ax.set_title(f"D{d}  {date} − первая", fontsize=8)
    fig.suptitle(f"{name}: что изменилось (красное — ярче, синее — темнее)")
    fig.tight_layout()
    fig.savefig(out / "diff.png", dpi=110)
    plt.close(fig)


def save_outputs(frames, out: Path, name: str, info=None):
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
    plt.close(fig)
    save_diff(frames, out, name)
    if info:
        import json
        (out / "frames_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(out / "frames.npz",
                        dates=np.array([f[0] for f in frames]),
                        detectors=np.array([f[1] for f in frames]),
                        images=np.stack([f[2] for f in frames]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ra", type=float)
    ap.add_argument("--dec", type=float)
    ap.add_argument("--target", choices=sorted(TARGETS), help="объект из targets.py")
    ap.add_argument("--name", default=None)
    ap.add_argument("--size", type=float, default=0.1)
    ap.add_argument("--per-group", type=int, default=None,
                    help="не больше N снимков на (эпоху, детектор) — быстрый режим")
    ap.add_argument("--when", default=None, help="дата для сдвига на собственное движение")
    a = ap.parse_args()

    if a.ra is not None:
        coord = SkyCoord(a.ra, a.dec, unit="deg")
        a.name = a.name or "custom"
    else:
        a.target = a.target or "wise0855"
        coord = coord_at(a.target, a.when)
        a.name = a.name or a.target
    print(f"{a.name}: RA={coord.ra.deg:.5f} Dec={coord.dec.deg:.5f} size={a.size}°")

    t0 = time.time()
    table = find_images(coord.ra.deg, coord.dec.deg, a.size)
    print(f"снимков: {len(table)} ({time.time() - t0:.0f} c); релизы: {sorted(set(map(str, table['release'])))}")
    if a.per_group:
        table = select_exposures(table, a.per_group)
        print(f"быстрый режим: оставлено {len(table)} (по {a.per_group} на эпоху и детектор)")

    t0 = time.time()
    cutouts = load_cutouts(table)
    print(f"загружено {len(cutouts)} вырезок ({time.time() - t0:.0f} c)")

    info = []
    frames = build_frames(cutouts, coord, a.size, info=info)
    for (date, det, im), inf in zip(frames, info):
        print(f"  D{det} {date}: {inf['n_exposures']:3d} эксп., λ={inf['wavelength_um']:.2f} мкм, "
              f"валидных пикселей {np.isfinite(im).mean():.0%}")
    save_outputs(frames, Path("out") / a.name, a.name, info)
    print(f"готово: out/{a.name}/")


if __name__ == "__main__":
    main()
