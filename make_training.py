"""Реальные кадры для учебных уровней 1–4 -> web/levels/ (PNG + training.json).

    python make_training.py [--api http://localhost:8077]

Уровень 1  WISE 0855 (самый холодный коричневый карлик): два кадра WISE, 2010 и 2024 — прыжок ~41 пикселя.
Уровень 2  Epsilon Indi B: четыре кадра WISE 2010–2024 + один «космический луч», нарисованный нарочно
           (в подписи уровня сказано, что сбой добавлен).
Уровень 3  Luhman 16: четыре кадра WISE 2010–2024, 2.79″ в год.
Уровень 4  CRBR 2422.8-3423 (протозвезда во льду): кадры SPHEREx D5 и спектр из /api/object —
           линзы CO₂ и CO загораются по настоящим числам.
Уровень 5 остаётся симуляцией: настоящего «неизвестного кандидата» у нас нет.

Кадры WISE берутся из wise.py (кэш cache/wise/), SPHEREx — с работающего сервера (--api).
Положения объектов — в пикселях кадра (x вправо, y вниз, 0 — левый верхний угол), как ждёт сайт.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
from pathlib import Path

import numpy as np
import requests
from astropy.coordinates import SkyCoord
from PIL import Image

from targets import coord_at
from wise import centroid, get_wise, track

OUT = Path(__file__).resolve().parent / "web" / "levels"


def wise_px(w, c):
    x, y = (float(v) for v in w["wcs"].world_to_pixel(c))
    return x, w["shape"][0] - 2 - y          # см. main._wise_px


def known_path(w, name):
    out = []
    for f in w["frames"]:
        px, py = wise_px(w, coord_at(name, f["date"]))
        out.append(centroid(f["image"], px, py, search=2) or (px, py))
    return out


def pick(frames, years):
    """Индексы кадров, ближайших к заданным годам."""
    yr = np.array([int(f["date"][:4]) + (int(f["date"][5:7]) - 0.5) / 12 for f in frames])
    return [int(np.argmin(np.abs(yr - y))) for y in years]


def stars_in(img, avoid, n=30, keep_off=4.0):
    from photutils.detection import find_peaks
    a = img.astype(float)
    med = np.median(a); sig = 1.4826 * np.median(np.abs(a - med)) or 1.0
    t = find_peaks(a - med, threshold=6 * sig, box_size=5, npeaks=n, border_width=1)
    out = []
    if t is None:
        return out
    for r in t:
        x, y = float(r["x_peak"]) + 0.5, float(r["y_peak"]) + 0.5
        if all(math.hypot(x - p["x"], y - p["y"]) > keep_off for p in avoid):
            out.append({"x": round(x, 2), "y": round(y, 2)})
    return out


def wise_level(n, w, path, idx, N, labels, sim, glitch=None):
    """Кадры idx, вырезанные квадратом N вокруг середины пути объекта."""
    pts = [path[i] for i in idx]
    cx = np.mean([p[0] for p in pts]); cy = np.mean([p[1] for p in pts])
    ny, nx = w["shape"]
    c0 = int(round(min(max(cx - N / 2, 0), nx - N))); r0 = int(round(min(max(cy - N / 2, 0), ny - N)))
    positions = [{"x": round(p[0] - c0 + 0.5, 2), "y": round(p[1] - r0 + 0.5, 2)} for p in pts]
    files = []
    imgs = []
    for k, i in enumerate(idx):
        img = w["frames"][i]["image"][r0:r0 + N, c0:c0 + N].copy()
        imgs.append(img)
    extras = []
    if glitch is not None:
        frame_k = glitch
        # пустое место: далеко от звёзд и пути объекта
        stars = stars_in(imgs[-1], positions, keep_off=0)
        best, bd = None, -1
        for gy in range(6, N - 6, 2):
            for gx in range(6, N - 6, 2):
                d = min([math.hypot(gx - s["x"], gy - s["y"]) for s in stars] +
                        [math.hypot(gx - p["x"], gy - p["y"]) * 0.8 for p in positions])
                if d > bd:
                    best, bd = (gx, gy), d
        gx, gy = best
        img = imgs[frame_k]
        img[gy, gx] = 255; img[gy, gx + 1] = 230; img[gy + 1, gx] = 200       # резкий «удар частицы»
        extras.append({"kind": "glitch", "positions": [{"x": gx + 1.0, "y": gy + 1.0}] * len(idx),
                       "frames": [frame_k], "moving": False})
    for k, img in enumerate(imgs):
        name = f"l{n}_{k}.png"
        Image.fromarray(img).save(OUT / name, optimize=True)
        files.append("levels/" + name)
    stars = stars_in(imgs[-1], positions)
    return {"N": N, "pixel_arcsec": w["pixel_arcsec"], "epochs": labels, "frames": files,
            "target": {"positions": positions}, "stars": stars, "extras": extras, "sim": sim}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8099")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = {}

    # 1. WISE 0855: 2010 и 2024
    c = coord_at("wise0855")
    w = get_wise(c.ra.deg, c.dec.deg, 0.1)
    path = known_path(w, "wise0855")
    idx = [0, len(w["frames"]) - 1]
    yrs = [w["frames"][i]["date"][:4] for i in idx]
    data[1] = wise_level(1, w, path, idx, 72, [f"{yrs[0]} · WISE", f"{yrs[1]} · NEOWISE"],
                         "Real WISE/NEOWISE frames · WISE 0855, the coldest brown dwarf known")

    # 2. Epsilon Indi B + нарочно добавленный сбой
    c = coord_at("eps_ind_b")
    w = get_wise(c.ra.deg, c.dec.deg, 0.1)
    path = known_path(w, "eps_ind_b")
    idx = pick(w["frames"], [2010.3, 2015.3, 2019.8, 2024.3])
    data[2] = wise_level(2, w, path, idx, 56, [w["frames"][i]["date"][:4] for i in idx],
                         "Real WISE/NEOWISE frames · Epsilon Indi B · one glitch added on purpose (a fake cosmic-ray hit)",
                         glitch=2)

    # 3. Luhman 16 (нет в targets.py): положение на 2024.5 найдено по движению на кадрах WISE
    #    (кнопка «Near Luhman 16» на сайте нарочно смотрит рядом, а не на него); ведём от последнего кадра
    ra, dec = 162.2963, -53.31751
    w = get_wise(ra, dec, 0.1)
    cx, cy = wise_px(w, SkyCoord(ra, dec, unit="deg"))
    path = track(w["frames"], cx, cy, start=len(w["frames"]) - 1)
    idx = pick(w["frames"], [2010.3, 2014.3, 2019.3, 2024.3])
    shift = math.hypot(path[idx[-1]][0] - path[idx[0]][0], path[idx[-1]][1] - path[idx[0]][1]) * w["pixel_arcsec"]
    print(f"Luhman 16: {shift:.1f}\" between {w['frames'][idx[0]]['date']} and {w['frames'][idx[-1]]['date']} (catalogue ~2.79\"/yr)")
    labels = [f"{w['frames'][i]['date'][:4]} · {'WISE' if w['frames'][i]['date'] < '2011' else 'NEOWISE'}" for i in idx]
    data[3] = wise_level(3, w, path, idx, 48, labels,
                         "Real WISE/NEOWISE frames · Luhman 16 moving at its real speed (2.79″ a year)")

    # 4. CRBR 2422.8-3423: SPHEREx D5 + настоящий спектр
    ra, dec = 246.8525, -24.6842
    fr = requests.get(f"{a.api}/api/frames", params=dict(ra=ra, dec=dec, fov=0.1, detector=5), timeout=900).json()
    ob = requests.get(f"{a.api}/api/object", params=dict(ra=ra, dec=dec), timeout=900).json()
    files = []
    for k, f in enumerate(fr["frames"]):
        raw = base64.b64decode(f["png"].split(",", 1)[1])
        name = f"l4_{k}.png"
        Image.open(io.BytesIO(raw)).save(OUT / name, optimize=True)
        files.append("levels/" + name)
    pos = fr["target"]["positions"]
    data[4] = {"N": fr["size_px"], "pixel_arcsec": fr["pixel_arcsec"], "epochs": [f["label"] for f in fr["frames"]],
               "frames": files, "target": {"positions": pos, "mol": ob["molecules"]},
               "stars": [s for s in fr.get("sources", []) if math.hypot(s["x"] - pos[0]["x"], s["y"] - pos[0]["y"]) > 3],
               "extras": [], "obj": {"molecules": ob["molecules"], "spectrum": ob["spectrum"]},
               "sim": "Real NASA SPHEREx frames · detector 5 · the young star CRBR 2422.8-3423"}
    print("CRBR molecules:", ob["molecules"], "spectrum points:", len(ob["spectrum"]["wavelength"]))

    (OUT / "training.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    for n, d in data.items():
        p = d["target"]["positions"]
        print(f"level {n}: N={d['N']} frames={len(d['frames'])} stars={len(d['stars'])} "
              f"target {p[0]} -> {p[-1]}")


if __name__ == "__main__":
    main()
