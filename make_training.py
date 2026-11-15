# AI disclosure: parts of this file were written with AI assistance (Claude, Anthropic), then reviewed and tested by the LUMEN team. See README, "AI use".
"""Реальные кадры для учебных уровней 1–5 -> web/levels/ (PNG + training.json).

    python make_training.py [--api http://localhost:8077]
    python make_training.py --only5        # только уровень 5: файлы уровней 1–4 не трогаются

Уровень 1  WISE 0855 (самый холодный коричневый карлик): два кадра WISE, 2010 и 2024 — прыжок ~41 пикселя.
Уровень 2  Epsilon Indi B: четыре кадра WISE 2010–2024 + один «космический луч», нарисованный нарочно
           (в подписи уровня сказано, что сбой добавлен).
Уровень 3  Luhman 16: четыре кадра WISE 2010–2024, 2.79″ в год.
Уровень 4  CRBR 2422.8-3423 (протозвезда во льду): кадры SPHEREx D5 и спектр из /api/object —
           линзы CO₂ и CO загораются по настоящим числам.
Уровень 5  Gliese 570 D (коричневый карлик T-типа, 19 св. лет): пять кадров WISE 2010–2024, без подсказок.
           Объект известный (Burgasser et al. 2000) — ученик находит его сам и сверяет с каталогом.
           Скорость меряется по всем эпохам WISE и кладётся в training.json рядом с каталожной (SIMBAD).

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
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
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


def stars_stack(imgs, avoid, keep_off=4.0, nsig=4.0):
    """Неподвижные точки по среднему из всех кадров уровня: по одной на пятно.

    Среднее гасит шум, поэтому находятся и слабые звёзды (на них тоже нажимают), а пересвеченная звезда
    даёт одну точку — середину пятна, а не по точке на каждый белый пиксель. Вытянутые пятна (блик от
    яркой звезды за краем кадра) звёздами не считаются.
    """
    from scipy.ndimage import find_objects, gaussian_filter, label
    a = gaussian_filter(np.mean([i.astype(float) for i in imgs], axis=0), 0.7)
    med = np.median(a); sig = 1.4826 * np.median(np.abs(a - med)) or 1.0
    lab, n = label(a - med > nsig * sig)
    out = []
    for k, sl in enumerate(find_objects(lab), 1):
        yy, xx = np.nonzero(lab == k)
        h, wd = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if len(yy) < 3 or len(yy) > 60 or max(h, wd) > 2.2 * min(h, wd) + 1 or len(yy) < 0.45 * h * wd:
            continue
        wt = a[yy, xx] - med
        x, y = float((xx * wt).sum() / wt.sum()) + 0.5, float((yy * wt).sum() / wt.sum()) + 0.5
        if all(math.hypot(x - p["x"], y - p["y"]) > keep_off for p in avoid):
            out.append({"x": round(x, 2), "y": round(y, 2)})
    return out


def wise_level(n, w, path, idx, N, labels, sim, glitch=None, shift=(0, 0)):
    """Кадры idx, вырезанные квадратом N вокруг середины пути объекта (shift — сдвиг выреза в пикселях)."""
    pts = [path[i] for i in idx]
    cx = np.mean([p[0] for p in pts]) + shift[0]; cy = np.mean([p[1] for p in pts]) + shift[1]
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
    stars = stars_stack(imgs, positions) if n == 5 else stars_in(imgs[-1], positions)
    return {"N": N, "pixel_arcsec": w["pixel_arcsec"], "epochs": labels, "frames": files,
            "target": {"positions": positions}, "stars": stars, "extras": extras, "sim": sim,
            "origin": [c0, r0]}


# Gliese 570 D по SIMBAD (GJ 570 D): положение J2000, собственное движение (mas/год), параллакс (mas)
GL570D = dict(ra=224.312344, dec=-21.363264, pmra=1038.08, pmdec=-1677.69, plx=169.3)


def level5():
    """Уровень 5: Gliese 570 D среди звёзд фона, пять февральских кадров WISE 2010–2024.

    Каталог нужен только чтобы навести вырезку и найти пятно на последнем кадре; дальше путь ведёт
    wise.track по самим кадрам, и скорость — прямая по центрам пятна во всех эпохах.
    """
    g = GL570D
    c = SkyCoord(g["ra"] * u.deg, g["dec"] * u.deg, pm_ra_cosdec=g["pmra"] * u.mas / u.yr,
                 pm_dec=g["pmdec"] * u.mas / u.yr, distance=(1000 / g["plx"]) * u.pc, obstime=Time("J2000"))
    mid = c.apply_space_motion(new_obstime=Time("2017-06-01"))
    w = get_wise(round(float(mid.ra.deg), 5), round(float(mid.dec.deg), 5), 0.1)
    fr = w["frames"]
    x1, y1 = wise_px(w, c.apply_space_motion(new_obstime=Time(fr[-1]["date"])))
    path = track(fr, x1, y1, start=len(fr) - 1)

    t = np.array([f["mjd"] for f in fr]) / 365.25
    (vx, _), cvx = np.polyfit(t, [p[0] for p in path], 1, cov=True)
    (vy, _), cvy = np.polyfit(t, [p[1] for p in path], 1, cov=True)
    k = w["pixel_arcsec"]
    v_e, v_n = -vx * k, -vy * k                       # x растёт к западу, y — вниз (к югу)
    speed = math.hypot(v_e, v_n)
    err = math.sqrt((v_e ** 2 * cvx[0, 0] + v_n ** 2 * cvy[0, 0])) * k / speed
    pa = math.degrees(math.atan2(v_e, v_n)) % 360
    cat = math.hypot(g["pmra"], g["pmdec"]) / 1000
    cat_pa = math.degrees(math.atan2(g["pmra"], g["pmdec"])) % 360
    span = math.hypot(path[-1][0] - path[0][0], path[-1][1] - path[0][1])
    print(f"Gliese 570 D: measured {speed:.3f} ± {err:.3f}\"/yr at PA {pa:.0f}° over {len(fr)} WISE epochs "
          f"{fr[0]['date']}..{fr[-1]['date']} ({span:.1f} px); catalogue (SIMBAD) {cat:.3f}\"/yr at PA {cat_pa:.0f}°")

    idx = pick(fr, [2010.12, 2014.12, 2017.12, 2021.12, 2024.12])     # все — февральские проходы
    N = 56
    d = wise_level(5, w, path, idx, N, [fr[i]["date"][:4] for i in idx],
                   "Real WISE/NEOWISE frames · 2010–2024 · the pale streak is glare from a bright star outside the frame",
                   shift=(-7, 2))
    c0, r0 = d.pop("origin")
    sky = w["wcs"].pixel_to_world(c0 + N / 2 - 0.5, w["shape"][0] - 2 - (r0 + N / 2 - 0.5))   # обратное к wise_px
    d["obj"] = {"name": "Gliese 570 D", "catalog": "SIMBAD · GJ 570 D · T-type brown dwarf",
                "ra": round(float(sky.ra.deg), 4), "dec": round(float(sky.dec.deg), 4),
                "motion": {"arcsec_per_yr": round(speed, 2), "arcsec_per_yr_err": round(err, 2), "pa_deg": round(pa),
                           "catalog_arcsec_per_yr": round(cat, 2), "significant": True},
                "shift_px": round(math.hypot(path[idx[-1]][0] - path[idx[0]][0], path[idx[-1]][1] - path[idx[0]][1])),
                "years": [fr[idx[0]]["date"][:4], fr[idx[-1]]["date"][:4]]}
    return d


def report(data):
    for n, d in data.items():
        p = d["target"]["positions"]
        print(f"level {n}: N={d['N']} frames={len(d['frames'])} stars={len(d['stars'])} "
              f"target {p[0]} -> {p[-1]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8099")
    ap.add_argument("--only5", action="store_true", help="rebuild level 5 only; levels 1–4 stay as they are")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.only5:
        data = json.loads((OUT / "training.json").read_text(encoding="utf-8"))
        data["5"] = level5()
        (OUT / "training.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        return report(data)
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

    data[5] = level5()
    for d in data.values():
        d.pop("origin", None)

    (OUT / "training.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    report(data)


if __name__ == "__main__":
    main()
