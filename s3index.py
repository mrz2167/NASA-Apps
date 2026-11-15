# AI disclosure: parts of this file were written with AI assistance (Claude, Anthropic), then reviewed and tested by the LUMEN team. See README, "AI use".
"""Свой список «какой снимок SPHEREx накрывает какую точку неба» — запасной путь, когда IRSA молчит.

IRSA (TAP) — единственное место, где можно спросить «какие экспозиции накрывают RA/Dec». Сами снимки
лежат ещё и в открытом бакете AWS nasa-irsa-spherex, но списка покрытия там нет. Этот модуль строит
его сам: читает из каждого файла кусок заголовка (11 КБ из 67 МБ) с привязкой к небу и временем.

    python s3index.py build [--release dr1 qr3] [--threads 96]   # часы; можно прервать и продолжить
    python s3index.py merge                                       # части -> cache/index/spherex_index.npz
    python s3index.py check                                       # сверка с закэшированными ответами IRSA

    find(ra, dec, size_deg) -> Table как у data.find_images, или None, если списка нет.

Детекторы смотрят на небо парами через дихроик (D1+D4, D2+D5, D3+D6; сдвиг ~12 пикселей), поэтому
заголовки читаются только у D1–D3, а D4–D6 берут привязку пары. Точка считается накрытой, если она
не ближе EDGE_PX к краю детектора, — запас больше сдвига пары и половины вырезки.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent
INDEX_DIR = Path(os.environ.get("SKYWATCH_CACHE") or ROOT / "cache") / "index"
PARTS = INDEX_DIR / "parts"
MERGED = INDEX_DIR / "spherex_index.npz"
BUCKET = "https://nasa-irsa-spherex.s3.us-east-1.amazonaws.com/"
IRSA_IBE = "https://irsa.ipac.caltech.edu/ibe/data/spherex/"
NPIX = 2040
EDGE_PX = 15                       # 97% снимков из ответа IRSA; при 45 было 93.5%, при 0 появляются лишние
RANGE = (11520, 23039)             # здесь лежат PC/CRVAL/MJD-BEG заголовка IMAGE (проверено на D1–D6)
RANGE_WIDE = (0, 31679)
NAME = re.compile(r"level2_(\d{4}W\d{2}_\d[A-Z])_(\d+)_(\d+)D(\d)_spx_(.+)\.fits$")
WL = {1: (0.7455, 1.1167), 2: (1.1033, 1.6499), 3: (1.6301, 2.4346),
      4: (2.4027, 3.8473), 5: (3.8113, 4.43), 6: (4.4115, 5.0096)}
RELEASE_PRIORITY = {"dr1": 0, "qr3": 1, "qr2": 2}
KEYS = ("CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "PC1_1", "PC1_2", "PC2_1", "PC2_2", "CDELT1", "CDELT2", "MJD-BEG")

_local = threading.local()


def _session() -> requests.Session:
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=2, pool_maxsize=2))
        _local.s = s
    return s


def _list(prefix: str, delimiter: str = "") -> list[str]:
    """Все ключи (или подпапки, если задан delimiter) под prefix."""
    out, token = [], None
    while True:
        url = f"{BUCKET}?list-type=2&prefix={quote(prefix)}" + (f"&delimiter={delimiter}" if delimiter else "")
        if token:
            url += "&continuation-token=" + quote(token, safe="")
        for attempt in range(6):
            try:
                r = _session().get(url, timeout=60)
                r.raise_for_status()
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2 + 3 * attempt)
        tag = "Prefix" if delimiter else "Key"
        found = re.findall(rf"<{tag}>([^<]+)</{tag}>", r.text)
        out += [f for f in found if f != prefix]
        m = re.search(r"<NextContinuationToken>([^<]+)<", r.text)
        if not m:
            return out
        token = m.group(1)


def _parse(text: str) -> list[float] | None:
    vals = []
    for k in KEYS:
        m = re.search(rf"{re.escape(k.ljust(8))}= *([-+0-9.Ee]+)", text)
        if not m:
            return None
        vals.append(float(m.group(1)))
    return vals


def _header(key: str) -> list[float] | None:
    for rng in (RANGE, RANGE_WIDE):
        for attempt in range(5):
            try:
                r = _session().get(BUCKET + key, headers={"Range": f"bytes={rng[0]}-{rng[1]}"}, timeout=40)
                if r.status_code in (200, 206):
                    vals = _parse(r.content.decode("latin1"))
                    if vals is not None:
                        return vals
                    break                                # ключей нет в этом куске: шире
                if r.status_code == 404:
                    return None
            except Exception:
                pass
            time.sleep(1 + 2 * attempt)
    return None


def build(releases: list[str], threads: int, only: str | None = None) -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    pool = ThreadPoolExecutor(threads)
    for rel in releases:
        weeks = [w.rstrip("/").split("/")[-1] for w in _list(f"{rel}/level2/", "/")]
        if only:
            weeks = [w for w in weeks if re.search(only, w)]
        print(f"{rel}: {len(weeks)} week folders", flush=True)
        for wi, week in enumerate(weeks):
            part = PARTS / f"{rel}_{week}.npz"
            if part.exists():
                continue
            t0 = time.time()
            keys = [k for k in _list(f"{rel}/level2/{week}/") if NAME.search(k)]
            have: dict[tuple, dict[int, str]] = {}       # (a, b, банк 1..3) -> {детектор: ключ}
            for k in keys:
                m = NAME.search(k)
                a, b, d = int(m.group(2)), int(m.group(3)), int(m.group(4))
                slot = have.setdefault((a, b, (d - 1) % 3 + 1), {})
                if d not in slot or "retry" in k:        # повторная обработка новее
                    slot[d] = k
            items = sorted(have.items())
            todo = [slot.get(bank) or slot[bank + 3] for (_, _, bank), slot in items]
            heads = list(pool.map(_header, todo))
            rows = [(a, b, bank, slot.get(bank, ""), slot.get(bank + 3, ""), h)
                    for ((a, b, bank), slot), h in zip(items, heads) if h is not None]
            failed = len(items) - len(rows)
            if failed > max(20, 0.02 * len(items)):
                print(f"  {week}: {failed}/{len(items)} headers failed, part NOT saved (rerun later)", flush=True)
                continue
            np.savez_compressed(
                part,
                a=np.array([r[0] for r in rows], np.int32), b=np.array([r[1] for r in rows], np.int16),
                bank=np.array([r[2] for r in rows], np.int8),
                key_lo=np.array([r[3] for r in rows], dtype="S"), key_hi=np.array([r[4] for r in rows], dtype="S"),
                geo=np.array([r[5] for r in rows], np.float64).reshape(-1, len(KEYS)))
            dt = time.time() - t0
            print(f"  [{wi + 1}/{len(weeks)}] {rel} {week}: {len(keys)} files, {len(rows)} pointings, "
                  f"{failed} failed, {dt:.0f} s ({len(rows) / max(dt, 1):.0f}/s)", flush=True)


def merge() -> None:
    """Части -> один компактный файл: строка на «экспозиция × пара детекторов»."""
    parts = sorted(PARTS.glob("*.npz"))
    if not parts:
        sys.exit("no parts yet")
    folders: list[str] = []                               # «dr1/level2/2025W20_1C/l2b-v27-2026-198»
    fi: dict[str, int] = {}
    cols: dict[str, list] = {k: [] for k in ("a", "b", "bank", "f_lo", "f_hi", "geo")}

    def folder_id(keys: np.ndarray) -> np.ndarray:
        out = np.full(len(keys), -1, np.int16)
        for i, k in enumerate(keys):
            if not k:
                continue
            f = k.decode().rsplit("/", 2)[0]
            if f not in fi:
                fi[f] = len(folders)
                folders.append(f)
            out[i] = fi[f]
        return out

    for p in parts:
        z = np.load(p)
        if len(z["a"]) == 0:
            continue
        cols["a"].append(z["a"]); cols["b"].append(z["b"]); cols["bank"].append(z["bank"])
        cols["f_lo"].append(folder_id(z["key_lo"])); cols["f_hi"].append(folder_id(z["key_hi"]))
        cols["geo"].append(z["geo"])
    geo = np.concatenate(cols["geo"])
    cd = geo[:, 4:8].reshape(-1, 2, 2) * geo[:, 8:10].reshape(-1, 2, 1)       # CDELT_i * PC_ij
    np.savez_compressed(
        MERGED, folders=np.array(folders),
        a=np.concatenate(cols["a"]), b=np.concatenate(cols["b"]), bank=np.concatenate(cols["bank"]),
        f_lo=np.concatenate(cols["f_lo"]), f_hi=np.concatenate(cols["f_hi"]),
        crval=geo[:, 0:2], crpix=geo[:, 2:4].astype(np.float32), cd=cd, mjd=geo[:, 10])
    print(f"{MERGED.name}: {len(geo)} pointings from {len(parts)} week folders, "
          f"{MERGED.stat().st_size / 1e6:.1f} MB")


_idx = None
_idx_mtime = 0.0
_idx_lock = threading.Lock()


def _index():
    """Список в памяти; перечитывается, если файл обновился (доложили новые недели)."""
    global _idx, _idx_mtime
    if not MERGED.exists():
        return None
    mt = MERGED.stat().st_mtime
    with _idx_lock:
        if _idx is None or mt != _idx_mtime:
            z = np.load(MERGED)
            d = {k: z[k] for k in z.files}
            ra, dec = np.radians(d["crval"][:, 0]), np.radians(d["crval"][:, 1])
            d["unit"] = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], 1).astype(np.float32)
            _idx, _idx_mtime = d, mt
    return _idx


def available() -> bool:
    return MERGED.exists()


def covering(ra: float, dec: float, edge_px: float = EDGE_PX):
    """Индексы строк списка, чей детектор накрывает точку, и её пиксель на каждом."""
    d = _index()
    if d is None:
        return None, None
    a, dl = np.radians(ra), np.radians(dec)
    v = np.array([np.cos(dl) * np.cos(a), np.cos(dl) * np.sin(a), np.sin(dl)], np.float32)
    near = np.nonzero(d["unit"] @ v > np.cos(np.radians(2.7)))[0]           # полудиагональ детектора 2.5°
    if len(near) == 0:
        return near, np.zeros((0, 2))
    a0, d0 = np.radians(d["crval"][near, 0]), np.radians(d["crval"][near, 1])
    cosc = np.sin(d0) * np.sin(dl) + np.cos(d0) * np.cos(dl) * np.cos(a - a0)
    xi = np.degrees(np.cos(dl) * np.sin(a - a0) / cosc)                      # гномоническая проекция
    eta = np.degrees((np.cos(d0) * np.sin(dl) - np.sin(d0) * np.cos(dl) * np.cos(a - a0)) / cosc)
    pix = np.linalg.solve(d["cd"][near], np.stack([xi, eta], 1)[..., None])[..., 0] + d["crpix"][near]
    ok = ((pix > edge_px) & (pix < NPIX - edge_px)).all(1)
    return near[ok], pix[ok]


def find(ra: float, dec: float, size_deg: float = 0.1):
    """Table со столбцами data.find_images (url — в виде IRSA, его понимает cloud.cutout_hdul) или None."""
    from astropy.table import Table
    rows_i, _ = covering(ra, dec)
    if rows_i is None:
        return None
    d = _index()
    best: dict[tuple, tuple] = {}
    for i in rows_i:
        for f_col, det in (("f_lo", int(d["bank"][i])), ("f_hi", int(d["bank"][i]) + 3)):
            f = int(d[f_col][i])
            if f < 0:
                continue
            folder = str(d["folders"][f])                                  # dr1/level2/<week>/<версия>
            rel, _, week, ver = folder.split("/")
            obs = f"{week}_{int(d['a'][i]):04d}_{int(d['b'][i])}D{det}"
            key = f"{folder}/{det}/level2_{obs}_spx_{ver}.fits"
            row = (f"{IRSA_IBE}{key}?center={ra},{dec}d&size={size_deg}", obs, rel,
                   round(float(d["mjd"][i]), 6), det, WL[det][0], WL[det][1])
            old = best.get((obs,))
            if old is None or RELEASE_PRIORITY.get(rel, 9) < RELEASE_PRIORITY.get(old[2], 9):
                best[(obs,)] = row
    names = ["url", "obs_id", "release", "time", "detector", "wl_min_um", "wl_max_um"]
    t = Table(rows=sorted(best.values(), key=lambda r: r[3]), names=names) if best else Table(names=names)
    t.meta.update(ra=ra, dec=dec, size_deg=size_deg, source="aws-index")
    return t


def check() -> None:
    """Сверка с ответами IRSA, сохранёнными в cache/queries: сколько экспозиций список находит."""
    from astropy.table import Table
    d = _index()
    if d is None:
        sys.exit("no index")
    weeks = {str(f).split("/")[2] for f in d["folders"]}
    for q in sorted((ROOT / "cache" / "queries").glob("*.ecsv")):
        t = Table.read(q)
        ra, dec = t.meta["ra"], t.meta["dec"]
        truth = {str(o) for o in t["obs_id"] if str(o)[:10] in weeks}
        mine = find(ra, dec)
        got = {str(o) for o in mine["obs_id"]}
        print(f"{q.stem} ra={ra:9.4f} dec={dec:+8.4f}: IRSA {len(truth):4d} (in indexed weeks), index {len(got):4d}, "
              f"common {len(truth & got):4d}, missed {len(truth - got):3d}, extra {len(got - truth):3d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "merge", "check"])
    ap.add_argument("--release", nargs="+", default=["dr1", "qr3"])
    ap.add_argument("--threads", type=int, default=96)
    ap.add_argument("--only", help="regex on week folder names")
    a = ap.parse_args()
    {"build": lambda: build(a.release, a.threads, a.only), "merge": merge, "check": check}[a.cmd]()
