"""Кандидаты в движущиеся объекты (астероиды, кометы) по сеансовым кадрам SPHEREx.

    found = find_movers(imgs, mjds)      # imgs — кадры ОДНОГО детектора на общей сетке, по времени

Режим «каждый сеанс» (epochs.split_cadence, cadence="visit") даёт кадры с разницей в дни.
Неподвижные звёзды есть на всех кадрах, поэтому из каждого кадра вычитается медиана остальных:
остаётся то, что есть только сейчас. Такое пятно — «вспышка» (transient). Если вспышки на
соседних по времени кадрах ложатся на прямую с разумной скоростью, это трек (track) — уверенный
кандидат в медленно ползущий объект (далёкий астероид, комета).

Честное ограничение: астероид главного пояса проходит ~0,25° в сутки — больше всего поля (0,1°),
так что на соседнем сеансе его уже нет. Такие объекты видны как одиночные вспышки; треки дают
только медленные (дальние) объекты. Одиночные вспышки бывают и артефактами (космическая частица
в кадре из 2 снимков, остаток яркой звезды), поэтому отбор строгий: форма звезды, а не пиксель,
далеко от ярких/пересвеченных звёзд, значимость ≥ TRANSIENT_SIGMA.
"""
from __future__ import annotations

import warnings

import numpy as np

from epochs import _flux_scale

DETECT_SIGMA = 6.0       # порог пика на кадре-разности (в σ шума этой разности)
TRANSIENT_SIGMA = 8.0    # одиночная вспышка без трека — только если ярче
MIN_NEIGHBOUR = 0.25     # второй по яркости сосед ≥ 25% пика: звёздное пятно, а не один пиксель
RESID_FRAC = 0.6         # вспышка слабее 60% звезды под ней на опорном кадре = остаток звезды
                         # (её яркость от сеанса к сеансу гуляет с длиной волны), а не новый объект
NAN_GUARD_PX = 3         # и у края покрытия кадра / дыр от пересвета
MAX_PER_FRAME = 3        # кадр с большим числом вспышек — испорченный (сдвиг, артефакт), выкидываем
MIN_TRACK_PX = 2.0       # трек короче — дрожание центроида на месте, а не движение
EDGE_PX = 3
MAX_SPEED_PX_DAY = 6.0   # трек: не быстрее ~37″/сут (иначе объект за сеанс уходит из поля)
MAX_GAP_DAYS = 6.0       # трек связывает кадры не дальше этого по времени
LINE_TOL_PX = 1.5        # отклонение третьей точки от прямой
MAX_CANDIDATES = 6
MIN_EXPOSURES = 3        # в кадре из 2 снимков медиана = среднее, космическая частица выживает
PEAK_FRAC_SLACK = 0.12   # доля света в пиковом пикселе не больше, чем у звёзд поля, + запас
PEAK_FRAC_DEFAULT = 0.40 # если звёзд для сравнения мало
STATIC_PX = 1.5          # вспышки в одном месте на разных кадрах = переменный источник, не движение


def _robust_sigma(a: np.ndarray) -> float:
    v = a[np.isfinite(a)]
    if v.size < 50:
        return np.inf
    m = np.median(v)
    return float(1.4826 * np.median(np.abs(v - m))) or np.inf


def _peaks(diff: np.ndarray, sigma: float, guard: np.ndarray, ref: np.ndarray | None = None) -> list[dict]:
    """Локальные максимумы разности > DETECT_SIGMA·σ, со звёздной формой, вне защитной маски."""
    ny, nx = diff.shape
    d = np.nan_to_num(diff, nan=-np.inf)
    out = []
    for y in range(EDGE_PX, ny - EDGE_PX):
        for x in range(EDGE_PX, nx - EDGE_PX):
            v = d[y, x]
            if v < DETECT_SIGMA * sigma or guard[y, x]:
                continue
            win = d[y - 1:y + 2, x - 1:x + 2]
            if v < win.max():
                continue
            nb = np.sort(win.ravel())[-2]          # второй по яркости в окне 3×3
            if not np.isfinite(nb) or nb < MIN_NEIGHBOUR * v:
                continue                            # одиночный пиксель: космическая частица
            w = np.clip(np.nan_to_num(diff[y - 1:y + 2, x - 1:x + 2]), 0, None)
            yy, xx = np.mgrid[-1:2, -1:2]
            s = w.sum() or 1.0
            if ref is not None:
                under = np.clip(np.nan_to_num(ref[y - 1:y + 2, x - 1:x + 2]), 0, None).sum()
                if s < RESID_FRAC * under:
                    continue                        # на месте звезды и слабее её — остаток
            out.append(dict(x=x + float((w * xx).sum() / s), y=y + float((w * yy).sum() / s),
                            snr=float(v / sigma), peak_frac=float(v / s)))
    return out


def _star_peak_frac(ref: np.ndarray, med: float, sig: float) -> float:
    """Типичная доля света в пиковом пикселе 3×3 у звёзд поля (на медианном кадре): чем больше,
    тем резче пятно. Космические частицы и горячие пиксели резче звёзд."""
    ny, nx = ref.shape
    r = np.nan_to_num(ref - med, nan=-np.inf)
    fr = []
    for y in range(2, ny - 2):
        for x in range(2, nx - 2):
            v = r[y, x]
            if v < 10 * sig:
                continue
            win = r[y - 1:y + 2, x - 1:x + 2]
            if v < win.max() or not np.all(np.isfinite(win)):
                continue
            fr.append(v / np.clip(win, 0, None).sum())
    return float(np.median(fr)) if len(fr) >= 3 else PEAK_FRAC_DEFAULT - PEAK_FRAC_SLACK


def find_movers(imgs: list[np.ndarray], mjds: list[float], n_exposures: list[int] | None = None) -> list[dict]:
    """Кадры одного детектора (общая сетка, по времени) -> кандидаты.

    Каждый кандидат: {"kind": "track"|"transient", "frames": [индексы кадров], "x": [...], "y": [...]
    (пиксели сетки numpy), "snr": макс. значимость, "speed_px_day": скорость или None,
    "confidence": 0..1}. Отсортированы по уверенности, не больше MAX_CANDIDATES.
    n_exposures — сколько снимков в каждом кадре; вспышки ищутся только в кадрах из ≥ MIN_EXPOSURES.
    """
    n = len(imgs)
    if n < 3:
        return []
    stack = np.stack([np.asarray(im, np.float32) for im in imgs])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        full_ref = np.nanmedian(stack, axis=0)
    fin = full_ref[np.isfinite(full_ref)]
    if fin.size < 100:
        return []
    # защита: дыры (пересвет) и края покрытия; остатки звёзд отсекаются в _peaks по RESID_FRAC
    # (яркость звезды меняется от сеанса к сеансу вместе с длиной волны — на её месте в разности
    # остаётся пятно, слабее самой звезды; настоящая вспышка на пустом небе медианного кадра)
    from scipy.ndimage import binary_dilation
    med = np.median(fin)
    sig_ref = _robust_sigma(full_ref)
    guard = binary_dilation(~np.isfinite(full_ref), iterations=NAN_GUARD_PX)
    max_pf = _star_peak_frac(full_ref, med, sig_ref) + PEAK_FRAC_SLACK

    dets = []   # (кадр, пик)
    for i in range(n):
        if n_exposures is not None and n_exposures[i] < MIN_EXPOSURES:
            continue
        others = np.delete(stack, i, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ref = np.nanmedian(others, axis=0)
        diff = stack[i] - _flux_scale(stack[i], ref) * ref
        sig = _robust_sigma(diff)
        if not np.isfinite(sig):
            continue
        found = _peaks(diff, sig, guard | binary_dilation(~np.isfinite(stack[i]), iterations=NAN_GUARD_PX), ref - med)
        found = [p for p in found if p["peak_frac"] <= max_pf]   # резче звезды — частица/горячий пиксель
        if len(found) > MAX_PER_FRAME:
            continue                     # испорченный кадр: настоящих вспышек столько не бывает
        dets.extend((i, p) for p in found)

    # вспышки в одном и том же месте на разных кадрах — переменный неподвижный источник
    static = set()
    for a in range(len(dets)):
        for b in range(a + 1, len(dets)):
            if dets[a][0] != dets[b][0] and np.hypot(dets[a][1]["x"] - dets[b][1]["x"],
                                                     dets[a][1]["y"] - dets[b][1]["y"]) <= STATIC_PX:
                static.update((a, b))
    dets = [d for k, d in enumerate(dets) if k not in static]

    # связывание: пары вспышек на разных кадрах, близких по времени, с разумной скоростью;
    # третья точка (если есть) должна лечь на ту же прямую
    used, cands = set(), []
    order = sorted(range(len(dets)), key=lambda k: -dets[k][1]["snr"])
    for a in order:
        if a in used:
            continue
        ia, pa = dets[a]
        best = None
        for b in range(len(dets)):
            ib, pb = dets[b]
            if b == a or b in used or ib == ia:
                continue
            dt = abs(mjds[ib] - mjds[ia])
            if dt <= 0 or dt > MAX_GAP_DAYS:
                continue
            dist = np.hypot(pb["x"] - pa["x"], pb["y"] - pa["y"])
            if dist < MIN_TRACK_PX or dist / dt > MAX_SPEED_PX_DAY:
                continue               # на месте — не движение (остаток), слишком быстро — не он
            if best is None or dist / dt < best[1]:
                best = (b, dist / dt)
        if best is not None:
            b = best[0]
            members = sorted([a, b], key=lambda k: mjds[dets[k][0]])
            (i1, p1), (i2, p2) = dets[members[0]], dets[members[1]]
            vx = (p2["x"] - p1["x"]) / (mjds[i2] - mjds[i1])
            vy = (p2["y"] - p1["y"]) / (mjds[i2] - mjds[i1])
            for c in range(len(dets)):     # продолжение трека
                ic, pc = dets[c]
                if c in members or c in used or ic in (dets[m][0] for m in members):
                    continue
                t = mjds[ic] - mjds[i1]
                if abs(t) > MAX_GAP_DAYS * 2:
                    continue
                if np.hypot(p1["x"] + vx * t - pc["x"], p1["y"] + vy * t - pc["y"]) <= LINE_TOL_PX:
                    members.append(c)
            members.sort(key=lambda k: mjds[dets[k][0]])
            used.update(members)
            snr = max(dets[m][1]["snr"] for m in members)
            cands.append(dict(kind="track", frames=[dets[m][0] for m in members],
                              x=[dets[m][1]["x"] for m in members], y=[dets[m][1]["y"] for m in members],
                              snr=snr, speed_px_day=float(np.hypot(vx, vy)),
                              confidence=float(min(1.0, 0.5 + 0.15 * (len(members) - 2) + 0.02 * (snr - DETECT_SIGMA)))))
        elif pa["snr"] >= TRANSIENT_SIGMA:
            used.add(a)
            cands.append(dict(kind="transient", frames=[ia], x=[pa["x"]], y=[pa["y"]], snr=pa["snr"],
                              speed_px_day=None,
                              confidence=float(min(0.45, 0.2 + 0.02 * (pa["snr"] - TRANSIENT_SIGMA)))))
    cands.sort(key=lambda c: -c["confidence"])
    return cands[:MAX_CANDIDATES]
