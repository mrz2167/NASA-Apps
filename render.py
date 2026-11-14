"""Картинки и анимации для сайта — всё в памяти, без файлов.

    from render import blink_gif, frame_png, diff_png
    gif = blink_gif(sky, detector=3, zoom_arcsec=90)      # bytes -> st.image(gif)
    png = frame_png(sky, index=0)                         # один кадр
    png = diff_png(sky, index=0)                          # «что изменилось»

Яркость растягивается asinh (видны и слабые, и яркие звёзды) с ОДНИМИ пределами на все кадры
детектора — иначе мигание «дышит» само по себе. Пиксели увеличиваются без сглаживания:
что видно на экране — то и есть в данных.
"""
from __future__ import annotations

import io

import numpy as np
from astropy.coordinates import SkyCoord
from PIL import Image, ImageDraw, ImageFont

PIX_ARCSEC = 6.15


# ---------- яркость ----------

def fill_saturated(img: np.ndarray, max_hole: int = 400) -> np.ndarray:
    """ТОЛЬКО ДЛЯ ПОКАЗА: заливает «дыры» от пересвеченных ядер ярких звёзд.

    Пиксели ядра яркой звезды помечены флагами переполнения и выброшены (NaN), из-за чего
    звезда выглядит чёрной дыркой. Заливаем закрытые NaN-области (не касающиеся края кадра,
    не больше max_hole пикселей) максимумом соседей — звезда снова белая.
    Для науки (фотометрия, центроид) используйте исходные кадры, а не этот результат.
    """
    from scipy import ndimage
    bad = ~np.isfinite(img)
    if not bad.any():
        return img
    lab, n = ndimage.label(bad)
    if n == 0:
        return img
    out = img.copy()
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    sizes = ndimage.sum(bad, lab, index=np.arange(1, n + 1))
    fillable = np.zeros(n + 1, bool)
    for k in range(1, n + 1):
        fillable[k] = (k not in edge) and sizes[k - 1] <= max_hole
    hole = fillable[lab]
    if not hole.any():
        return img
    work = np.where(np.isfinite(img), img, -np.inf)
    for _ in range(30):  # растим максимум соседей внутрь дыры
        grown = ndimage.grey_dilation(work, size=3)
        upd = hole & ~np.isfinite(work)
        if not upd.any():
            break
        work[upd] = grown[upd]
    out[hole] = work[hole]
    out[hole & ~np.isfinite(out)] = np.nan
    return out


def stretch_limits(images, lo_pct: float = 30.0, hi_pct: float = 99.8) -> tuple[float, float]:
    """Общие пределы яркости для набора кадров."""
    vals = np.concatenate([im[np.isfinite(im)].ravel() for im in images]) if images else np.array([0.0])
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(vals, [lo_pct, hi_pct])
    return float(lo), float(hi if hi > lo else lo + 1)


def to_uint8(img: np.ndarray, lo: float, hi: float, mode: str = "asinh", soft: float = 0.05) -> np.ndarray:
    """2D -> 0..255. mode: 'asinh' (по умолчанию) или 'linear'. NaN -> тёмно-серый."""
    x = (np.nan_to_num(img, nan=lo) - lo) / (hi - lo)
    x = np.clip(x, 0, 1)
    if mode == "asinh":
        x = np.arcsinh(x / soft) / np.arcsinh(1 / soft)
    out = (x * 255).astype(np.uint8)
    out[~np.isfinite(img)] = 25
    return out


def diverging_rgb(img: np.ndarray, lim: float) -> np.ndarray:
    """Разность -> RGB: красное — стало ярче, синее — темнее, белое — без изменений."""
    from matplotlib import colormaps
    x = np.clip(np.nan_to_num(img) / lim, -1, 1) * 0.5 + 0.5
    rgb = (colormaps["RdBu_r"](x)[..., :3] * 255).astype(np.uint8)
    rgb[~np.isfinite(img)] = 40
    return rgb


# ---------- геометрия ----------

def zoom_box(shape, wcs, coord: SkyCoord | None, zoom_arcsec: float | None):
    """Срез (y0, y1, x0, x1) квадрата ±zoom_arcsec вокруг coord (или центра)."""
    ny, nx = shape
    if zoom_arcsec is None:
        return 0, ny, 0, nx
    if coord is not None and wcs is not None:
        cx, cy = (float(v) for v in wcs.world_to_pixel(coord))
    else:
        cx, cy = (nx - 1) / 2, (ny - 1) / 2
    r = max(3, int(round(zoom_arcsec / PIX_ARCSEC)))
    x0, y0 = int(round(cx)) - r, int(round(cy)) - r
    x0, y0 = max(0, min(x0, nx - 2 * r - 1)), max(0, min(y0, ny - 2 * r - 1))
    return y0, min(ny, y0 + 2 * r + 1), x0, min(nx, x0 + 2 * r + 1)


def _to_canvas(arr: np.ndarray, scale: int) -> Image.Image:
    """Массив (север вниз в numpy) -> картинка: север вверх, восток влево, крупные пиксели."""
    arr = np.flipud(arr)  # origin=lower -> экран
    img = Image.fromarray(arr)
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)


def _sky_to_canvas(coord: SkyCoord, wcs, box, scale: int, shape_h: int):
    """Небесная координата -> пиксель картинки после зума, переворота и увеличения."""
    y0, y1, x0, x1 = box
    x, y = (float(v) for v in wcs.world_to_pixel(coord))
    px = (x - x0 + 0.5) * scale
    py = ((y1 - y0) - (y - y0) - 0.5) * scale
    return px, py


# ---------- подписи ----------

def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _label(img: Image.Image, text: str, size: int = 14, pos=(6, 4)):
    d = ImageDraw.Draw(img)
    f = _font(size)
    x, y = pos
    bbox = d.textbbox((x, y), text, font=f)
    d.rectangle((bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2), fill=(0, 0, 0))
    d.text((x, y), text, fill=(255, 255, 255), font=f)


def _marker(img: Image.Image, px: float, py: float, r: float, color=(255, 70, 70)):
    """Четыре уголка вокруг объекта — сам объект и его окрестность остаются открытыми."""
    d = ImageDraw.Draw(img)
    a = r * 0.45
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx, cy = px + sx * r, py + sy * r
            d.line((cx, cy, cx - sx * a, cy), fill=color, width=2)
            d.line((cx, cy, cx, cy - sy * a), fill=color, width=2)


def _scalebar(img: Image.Image, scale: int, arcsec: float = 30):
    d = ImageDraw.Draw(img)
    L = arcsec / PIX_ARCSEC * scale
    x1, y = img.width - 10, img.height - 14
    d.line((x1 - L, y, x1, y), fill=(255, 255, 255), width=3)
    f = _font(12)
    d.text((x1 - L, y - 16), f'{arcsec:.0f}"', fill=(255, 255, 255), font=f)
    # стороны света: север вверх, восток влево
    d.text((10, img.height - 30), "N^  E<", fill=(200, 200, 200), font=f)


def _trail(img: Image.Image, pts, upto: int, color=(255, 210, 60)):
    """След движения: полые кружки эпох до текущей включительно и тонкая линия между ними.
    Полые — чтобы не закрывать слабый объект."""
    d = ImageDraw.Draw(img)
    seen = pts[: upto + 1]
    if len(seen) >= 2:
        d.line(seen, fill=color, width=1)
    for i, (x, y) in enumerate(seen):
        r = 3 if i < upto else 5
        d.ellipse((x - r, y - r, x + r, y + r), outline=color, width=1 if i < upto else 2)


# ---------- кадры ----------

def _det_items(sky: dict, detector: int, drop_sparse: bool = False):
    """Кадры детектора [(индекс в sky['frames'], кадр)].

    drop_sparse — выкинуть кадры, собранные из < 1/4 обычного числа снимков (шумные).
    """
    items = [(i, f) for i, f in enumerate(sky["frames"]) if f[1] == detector]
    if not items:
        raise ValueError(f"no frames for detector D{detector}; available: {sorted(sky['by_det'])}")
    info = sky.get("info") or []
    if drop_sparse and len(info) == len(sky["frames"]) and len(items) > 2:
        n = np.array([info[i]["n_exposures"] for i, _ in items])
        keep = n >= 0.25 * np.median(n)
        if keep.sum() >= 2:
            items = [it for it, k in zip(items, keep) if k]
    return items


def render_frames(sky: dict, detector: int, zoom_arcsec: float | None = None, scale: int | None = None,
                  mark_target: bool = True, track: list[SkyCoord] | None = None,
                  labels: bool = True, mode: str = "asinh", fill_cores: bool = True,
                  drop_sparse: bool = True) -> list[Image.Image]:
    """Кадры одного детектора -> список PIL-картинок (общая растяжка яркости).

    zoom_arcsec — полуширина квадрата вокруг объекта (например 90″); None — весь кадр.
    track — положения объекта по эпохам: список SkyCoord на каждый кадр детектора (по порядку,
            как в sky['frames']) или функция date -> SkyCoord. Рисуется растущий след.
            Например, предсказание по каталогу (targets.predicted_track) или измерение Питониста 3.
    drop_sparse — пропускать шумные кадры из малого числа снимков.
    """
    all_items = _det_items(sky, detector)
    items = _det_items(sky, detector, drop_sparse)
    if track is not None and not callable(track):
        pos = {i: c for (i, _), c in zip(all_items, track)}
        track = [pos[i] for i, _ in items if i in pos]
    elif callable(track):
        track = [track(f[0]) for _, f in items]
    wcs, coord = sky["wcs"], sky["coord"]
    box = zoom_box(sky["shape"], wcs, coord, zoom_arcsec)
    y0, y1, x0, x1 = box
    crops = [f[2][y0:y1, x0:x1] for _, f in items]
    if fill_cores:
        crops = [fill_saturated(c) for c in crops]
    lo, hi = stretch_limits(crops)
    if scale is None:
        scale = max(4, min(16, 480 // max(1, (x1 - x0))))
    pts = [_sky_to_canvas(c, wcs, box, scale, y1 - y0) for c in track] if track else None

    out = []
    for k, ((i, (date, det, _)), crop) in enumerate(zip(items, crops)):
        img = _to_canvas(to_uint8(crop, lo, hi, mode), scale).convert("RGB")
        if pts:
            _trail(img, pts, min(k, len(pts) - 1))
        if mark_target and coord is not None:
            px, py = _sky_to_canvas(coord, wcs, box, scale, y1 - y0)
            _marker(img, px, py, r=4 * scale)
        if labels:
            inf = sky["info"][i] if i < len(sky.get("info", [])) else {}
            wl = inf.get("wavelength_um")
            wl_txt = f"  {wl:.2f} um" if wl and np.isfinite(wl) else ""
            _label(img, f"D{det}  {date}{wl_txt}")
            _scalebar(img, scale)
        out.append(img)
    return out


def _gif_bytes(images: list[Image.Image], duration: float) -> bytes:
    buf = io.BytesIO()
    images[0].save(buf, format="GIF", save_all=True, append_images=images[1:],
                   duration=int(duration * 1000), loop=0, disposal=2)
    return buf.getvalue()


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def blink_gif(sky: dict, detector: int, zoom_arcsec: float | None = None, duration: float = 0.7,
              **kw) -> bytes:
    """Мигание одного детектора -> GIF в байтах (для st.image или скачивания)."""
    return _gif_bytes(render_frames(sky, detector, zoom_arcsec, **kw), duration)


def frame_png(sky: dict, index: int, zoom_arcsec: float | None = None, **kw) -> bytes:
    """Один кадр (по индексу в sky['frames']) -> PNG в байтах."""
    det = sky["frames"][index][1]
    items = _det_items(sky, det)
    k = [i for i, _ in items].index(index)
    return _png_bytes(render_frames(sky, det, zoom_arcsec, drop_sparse=False, **kw)[k])


def diff_png(sky: dict, index: int, zoom_arcsec: float | None = None, scale: int | None = None,
             labels: bool = True) -> bytes:
    """Разность sky['diffs'][index] -> PNG (красное — ярче, синее — темнее)."""
    date, det, img = sky["diffs"][index]
    box = zoom_box(sky["shape"], sky["wcs"], sky["coord"], zoom_arcsec)
    y0, y1, x0, x1 = box
    crop = img[y0:y1, x0:x1]
    same_det = [d[2][y0:y1, x0:x1] for d in sky["diffs"] if d[1] == det]
    lim = float(np.nanpercentile(np.abs(np.concatenate([c[np.isfinite(c)] for c in same_det])), 99.5)) or 1.0
    if scale is None:
        scale = max(4, min(16, 480 // max(1, (x1 - x0))))
    pic = _to_canvas(diverging_rgb(crop, lim), scale)
    if labels:
        _label(pic, f"D{det}  {date} - first")
        _scalebar(pic, scale)
    return _png_bytes(pic)


def save(data: bytes, path) -> None:
    """Сохранить байты (GIF/PNG) в файл — для презентации."""
    with open(path, "wb") as f:
        f.write(data)
