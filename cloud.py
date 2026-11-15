# AI disclosure: parts of this file were written with AI assistance (Claude, Anthropic), then reviewed and tested by the LUMEN team. See README, "AI use".
"""Вырезки SPHEREx прямо из открытого бакета AWS — без сервиса вырезок IRSA.

    cutout_hdul(irsa_url) -> fits.HDUList   (PRIMARY, IMAGE, FLAGS, VARIANCE, ZODI, WCS-WAVE)

Бакет: https://nasa-irsa-spherex.s3.amazonaws.com/  (s3://nasa-irsa-spherex, us-east-1, без ключей).
Ключ файла = путь из IRSA TAP без префикса "ibe/data/spherex/".

Снимки Level 2 — несжатые float32 2040×2040 (FLAGS сжат RICE построчно), поэтому
HTTP Range-запросами читаются только нужные строки: ~1–2 МБ вместо 65 МБ на снимок.
Результат совпадает по формату с вырезкой IRSA: те же HDU, WCS сдвинут на угол вырезки
(CRPIXn, CRPIXnA, CRPIXnW), поэтому data.load_cutout и всё дальше работают без изменений.
"""
from __future__ import annotations

import io
import re
import threading
import warnings
from urllib.parse import parse_qs, urlsplit

import numpy as np
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_scales

warnings.simplefilter("ignore", FITSFixedWarning)

BUCKET = "https://nasa-irsa-spherex.s3.amazonaws.com/"
IBE_PREFIX = "ibe/data/spherex/"
KEEP = ("IMAGE", "FLAGS", "VARIANCE", "ZODI", "WCS-WAVE")
BLOCK = 2880
_HDR_GUESS = 40320          # хватает на заголовок FLAGS + таблицу дескрипторов строк
_COMPRESSION_KEY = re.compile(
    r"^Z(IMAGE|TENSION|BITPIX|NAXIS\d*|PCOUNT|GCOUNT|TILE\d+|CMPTYPE|NAME\d+|VAL\d+|"
    r"QUANTIZ|DITHER0|SIMPLE|EXTEND|BLOCKED|HECKSUM|DATASUM)$")
_TABLE_KEY = re.compile(r"^(TTYPE|TFORM|TDIM|TUNIT|TNULL|TSCAL|TZERO|TDISP)\d+$|^(TFIELDS|THEAP)$")
_CRPIX = re.compile(r"^CRPIX([12])([A-Z]?)$")

_local = threading.local()


def _session() -> requests.Session:
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=8))
        _local.s = s
    return s


def parse_irsa_url(url: str):
    """IRSA-ссылка на вырезку -> (ключ в бакете, ra, dec, size_deg) или None."""
    u = urlsplit(url)
    path = u.path.lstrip("/")
    if not path.startswith(IBE_PREFIX):
        return None
    q = parse_qs(u.query)
    try:
        ra, dec = (float(v) for v in q["center"][0].rstrip("d").split(","))
        size = float(q["size"][0])
    except (KeyError, ValueError):
        return None
    return path[len(IBE_PREFIX):], ra, dec, size


def _get(url: str, a: int, b: int) -> bytes:
    """Байты [a, b) файла (с повтором при обрыве соединения или 5xx)."""
    for attempt in range(4):
        try:
            r = _session().get(url, headers={"Range": f"bytes={a}-{b - 1}"}, timeout=60)
            if r.status_code == 416:
                return b""
            if r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code}", response=r)
            r.raise_for_status()
            if len(r.content) != b - a and r.status_code == 206 and not r.headers.get("Content-Range", "").endswith(f"/{a + len(r.content)}"):
                raise requests.ConnectionError("short response")
            return r.content
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            if isinstance(e, requests.HTTPError) and e.response is not None and e.response.status_code < 500:
                raise
            if attempt == 3:
                raise
            _local.s = None            # новое соединение
            import time
            time.sleep(0.5 * (attempt + 1))


def _header_at(url: str, off: int, guess: int = _HDR_GUESS):
    """Заголовок HDU по смещению off -> (Header, длина заголовка, байты, прочитанные сверх него)."""
    buf = _get(url, off, off + guess)
    pos = 0
    while True:
        if pos + BLOCK > len(buf):
            more = _get(url, off + len(buf), off + len(buf) + max(guess, BLOCK))
            if not more:
                raise ValueError(f"no end of header at offset {off}")
            buf += more
        blk = buf[pos:pos + BLOCK]
        pos += BLOCK
        if any(blk[k:k + 3] == b"END" and not blk[k + 3:k + 80].strip() for k in range(0, BLOCK, 80)):
            break
    return fits.Header.fromstring(buf[:pos].decode("ascii")), pos, buf[pos:]


def _data_bytes(h: fits.Header) -> int:
    n = h.get("NAXIS", 0)
    if not n:
        return 0
    size = abs(h["BITPIX"]) // 8 * int(np.prod([h[f"NAXIS{i}"] for i in range(1, n + 1)]))
    return size + h.get("PCOUNT", 0)


def _pad(n: int) -> int:
    return (n + BLOCK - 1) // BLOCK * BLOCK


def _walk(url: str) -> dict:
    """Заголовки нужных HDU: {имя: (Header, смещение данных, лишние байты)}."""
    out, off = {}, 0
    for _ in range(12):
        h, hlen, extra = _header_at(url, off)
        name = h.get("EXTNAME", "PRIMARY").strip()
        out[name] = (h, off + hlen, extra)
        if all(k in out for k in KEEP):
            break
        off += hlen + _pad(_data_bytes(h))
    missing = [k for k in KEEP if k not in out]
    if missing:
        raise ValueError(f"HDU {missing} not found in the file")
    return out


def _box(h: fits.Header, ra: float, dec: float, size_deg: float):
    """Пиксельный прямоугольник вырезки size_deg вокруг (ra, dec), обрезанный краями детектора."""
    w = WCS(h)
    x, y = (float(v) for v in w.world_to_pixel(SkyCoord(ra, dec, unit="deg")))
    scale = float(np.mean(proj_plane_pixel_scales(w.celestial)))
    n = max(1, int(round(size_deg / scale)))
    nx, ny = h["NAXIS1"], h["NAXIS2"]
    x0, y0 = int(round(x - n / 2)), int(round(y - n / 2))
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(nx, x0 + n), min(ny, y0 + n)
    if x1c - x0c < 2 or y1c - y0c < 2:
        raise ValueError("point is outside the detector")
    return x0c, x1c, y0c, y1c


def _rows_f32(url: str, h: fits.Header, data0: int, x0, x1, y0, y1) -> np.ndarray:
    if h["BITPIX"] != -32:
        raise ValueError(f"expected float32, BITPIX={h['BITPIX']}")
    nx = h["NAXIS1"]
    raw = _get(url, data0 + y0 * nx * 4, data0 + y1 * nx * 4)
    arr = np.frombuffer(raw, ">f4").reshape(y1 - y0, nx)[:, x0:x1].astype(np.float32)
    return arr * h.get("BSCALE", 1.0) + h.get("BZERO", 0.0) if ("BSCALE" in h or "BZERO" in h) else arr


def _rows_rice(url: str, h: fits.Header, data0: int, extra: bytes, x0, x1, y0, y1) -> np.ndarray:
    """Строки y0..y1 сжатого RICE_1 изображения (плитки по одной строке)."""
    from astropy.io.fits.hdu.compressed._codecs import Rice1
    nx = h["ZNAXIS1"]
    if (h.get("ZCMPTYPE", "").strip() != "RICE_1" or h.get("ZTILE1") != nx or h.get("ZTILE2", 1) != 1
            or not str(h["TFORM1"]).startswith("1P")):
        raise ValueError("unsupported FLAGS compression")
    zval = {h[f"ZNAME{i}"].strip(): h[f"ZVAL{i}"] for i in range(1, 10) if f"ZNAME{i}" in h}
    bytepix, blocksize = int(zval.get("BYTEPIX", 4)), int(zval.get("BLOCKSIZE", 32))
    table_len = h["NAXIS1"] * h["NAXIS2"]
    desc_raw = extra[:table_len] if len(extra) >= table_len else _get(url, data0, data0 + table_len)
    desc = np.frombuffer(desc_raw, ">i4").reshape(-1, 2)[y0:y1]
    cnt, off = desc[:, 0].astype(np.int64), desc[:, 1].astype(np.int64)
    heap0 = data0 + h.get("THEAP", table_len)
    lo, hi = int(off.min()), int((off + cnt).max())
    heap = _get(url, heap0 + lo, heap0 + hi)
    codec = Rice1(blocksize=blocksize, bytepix=bytepix, tilesize=nx)
    dt = {1: np.uint8, 2: np.int16, 4: np.int32}[bytepix]
    rows = [np.asarray(codec.decode(np.frombuffer(heap[o - lo:o - lo + c], np.uint8))).view(dt)
            for c, o in zip(cnt, off)]
    arr = np.stack(rows)[:, x0:x1].astype(np.int64)
    if "BZERO" in h or "BSCALE" in h:
        arr = arr * h.get("BSCALE", 1) + h.get("BZERO", 0)
    return arr.astype(np.int32)


def _shift_wcs(h: fits.Header, x0: int, y0: int) -> fits.Header:
    h = h.copy()
    for k in list(h.keys()):
        m = _CRPIX.match(k)
        if m:
            h[k] = h[k] - (x0 if m.group(1) == "1" else y0)
    for k in ("CHECKSUM", "DATASUM"):
        h.remove(k, ignore_missing=True)
    return h


def _image_header(table_hdr: fits.Header) -> fits.Header:
    """Заголовок сжатой таблицы -> заголовок обычного изображения (без служебных ключей)."""
    h = fits.Header()
    for card in table_hdr.cards:
        k = card.keyword
        if k in ("XTENSION", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "PCOUNT", "GCOUNT", "BZERO", "BSCALE") \
                or _COMPRESSION_KEY.match(k) or _TABLE_KEY.match(k):
            continue
        h.append(card)
    return h


def _small_hdu(h: fits.Header, extra: bytes, data0: int, url: str):
    """Маленькое HDU целиком (WCS-WAVE: заголовок + 720 байт данных)."""
    n = _pad(_data_bytes(h))
    data = extra[:n] if len(extra) >= n else _get(url, data0, data0 + n)
    raw = fits.PrimaryHDU().header.tostring().encode() + h.tostring().encode() + data
    with fits.open(io.BytesIO(raw)) as hd:
        return hd[1].copy()


_TAIL = 64 * 1024          # конец файла: заголовок и данные WCS-WAVE (последний HDU)
_POOL = None
_POOL_LOCK = threading.Lock()


def _pool():
    """Общий пул для запросов внутри одного файла (сессии в его потоках переиспользуются)."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            from concurrent.futures import ThreadPoolExecutor
            _POOL = ThreadPoolExecutor(max_workers=48, thread_name_prefix="s3")
        return _POOL


def _tail_wave(url: str):
    """WCS-WAVE из хвоста файла (один запрос) -> (Header, смещение данных, байты данных) или None."""
    r = _session().get(url, headers={"Range": f"bytes=-{_TAIL}"}, timeout=60)
    r.raise_for_status()
    total = int(r.headers["Content-Range"].rsplit("/", 1)[1])
    buf, start = r.content, total - len(r.content)
    first = (-start) % BLOCK           # блоки FITS выровнены от начала файла
    for pos in range(first, len(buf) - BLOCK + 1, BLOCK):
        if not buf.startswith(b"XTENSION= 'BINTABLE'", pos):
            continue
        try:
            h, hlen, _ = _header_from(buf, pos)
        except ValueError:
            continue
        if h.get("EXTNAME", "").strip() == "WCS-WAVE":
            return h, start + pos + hlen, buf[pos + hlen:]
    return None


def _header_from(buf: bytes, pos: int):
    i = pos
    while i + BLOCK <= len(buf):
        blk = buf[i:i + BLOCK]
        i += BLOCK
        if any(blk[k:k + 3] == b"END" and not blk[k + 3:k + 80].strip() for k in range(0, BLOCK, 80)):
            return fits.Header.fromstring(buf[pos:i].decode("ascii")), i - pos, buf[i:]
    raise ValueError("header does not fit in the buffer")


def _is(h: fits.Header, name: str) -> bool:
    return h.get("EXTNAME", "").strip() == name


def cutout_hdul(irsa_url: str) -> fits.HDUList:
    """Вырезка из бакета AWS по ссылке IRSA (…/ibe/data/spherex/<ключ>?center=ra,decd&size=deg).

    Цепочка ожиданий: начало файла -> заголовок FLAGS -> заголовки VARIANCE и ZODI -> строки;
    хвост с WCS-WAVE и строки IMAGE/FLAGS читаются параллельно.
    """
    parsed = parse_irsa_url(irsa_url)
    if parsed is None:
        raise ValueError(f"not an IRSA SPHEREx URL: {irsa_url}")
    key, ra, dec, size = parsed
    url = BUCKET + key
    pool = _pool()

    tail_f = pool.submit(_tail_wave, url)
    prim_h, plen, extra = _header_at(url, 0)
    img_off = plen + _pad(_data_bytes(prim_h))
    if img_off - plen <= len(extra) and extra[img_off - plen:].startswith(b"XTENSION"):
        img_h, ilen, _ = _header_from(extra, img_off - plen)
    else:
        img_h, ilen, _ = _header_at(url, img_off)
    if not _is(img_h, "IMAGE"):
        return _cutout_walk(url, ra, dec, size)
    img0 = img_off + ilen
    x0, x1, y0, y1 = _box(img_h, ra, dec, size)
    box = (x0, x1, y0, y1)

    img_f = pool.submit(_rows_f32, url, img_h, img0, *box)
    flags_off = img0 + _pad(_data_bytes(img_h))
    fl_h, flen, fl_extra = _header_at(url, flags_off)
    if not _is(fl_h, "FLAGS"):
        return _cutout_walk(url, ra, dec, size)
    fl0 = flags_off + flen
    fl_f = pool.submit(_rows_rice if fl_h.get("ZIMAGE") else _rows_f32_any, url, fl_h, fl0, fl_extra, *box)

    var_off = fl0 + _pad(_data_bytes(fl_h))
    guess_zodi = var_off + 11520 + _pad(_data_bytes(img_h))  # заголовок VARIANCE обычно 11520 байт
    var_hf = pool.submit(_header_at, url, var_off)
    zod_hf = pool.submit(_header_at, url, guess_zodi)
    var_h, vlen, _ = var_hf.result()
    if not _is(var_h, "VARIANCE"):
        return _cutout_walk(url, ra, dec, size)
    var_f = pool.submit(_rows_f32, url, var_h, var_off + vlen, *box)
    zodi_off = var_off + vlen + _pad(_data_bytes(var_h))
    try:
        zod_h, zlen, _ = zod_hf.result()
    except Exception:
        zod_h = fits.Header()
    if zodi_off != guess_zodi or not _is(zod_h, "ZODI"):
        zod_h, zlen, _ = _header_at(url, zodi_off)
        if not _is(zod_h, "ZODI"):
            return _cutout_walk(url, ra, dec, size)
    zod_f = pool.submit(_rows_f32, url, zod_h, zodi_off + zlen, *box)

    wave = tail_f.result()
    if wave is None:
        return _cutout_walk(url, ra, dec, size)
    wave_hdu = _small_hdu(wave[0], wave[2], wave[1], url)

    hdus = [fits.PrimaryHDU(header=_shift_wcs(prim_h, 0, 0))]
    for name, h, fut in (("IMAGE", img_h, img_f), ("FLAGS", fl_h, fl_f),
                         ("VARIANCE", var_h, var_f), ("ZODI", zod_h, zod_f)):
        hdus.append(_image_hdu(name, h, fut.result(), x0, y0))
    hdus.append(wave_hdu)
    return fits.HDUList(hdus)


def _rows_f32_any(url, h, data0, _extra, x0, x1, y0, y1):
    return _rows_f32(url, h, data0, x0, x1, y0, y1)


def _image_hdu(name: str, h: fits.Header, data: np.ndarray, x0: int, y0: int) -> fits.ImageHDU:
    if h.get("ZIMAGE"):
        h = _image_header(h)
    else:
        h = h.copy()
        for k in ("BZERO", "BSCALE"):
            h.remove(k, ignore_missing=True)
    return fits.ImageHDU(data=data, header=_shift_wcs(h, x0, y0), name=name)


def _cutout_walk(url: str, ra: float, dec: float, size: float) -> fits.HDUList:
    """Медленный, но общий путь: последовательный обход всех HDU (если раскладка файла другая)."""
    hd = _walk(url)
    x0, x1, y0, y1 = _box(hd["IMAGE"][0], ra, dec, size)
    hdus = [fits.PrimaryHDU(header=_shift_wcs(hd["PRIMARY"][0], 0, 0))]
    for name in ("IMAGE", "FLAGS", "VARIANCE", "ZODI"):
        h, d0, extra = hd[name]
        data = (_rows_rice(url, h, d0, extra, x0, x1, y0, y1) if h.get("ZIMAGE")
                else _rows_f32(url, h, d0, x0, x1, y0, y1))
        hdus.append(_image_hdu(name, h, data, x0, y0))
    h, d0, extra = hd["WCS-WAVE"]
    hdus.append(_small_hdu(h, extra, d0, url))
    return fits.HDUList(hdus)
