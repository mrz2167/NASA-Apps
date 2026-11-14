"""Automatic, conservative cleanup of old visitor fields on the cache disk.

Every new sky position a visitor opens adds 100-700 cutout files (plus a query, a Gaia table, maybe WISE
frames) and nothing removed them, so the 5 GB volume would fill up. This module frees space by itself:

    trigger    only when free space on the cache disk is below SKYWATCH_CLEANUP_MIN_FREE_GB (1.0);
               then whole fields are deleted, least recently used first, until free space is above
               SKYWATCH_CLEANUP_TARGET_FREE_GB (1.5) or nothing deletable is left.
    deletable  cutouts/*.fits, queries/*.ecsv, gaia/*.ecsv, wise/<key>/, fields/*.json of fields that are
               not protected, not used in the last SKYWATCH_CLEANUP_KEEP_HOURS (24) and have no file
               modified in the last SKYWATCH_CLEANUP_FRESH_MIN (60) minutes.
    protected  every position returned by protected_fields(): the demo/"Try" fields, "Surprise me", any
               other (name, ra, dec) list in main.py, PINNED_FIELDS below, and seed/queries. Never touched:
               index/, skies/ (skycache prunes itself), reports.db, anything this module doesn't recognise.
    last used  cache/fields/<ra>_<dec>_<size>.json, written when the server serves a field (note_used):
               its mtime is the "last opened" time and it lists the field's cutout files, so fields built
               from the AWS index (no query file) are known too. Without a record the newest file counts.

Nothing here runs at import time or inside a request. main.py starts one daemon thread (start()) and
wraps _load_sky with track(). On the command line the default is a dry run:

    python cleanup.py                 # report: what is protected, what could be deleted, sizes
    python cleanup.py --delete        # really delete, same rules as the server (only below the threshold)
    python cleanup.py --delete --force   # ...even when the disk isn't nearly full

SKYWATCH_CLEANUP: "1" on, "0" off; unset = on only on Railway (a RAILWAY_* variable is present), so a
developer's laptop with a full disk never deletes its own cache.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED = ROOT / "seed" / "queries"

# Extra fields that must never be deleted: add (name, ra, dec) tuples here, or keep any UPPERCASE list of
# such tuples at module level in main.py (it is picked up by fields_in_module), or call protect().
PINNED_FIELDS: list[tuple[str, float, float]] = []

DELETABLE_DIRS = ("cutouts", "queries", "gaia", "wise", "fields")   # the only folders anything is deleted from
KEPT_DIRS = ("skies", "index")                                      # known, never touched here
GROUP_DEG = 0.01            # files whose centres are this close belong to one field


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    min_free: float = 1.0e9          # run only below this many free bytes
    target_free: float = 1.5e9       # ...and delete until this many are free
    keep_s: float = 24 * 3600        # a field used this recently is never deleted
    fresh_s: float = 3600            # a field with a file modified this recently is never deleted
    protect_deg: float = 0.03        # a field this close to a protected position is protected
    orphans: bool = False            # also delete old cutouts no query/record refers to (off: report only)
    every_s: float = 600             # how often the background job looks at the disk

    @classmethod
    def from_env(cls) -> "Config":
        c = cls()
        c.min_free = _env_float("SKYWATCH_CLEANUP_MIN_FREE_GB", 1.0) * 1e9
        c.target_free = max(c.min_free, _env_float("SKYWATCH_CLEANUP_TARGET_FREE_GB", 1.5) * 1e9)
        c.keep_s = max(0.0, _env_float("SKYWATCH_CLEANUP_KEEP_HOURS", 24)) * 3600
        c.fresh_s = max(0.0, _env_float("SKYWATCH_CLEANUP_FRESH_MIN", 60)) * 60
        c.protect_deg = max(GROUP_DEG, _env_float("SKYWATCH_CLEANUP_PROTECT_DEG", 0.03))
        c.orphans = os.environ.get("SKYWATCH_CLEANUP_ORPHANS", "0") == "1"
        c.every_s = max(30.0, _env_float("SKYWATCH_CLEANUP_EVERY_MIN", 10) * 60)
        return c


def enabled() -> bool:
    """Is the automatic job on? SKYWATCH_CLEANUP=1/0; unset -> only on Railway."""
    v = os.environ.get("SKYWATCH_CLEANUP", "").strip().lower()
    if v in ("1", "true", "on", "yes"):
        return True
    if v in ("0", "false", "off", "no"):
        return False
    return any(k.startswith("RAILWAY_") for k in os.environ)


def _cache_root() -> Path:
    from data import CACHE
    return Path(CACHE)


def sep_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    a1, d1, a2, d2 = (math.radians(v) for v in (ra1, dec1, ra2, dec2))
    h = math.sin((d2 - d1) / 2) ** 2 + math.cos(d1) * math.cos(d2) * math.sin((a2 - a1) / 2) ** 2
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(h))))


# ---------- protected fields ----------

_PROVIDERS: list[tuple[str, object, bool]] = []      # (label, list or callable, must not be empty)


def protect(label: str, source, required: bool = False) -> None:
    """Register a list of fields that must never be deleted.

    source: a list (read again on every run, so later appends count) or a function returning one, of
    (name, ra, dec) / (ra, dec) tuples or dicts with "ra" and "dec"; or a module, in which case every
    UPPERCASE module-level list of (name, ra, dec) tuples in it is protected (fields_in_module).
    required=True: if this source comes back empty or fails, nothing is deleted at all.
    """
    _PROVIDERS[:] = [p for p in _PROVIDERS if p[0] != label] + [(label, source, required)]


def _positions(items) -> list[tuple[float, float]]:
    out = []
    for it in items or []:
        try:
            if isinstance(it, dict):
                ra, dec = float(it["ra"]), float(it["dec"])
            else:
                nums = [v for v in it if isinstance(v, (int, float)) and not isinstance(v, bool)]
                ra, dec = float(nums[0]), float(nums[1])
        except (TypeError, ValueError, KeyError, IndexError):
            continue
        if 0 <= ra <= 360 and -90 <= dec <= 90:
            out.append((ra, dec))
    return out


def fields_in_module(mod) -> list[tuple[str, float, float]]:
    """Every UPPERCASE module-level list of (name, ra, dec) tuples (WARM_FIELDS and any list added later)."""
    out = []
    for name, val in list(vars(mod).items()):
        if not name.isupper() or not isinstance(val, (list, tuple)) or not val:
            continue
        ok = all(isinstance(e, (list, tuple)) and len(e) >= 3 and isinstance(e[0], str)
                 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in e[1:3])
                 and 0 <= e[1] <= 360 and -90 <= e[2] <= 90 for e in val)
        if ok:
            out.extend((e[0], float(e[1]), float(e[2])) for e in val)
    return out


def protected_fields() -> tuple[list[tuple[float, float]], list[str]]:
    """(positions that must never be deleted, problems). Any problem means: don't delete anything."""
    pos, problems = _positions(PINNED_FIELDS), []
    for label, source, required in list(_PROVIDERS):
        try:
            if isinstance(source, type(sys)):
                items = fields_in_module(source)
            else:
                items = source() if callable(source) else source
            got = _positions(items)
        except Exception as e:
            got = []
            problems.append(f"protected list '{label}' failed: {e!r}")
        if required and not got:
            problems.append(f"protected list '{label}' is empty")
        pos.extend(got)
    return pos, problems


# ---------- "last opened" records (the only thing this module writes during normal serving) ----------

_ACTIVE: dict[tuple[float, float], float] = {}     # (ra, dec) rounded -> last time a request asked for it
_NOTED: dict[str, tuple[int, float]] = {}          # record name -> (cutouts listed, when written)
NOTE_EVERY_S = 600


def _field_id(ra: float, dec: float, size: float) -> str:
    return f"{ra:.4f}_{dec:+.4f}_{size:g}"


def note_active(ra: float, dec: float) -> None:
    """A request is opening this position (in memory only; called before the field is built)."""
    try:
        _ACTIVE[(round(float(ra), 3), round(float(dec), 3))] = time.time()
        if len(_ACTIVE) > 5000:
            cut = time.time() - 48 * 3600
            for k in [k for k, t in list(_ACTIVE.items()) if t < cut]:
                _ACTIVE.pop(k, None)
    except Exception:
        pass


def note_used(sky: dict) -> None:
    """Record that this field was just served: touch cache/fields/<id>.json (mtime = last opened) and
    keep the list of its cutout files there. At most one small write per field per 10 minutes; never raises."""
    try:
        ra, dec, size = float(sky["coord"].ra.deg), float(sky["coord"].dec.deg), float(sky["size_deg"])
        note_active(ra, dec)
        cuts = sky.get("cutouts") or []
        fid = _field_id(ra, dec, size)
        now = time.time()
        seen = _NOTED.get(fid)
        if seen and seen[0] == len(cuts) and now - seen[1] < NOTE_EVERY_S:
            return
        from data import cutout_path
        names = {cutout_path(c["url"]).name for c in cuts if isinstance(c, dict) and c.get("url")}
        folder = _cache_root() / "fields"
        path = folder / f"{fid}.json"
        old: set = set()
        if path.exists():
            try:
                old = set(json.loads(path.read_text(encoding="utf-8")).get("cutouts", []))
            except Exception:
                old = set()
        if path.exists() and names <= old:
            os.utime(path, None)
        else:
            folder.mkdir(parents=True, exist_ok=True)
            tmp = folder / f"{fid}.json.tmp{threading.get_ident()}"
            tmp.write_text(json.dumps({"ra": ra, "dec": dec, "size_deg": size, "cutouts": sorted(old | names)}),
                           encoding="utf-8")
            os.replace(tmp, path)
        _NOTED[fid] = (len(cuts), now)
        if len(_NOTED) > 5000:
            _NOTED.clear()
    except Exception as e:
        try:
            print(f"[cleanup] could not record field use: {e!r}", flush=True)
        except Exception:
            pass


def track(load_sky):
    """Wrap main._load_sky: note the position before the field is built and record its use afterwards."""
    import functools

    @functools.wraps(load_sky)
    def wrapper(ra, dec, *args, **kw):
        note_active(ra, dec)
        sky = load_sky(ra, dec, *args, **kw)
        note_used(sky)
        return sky
    return wrapper


# ---------- reading the cache (no writes below until _delete_group) ----------

_HEX12 = re.compile(r"^[0-9a-f]{12}$")
_GAIA_NAME = re.compile(r"^(\d+\.\d+)_([+-]\d+\.\d+)_([0-9.]+)\.ecsv$")
_META = re.compile(r"^#\s*-\s*\{\s*(ra|dec|size_deg):\s*([-+0-9.eE]+)\s*\}")
_QUERY_CACHE: dict = {}       # (path, mtime, size) -> parsed query


def _cutout_name(url: str) -> str:
    from data import cutout_path
    return cutout_path(url).name


def read_query(path: Path) -> dict | None:
    """{ra, dec, size_deg, cutouts: set of file names} of a saved search, or None if it can't be understood."""
    try:
        st = path.stat()
        key = (str(path), st.st_mtime, st.st_size)
        if key in _QUERY_CACHE:
            return _QUERY_CACHE[key]
        meta, urls, header = {}, [], False
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    m = _META.match(line)
                    if m:
                        meta[m.group(1)] = float(m.group(2))
                elif not line.strip():
                    continue
                elif not header:
                    if line.split()[0] != "url":
                        raise ValueError("first column is not url")
                    header = True
                else:
                    url = line.split(" ", 1)[0].strip().strip('"')
                    if not url.startswith("http"):
                        raise ValueError("row without a url")
                    urls.append(url)
        if not header or not {"ra", "dec"} <= set(meta):
            raise ValueError("no header or no centre")
    except Exception:
        try:                                    # an ECSV layout the quick reader doesn't know: ask astropy
            from astropy.table import Table
            t = Table.read(path)
            meta = {k: float(t.meta[k]) for k in ("ra", "dec")}
            meta["size_deg"] = float(t.meta.get("size_deg", 0.1))
            urls = [str(u) for u in t["url"]]
        except Exception:
            return None
    q = {"ra": meta["ra"], "dec": meta["dec"], "size_deg": meta.get("size_deg", 0.1),
         "cutouts": {_cutout_name(u) for u in urls}}
    try:
        if len(_QUERY_CACHE) > 5000:
            _QUERY_CACHE.clear()
        _QUERY_CACHE[key] = q
    except Exception:
        pass
    return q


def wise_centre(meta: dict) -> tuple[float, float]:
    """Sky position of the middle of a cached WISE cutout. CRVAL in WiseView's answer is the centre of the
    unWISE tile (up to a degree away), so the cutout centre comes from the frame's WCS (as wise._wcs)."""
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [float(meta["CRVAL1"]), float(meta["CRVAL2"])]
    w.wcs.crpix = [float(meta["CRPIX1"]), float(meta["CRPIX2"])]
    w.wcs.cdelt = [-2.75 / 3600, 2.75 / 3600]
    ra, dec = w.wcs_pix2world((float(meta["NAXIS1"]) + 1) / 2, (float(meta["NAXIS2"]) + 1) / 2, 1)
    ra, dec = float(ra) % 360, float(dec)
    if not (math.isfinite(ra) and math.isfinite(dec)):
        raise ValueError("no position")
    return ra, dec


@dataclass
class Group:
    """Everything on disk that belongs to one sky position."""
    ra: float
    dec: float
    files: list = field(default_factory=list)       # (kind, Path, bytes, mtime) of query/gaia/record/wise files
    dirs: list = field(default_factory=list)        # wise folders (removed once empty)
    cutouts: set = field(default_factory=set)       # cutout file names this field refers to
    last_used: float = 0.0
    newest: float = 0.0
    status: str = "deletable"                       # deletable | protected | recent | fresh
    bytes: int = 0                                  # own files + the cutouts that exist


def check_root(root: Path) -> str | None:
    """Why this folder must not be cleaned (None = it looks like our cache)."""
    try:
        r = root.resolve()
    except OSError as e:
        return f"cache folder not readable: {e}"
    if not r.is_dir():
        return f"cache folder {r} does not exist"
    if len(r.parts) < 2:
        return f"refusing to clean a drive root ({r})"
    if not (r / "cutouts").is_dir():
        return f"{r} has no 'cutouts' folder: not a cache"
    if any((r / n).exists() for n in (".git", "main.py", "requirements.txt")):
        return f"{r} looks like a source tree, not a cache"
    return None


def _inside(path: Path, root: Path, sub: str) -> bool:
    """True only if path really is root/<sub>/… for a deletable folder (links and '..' resolved)."""
    if sub not in DELETABLE_DIRS:
        return False
    try:
        base = (root.resolve() / sub)
        p = path.resolve()
        rel = p.relative_to(base)
    except (OSError, ValueError):
        return False
    return len(rel.parts) in ((1, 2) if sub == "wise" else (1,)) and base.parent == root.resolve()


def _listdir(folder: Path):
    try:
        with os.scandir(folder) as it:
            return sorted(it, key=lambda e: e.name)
    except OSError:
        return []


def scan(root: Path, seed: Path | None = SEED) -> dict:
    """Read-only inventory of the cache: fields (grouped by position), cutout files, things left alone."""
    root = Path(root)
    groups: list[Group] = []
    unknown: list[tuple[str, int]] = []
    bad_queries = 0

    def group_for(ra: float, dec: float) -> Group:
        for g in groups:
            if sep_deg(ra, dec, g.ra, g.dec) <= GROUP_DEG:
                return g
        groups.append(Group(ra, dec))
        return groups[-1]

    def add(kind: str, e, ra: float, dec: float) -> Group:
        st = e.stat(follow_symlinks=False)
        g = group_for(ra, dec)
        g.files.append((kind, Path(e.path), st.st_size, st.st_mtime))
        return g

    def skip(e) -> None:
        try:
            size = e.stat(follow_symlinks=False).st_size if not e.is_dir(follow_symlinks=False) else \
                sum(x.stat(follow_symlinks=False).st_size for x in _listdir(e.path) if x.is_file(follow_symlinks=False))
        except OSError:
            size = 0
        unknown.append((str(Path(e.path).relative_to(root)), size))

    def plain_file(e, sub: str) -> bool:
        return (not e.is_symlink() and e.is_file(follow_symlinks=False) and _inside(Path(e.path), root, sub))

    # cutouts: name -> (bytes, mtime)
    cutouts: dict[str, tuple[int, float]] = {}
    for e in _listdir(root / "cutouts"):
        if e.name.endswith(".fits") and plain_file(e, "cutouts"):
            st = e.stat(follow_symlinks=False)
            cutouts[e.name] = (st.st_size, st.st_mtime)
        else:
            skip(e)

    for e in _listdir(root / "queries"):
        q = read_query(Path(e.path)) if e.name.endswith(".ecsv") and plain_file(e, "queries") else None
        if q is None:
            bad_queries += e.name.endswith(".ecsv")
            skip(e)
            continue
        add("query", e, q["ra"], q["dec"]).cutouts |= q["cutouts"]

    for e in _listdir(root / "fields"):
        rec = None
        if e.name.endswith(".json") and plain_file(e, "fields"):
            try:
                rec = json.loads(Path(e.path).read_text(encoding="utf-8"))
                rec = (float(rec["ra"]), float(rec["dec"]), {str(n) for n in rec.get("cutouts", [])})
            except Exception:
                rec = None
        if rec is None:
            skip(e)
            continue
        g = add("record", e, rec[0], rec[1])
        g.cutouts |= {n for n in rec[2] if "/" not in n and "\\" not in n and n.endswith(".fits")}

    for e in _listdir(root / "gaia"):
        m = _GAIA_NAME.match(e.name)
        if m and plain_file(e, "gaia"):
            add("gaia", e, float(m.group(1)), float(m.group(2)))
        else:
            skip(e)

    for e in _listdir(root / "wise"):
        inner = _listdir(e.path) if e.is_dir(follow_symlinks=False) and not e.is_symlink() else None
        pos = None
        if inner is not None and _HEX12.match(e.name) and _inside(Path(e.path), root, "wise") and \
                all(re.match(r"^(meta\.json|\d\d\.png)$", x.name) and plain_file(x, "wise") for x in inner):
            try:
                pos = wise_centre(json.loads((Path(e.path) / "meta.json").read_text(encoding="utf-8")))
            except Exception:
                pos = None
        if pos is None or not (0 <= pos[0] <= 360 and -90 <= pos[1] <= 90):
            skip(e)
            continue
        g = None
        for x in inner:
            g = add("wise", x, *pos)
        g.dirs.append(Path(e.path))

    # root level: known folders and the reports database are left alone silently, the rest is reported
    for e in _listdir(root):
        if e.name in DELETABLE_DIRS or e.name in KEPT_DIRS or e.name.startswith("reports.db"):
            continue
        skip(e)

    seed_pos, seed_cutouts = [], set()
    if seed is not None:
        for e in _listdir(seed):
            q = read_query(Path(e.path)) if e.name.endswith(".ecsv") else None
            if q is not None:
                seed_pos.append((q["ra"], q["dec"]))
                seed_cutouts |= q["cutouts"]

    referenced = set(seed_cutouts)
    for g in groups:
        referenced |= g.cutouts
    orphans = {n: v for n, v in cutouts.items() if n not in referenced}
    return {"root": root, "groups": groups, "cutouts": cutouts, "orphans": orphans, "unknown": unknown,
            "bad_queries": bad_queries, "seed_pos": seed_pos, "seed_cutouts": seed_cutouts}


def plan(root: Path, cfg: Config | None = None, protected: list | None = None, now: float | None = None,
         active: dict | None = None, seed: Path | None = SEED) -> dict:
    """Classify every field and order the deletable ones, least recently used first. Reads only."""
    cfg = cfg or Config.from_env()
    now = time.time() if now is None else now
    inv = scan(root, seed)
    prot = list(protected or []) + inv["seed_pos"]
    act = [(k[0], k[1], t) for k, t in list((_ACTIVE if active is None else active).items())]
    cutouts = inv["cutouts"]
    for g in inv["groups"]:
        times = [m for _, _, _, m in g.files] + [cutouts[n][1] for n in g.cutouts if n in cutouts]
        g.newest = max(times) if times else 0.0
        g.last_used = max([g.newest] + [t for ra, dec, t in act if sep_deg(ra, dec, g.ra, g.dec) <= GROUP_DEG * 2])
        g.bytes = sum(s for _, _, s, _ in g.files) + sum(cutouts[n][0] for n in g.cutouts if n in cutouts)
        if any(sep_deg(ra, dec, g.ra, g.dec) <= cfg.protect_deg for ra, dec in prot) or g.cutouts & inv["seed_cutouts"]:
            g.status = "protected"
        elif now - g.last_used < cfg.keep_s:
            g.status = "recent"
        elif now - g.newest < cfg.fresh_s:
            g.status = "fresh"
        else:
            g.status = "deletable"

    order = sorted((g for g in inv["groups"] if g.status == "deletable"), key=lambda g: (g.last_used, g.ra, g.dec))
    refs: dict[str, int] = {}
    for g in inv["groups"]:
        for n in g.cutouts:
            refs[n] = refs.get(n, 0) + 1
    for n in inv["seed_cutouts"]:
        refs[n] = refs.get(n, 0) + 1_000_000
    steps = []
    for g in order:                              # a cutout goes only when no field that stays refers to it
        gone = []
        for n in sorted(g.cutouts):
            refs[n] -= 1
            if refs[n] == 0 and n in cutouts:
                gone.append(n)
        size = sum(s for _, _, s, _ in g.files) + sum(cutouts[n][0] for n in gone)
        steps.append({"group": g, "cutouts": gone, "bytes": size})

    old_orphans = {n: v for n, v in inv["orphans"].items() if now - v[1] >= max(cfg.keep_s, cfg.fresh_s)}
    by_status: dict[str, dict] = {}
    for g in inv["groups"]:
        s = by_status.setdefault(g.status, {"fields": 0, "bytes": 0})
        s["fields"] += 1
        s["bytes"] += g.bytes
    return {"root": Path(root), "cfg": cfg, "now": now, "inventory": inv, "steps": steps, "by_status": by_status,
            "deletable_bytes": sum(s["bytes"] for s in steps),
            "orphans_deletable": inv["bad_queries"] == 0 and cfg.orphans,
            "old_orphans": old_orphans, "orphan_bytes": sum(v[0] for v in inv["orphans"].values())}


# ---------- deleting ----------

def _unlink(path: Path, root: Path, sub: str, suffixes: tuple) -> int:
    """Delete one file if (and only if) it is a plain file of the expected type inside root/<sub>.
    Returns the bytes freed; raises ValueError for a path that must not be touched."""
    if not _inside(path, root, sub) or path.suffix not in suffixes or path.is_symlink() \
            or path.name.startswith("reports") or "index" in path.resolve().relative_to(root.resolve()).parts:
        raise ValueError(f"refusing to delete {path}")
    try:
        size = path.stat().st_size
        path.unlink()
        return size
    except FileNotFoundError:
        return 0


_SUFFIX = {"query": ("queries", (".ecsv",)), "gaia": ("gaia", (".ecsv",)), "record": ("fields", (".json",)),
           "wise": ("wise", (".json", ".png"))}


def _delete_group(step: dict, root: Path, cfg: Config, scan_time: float) -> tuple[int, int, int]:
    """Delete one field: cutouts first, the query last (a crash in between leaves a field the next run
    finishes, not files nobody owns). Returns (bytes freed, files deleted, files that failed)."""
    g: Group = step["group"]
    for ra, dec in list(_ACTIVE):                # somebody opened it after the scan
        if _ACTIVE.get((ra, dec), 0) > g.last_used and sep_deg(ra, dec, g.ra, g.dec) <= GROUP_DEG * 2:
            return 0, 0, 0
    for _, p, _, m in g.files:                   # ...or one of its files changed
        try:
            if p.stat().st_mtime > m + 1:
                return 0, 0, 0
        except OSError:
            pass
    freed = n = failed = 0

    def rm(path: Path, sub: str, suffixes: tuple) -> None:
        nonlocal freed, n, failed
        try:
            b = _unlink(path, root, sub, suffixes)
            freed += b
            n += 1
        except Exception as e:
            failed += 1
            print(f"[cleanup] kept {path.name}: {e}", flush=True)

    for name in step["cutouts"]:
        rm(root / "cutouts" / name, "cutouts", (".fits",))
    rank = {"wise": 0, "gaia": 1, "record": 2, "query": 3}
    for kind, p, _, _ in sorted(g.files, key=lambda f: rank[f[0]]):
        rm(p, *_SUFFIX[kind])
    for d in g.dirs:
        try:
            if _inside(d, root, "wise") and not d.is_symlink():
                os.rmdir(d)                      # only if empty: never removes anything we didn't list
        except OSError:
            pass
    return freed, n, failed


_RUN_LOCK = threading.Lock()
_STATE: dict = {"last_check": None, "last_run": None, "last_freed_bytes": 0, "last_deleted_fields": 0,
                "total_freed_bytes": 0, "deletable_bytes": None, "deletable_fields": None, "note": "not checked yet"}


def run(root: Path | None = None, cfg: Config | None = None, dry_run: bool = True, force: bool = False,
        protected: list | None = None, free_fn=None, now: float | None = None, seed: Path | None = SEED,
        require_protected: bool = True) -> dict:
    """One cleanup pass. dry_run=True (the default) only reports. Returns a summary dict; `refused` is set
    when nothing may be deleted (wrong folder, protected lists missing).

    protected=None -> protected_fields() (the lists main.py registered); force=True ignores the trigger
    threshold but still stops at the target. free_fn(root) -> free bytes (tests)."""
    cfg = cfg or Config.from_env()
    root = Path(root) if root is not None else _cache_root()
    free_fn = free_fn or (lambda r: shutil.disk_usage(r).free)
    out = {"dry_run": dry_run, "root": str(root), "refused": None, "triggered": False, "freed_bytes": 0,
           "deleted_fields": 0, "deleted_files": 0, "failed_files": 0, "would_delete": []}
    why = check_root(root)
    problems: list[str] = []
    if protected is None:
        protected, problems = protected_fields()
    if why is None and require_protected and not protected:
        problems.append("no protected fields are known")
    if why is None and problems:
        why = "; ".join(problems)
    if why is not None and (not dry_run or check_root(root) is not None):
        out["refused"] = why
        return out
    out["warning"] = why                          # dry run on a good folder: report anyway, say what is missing

    if not _RUN_LOCK.acquire(blocking=False):
        out["refused"] = "another cleanup is running"
        return out
    try:
        p = plan(root, cfg, protected, now=now, seed=seed)
        inv = p["inventory"]
        free = free_fn(root)
        out.update(free_before=free, by_status=p["by_status"], deletable_bytes=p["deletable_bytes"],
                   deletable_fields=len(p["steps"]), orphan_files=len(inv["orphans"]), orphan_bytes=p["orphan_bytes"],
                   unknown=inv["unknown"][:50], unknown_count=len(inv["unknown"]), plan=p)
        out["triggered"] = force or free < cfg.min_free
        if not out["triggered"]:
            out["free_after"] = free
            return out
        t_scan = time.time()
        for step in p["steps"]:
            if free >= cfg.target_free:
                break
            if dry_run:
                out["would_delete"].append((step["group"].ra, step["group"].dec, step["bytes"]))
                freed, n, failed = step["bytes"], len(step["cutouts"]) + len(step["group"].files), 0
                free += freed
            else:
                freed, n, failed = _delete_group(step, root, cfg, t_scan)
                free = free_fn(root)
            out["freed_bytes"] += freed
            out["deleted_files"] += n
            out["failed_files"] += failed
            out["deleted_fields"] += bool(n)
        if free < cfg.target_free and p["orphans_deletable"]:      # last resort, opt-in: files nobody refers to
            for name, (size, mtime) in sorted(p["old_orphans"].items(), key=lambda kv: kv[1][1]):
                if free >= cfg.target_free:
                    break
                if dry_run:
                    freed = size
                    free += size
                else:
                    try:
                        freed = _unlink(root / "cutouts" / name, root, "cutouts", (".fits",))
                    except Exception:
                        out["failed_files"] += 1
                        continue
                    free = free_fn(root)
                out["freed_bytes"] += freed
                out["deleted_files"] += 1
        out["free_after"] = free
        return out
    finally:
        _RUN_LOCK.release()


def _gb(b) -> str:
    return f"{(b or 0) / 1e9:.2f} GB"


def check() -> dict | None:
    """What the background job does every few minutes. Never raises."""
    try:
        cfg = Config.from_env()
        root = _cache_root()
        now = time.time()
        _STATE["last_check"] = round(now)
        if check_root(root) is not None:
            _STATE["note"] = check_root(root)
            return None
        free = shutil.disk_usage(root).free
        low = free < cfg.min_free
        if not low and _STATE["deletable_bytes"] is not None and now - _STATE.get("estimated_at", 0) < 3600:
            _STATE["note"] = "idle: enough free space"
            return None
        r = run(root, cfg, dry_run=not low)
        if r.get("refused"):
            _STATE["note"] = f"refused: {r['refused']}"
            if low:
                print(f"[cleanup] {_gb(free)} free but nothing deleted: {r['refused']}", flush=True)
            return r
        left = r["deletable_bytes"] - (r["freed_bytes"] if low else 0)
        _STATE.update(deletable_bytes=max(0, left), deletable_fields=r["deletable_fields"] - (r["deleted_fields"] if low else 0),
                      estimated_at=now, unknown_files=r["unknown_count"], orphan_bytes=r["orphan_bytes"])
        if low:
            _STATE.update(last_run=round(now), last_freed_bytes=r["freed_bytes"], last_deleted_fields=r["deleted_fields"],
                          total_freed_bytes=_STATE["total_freed_bytes"] + r["freed_bytes"],
                          note="freed space" if r["freed_bytes"] else "disk low, nothing deletable")
            print(f"[cleanup] {_gb(r['free_before'])} free (< {_gb(cfg.min_free)}): deleted {r['deleted_fields']} old "
                  f"fields, {r['deleted_files']} files, freed {_gb(r['freed_bytes'])}; now {_gb(r['free_after'])} free, "
                  f"{_gb(left)} more deletable, {r['failed_files']} files failed", flush=True)
        else:
            _STATE["note"] = "idle: enough free space"
        return r
    except Exception as e:
        _STATE["note"] = f"error: {e!r}"
        try:
            print(f"[cleanup] failed: {e!r}", flush=True)
        except Exception:
            pass
        return None


def status() -> dict:
    """For /api/health."""
    cfg = Config.from_env()
    s = {k: v for k, v in _STATE.items() if k != "estimated_at"}
    s.update(enabled=enabled(), running=_THREAD is not None and _THREAD.is_alive(),
             min_free_gb=round(cfg.min_free / 1e9, 2), target_free_gb=round(cfg.target_free / 1e9, 2))
    return s


_THREAD: threading.Thread | None = None


def start(first_delay_s: float = 90.0) -> bool:
    """Start the background job (once). Does nothing when the job is switched off."""
    global _THREAD
    if not enabled():
        _STATE["note"] = "off (SKYWATCH_CLEANUP=1 turns it on)"
        return False
    if _THREAD is not None and _THREAD.is_alive():
        return True

    def loop():
        time.sleep(first_delay_s)
        while True:
            check()
            try:
                time.sleep(Config.from_env().every_s)
            except Exception:
                time.sleep(600)

    _THREAD = threading.Thread(target=loop, name="cleanup", daemon=True)
    _THREAD.start()
    return True


# ---------- command line ----------

def _main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Clean old visitor fields from the cache (dry run by default).")
    ap.add_argument("--cache", help="cache folder (default: the server's, SKYWATCH_CACHE or ./cache)")
    ap.add_argument("--dry-run", action="store_true", help="only report (the default)")
    ap.add_argument("--delete", action="store_true", help="really delete")
    ap.add_argument("--force", action="store_true", help="act even when free space is above the threshold")
    ap.add_argument("--list", action="store_true", help="print every field")
    a = ap.parse_args(argv)
    if a.delete and a.dry_run:
        ap.error("--delete and --dry-run together")
    try:                                         # the site's own lists of fields (importing main starts nothing)
        import main as _site                     # noqa: F401  registers the protected lists
    except Exception as e:
        print(f"could not load main.py ({e!r}): the protected lists are unknown")
    root = Path(a.cache) if a.cache else _cache_root()
    r = run(root, dry_run=not a.delete, force=a.force)
    if r["refused"]:
        print(f"REFUSED: {r['refused']}")
        return 2
    cfg = Config.from_env()
    print(f"cache: {r['root']}   free: {_gb(r['free_before'])}   trigger below {_gb(cfg.min_free)}, "
          f"target {_gb(cfg.target_free)}")
    if r.get("warning"):
        print(f"WARNING (a real run would be refused): {r['warning']}")
    for st in ("protected", "recent", "fresh", "deletable"):
        s = r["by_status"].get(st, {"fields": 0, "bytes": 0})
        print(f"  {st:<10} {s['fields']:>4} fields  {s['bytes'] / 1e6:>9.1f} MB")
    print(f"  cutouts nobody refers to: {r['orphan_files']} files, {r['orphan_bytes'] / 1e6:.1f} MB "
          f"({'deletable when old' if r['plan']['orphans_deletable'] else 'left alone'})")
    print(f"  not recognised, left alone: {r['unknown_count']}")
    for name, size in r["unknown"]:
        print(f"      {name}  ({size / 1e6:.1f} MB)")
    if a.list:
        for g in sorted(r["plan"]["inventory"]["groups"], key=lambda g: (g.status, g.last_used)):
            print(f"  {g.status:<10} {g.ra:9.4f} {g.dec:+9.4f}  {g.bytes / 1e6:7.1f} MB  {len(g.cutouts):>4} cutouts  "
                  f"last used {time.strftime('%Y-%m-%d %H:%M', time.localtime(g.last_used))}")
    verb = "deleted" if a.delete else "would delete"
    if not r["triggered"]:
        print(f"nothing to do: free space is above the threshold ({_gb(r['deletable_bytes'])} in "
              f"{r['deletable_fields']} fields could be freed when needed)")
    else:
        print(f"{verb} {r['deleted_fields']} fields, {r['deleted_files']} files, {_gb(r['freed_bytes'])}; "
              f"free afterwards {_gb(r['free_after'])}")
    return 0


if __name__ == "__main__":
    import cleanup as _self          # the same module object main.py registers its lists with
    sys.exit(_self._main(sys.argv[1:]))
