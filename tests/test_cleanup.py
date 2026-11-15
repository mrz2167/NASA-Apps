# AI disclosure: parts of this file were written with AI assistance (Claude, Anthropic), then reviewed and tested by the LUMEN team. See README, "AI use".
"""cleanup.py on a throw-away cache folder (no network, no real cache): what may be deleted and what never is."""
from __future__ import annotations

import json
import os
import sys
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cleanup  # noqa: E402
from data import cutout_path  # noqa: E402

HOUR, DAY = 3600.0, 86400.0
PROTECTED = [(10.0, 10.0)]          # the "demo field" of these tests


def _url(obs: str, ra: float, dec: float) -> str:
    return (f"https://irsa.ipac.caltech.edu/ibe/data/spherex/dr1/level2/2025W17_4B/l2b-v27/3/"
            f"level2_{obs}_spx_l2b-v27.fits?center={ra},{dec}d&size=0.1")


def _age(path: Path, age_s: float) -> None:
    t = time.time() - age_s
    os.utime(path, (t, t))


def make_field(root: Path, ra: float, dec: float, age_s: float, n: int = 4, size: int = 1000, query: bool = True,
               gaia: bool = True, wise: str | None = None, record: bool = False, urls: list | None = None) -> dict:
    """A fake field: n cutouts + query (+ Gaia table, WISE folder, "last opened" record), all age_s old."""
    for d in ("cutouts", "queries", "gaia", "wise", "fields"):
        (root / d).mkdir(parents=True, exist_ok=True)
    urls = urls or [_url(f"2025W17_4B_{int(ra * 100) % 10000:04d}_{i}D3", ra, dec) for i in range(n)]
    out = {"cutouts": [], "other": []}
    for u in urls:
        p = root / "cutouts" / cutout_path(u).name
        p.write_bytes(b"x" * size)
        _age(p, age_s)
        out["cutouts"].append(p)
    tag = f"{ra:.4f}_{dec:+.4f}"
    if query:
        q = root / "queries" / f"{abs(hash(tag)) % 16 ** 12:012x}.ecsv"
        q.write_text("# %ECSV 1.0\n# ---\n# datatype:\n# - {name: url, datatype: string}\n"
                     "# - {name: obs_id, datatype: string}\n# meta: !!omap\n"
                     f"# - {{ra: {ra}}}\n# - {{dec: {dec}}}\n# - {{size_deg: 0.1}}\n# schema: astropy-2.0\n"
                     "url obs_id\n" + "".join(f"{u} obs{i}\n" for i, u in enumerate(urls)), encoding="utf-8")
        out["other"].append(q)
    if gaia:
        g = root / "gaia" / f"{tag}_0.1.ecsv"
        g.write_text("# %ECSV 1.0\nra dec\n", encoding="utf-8")
        out["other"].append(g)
    if wise:
        w = root / "wise" / wise
        w.mkdir()
        (w / "meta.json").write_text(json.dumps({"CRVAL1": ra, "CRVAL2": dec + 0.5, "CRPIX1": 66.0,
                                                  "CRPIX2": 66.0 + 0.5 * 3600 / 2.75, "NAXIS1": 131, "NAXIS2": 131,
                                                  "ims": ["a", "b"]}))
        for i in range(2):
            (w / f"{i:02d}.png").write_bytes(b"p" * 50)
        out["other"] += [w / "meta.json", w / "00.png", w / "01.png"]
        out["wise_dir"] = w
    if record:
        r = root / "fields" / f"{tag}_0.1.json"
        r.write_text(json.dumps({"ra": ra, "dec": dec, "size_deg": 0.1, "cutouts": [p.name for p in out["cutouts"]]}))
        out["other"].append(r)
    for p in out["other"]:
        _age(p, age_s)
    out["all"] = out["cutouts"] + out["other"]
    return out


def tree_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def fake_disk(root: Path, free_at_start: float):
    """free_fn: a disk that had free_at_start bytes free and gains whatever gets deleted under root."""
    start = tree_bytes(root)
    return lambda r: free_at_start + (start - tree_bytes(root))


def exists(f: dict) -> bool:
    return all(p.exists() for p in f["all"])


def gone(f: dict) -> bool:
    return not any(p.exists() for p in f["all"])


@pytest.fixture
def cache(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    (root / "cutouts").mkdir(parents=True)
    monkeypatch.setattr(cleanup, "_ACTIVE", {})
    monkeypatch.setattr(cleanup, "_NOTED", {})
    monkeypatch.setattr(cleanup, "_cache_root", lambda: root)
    return root


def cfg(**kw) -> cleanup.Config:
    c = cleanup.Config(min_free=1e6, target_free=1e6 + 1)       # "low on space" below 1 MB in these tests
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def go(root: Path, free: float, c: cleanup.Config | None = None, **kw) -> dict:
    kw.setdefault("protected", PROTECTED)
    kw.setdefault("seed", None)
    kw.setdefault("dry_run", False)
    return cleanup.run(root, c or cfg(), free_fn=fake_disk(root, free), **kw)


# ---------- trigger ----------

def test_nothing_happens_above_threshold(cache):
    old = make_field(cache, 50, 5, 30 * DAY, wise="aaaaaaaaaaaa", record=True)
    before = tree_bytes(cache)
    r = go(cache, free=2e6)
    assert r["triggered"] is False and r["freed_bytes"] == 0 and r["deleted_fields"] == 0
    assert exists(old) and tree_bytes(cache) == before
    assert r["deletable_fields"] == 1 and r["deletable_bytes"] > 0          # the estimate for /api/health


def test_dry_run_deletes_nothing(cache):
    old = make_field(cache, 50, 5, 30 * DAY)
    before = sorted(str(p) for p in cache.rglob("*"))
    r = go(cache, free=0, dry_run=True)
    assert r["triggered"] and r["deleted_fields"] == 1 and r["freed_bytes"] > 0 and r["would_delete"]
    assert exists(old) and sorted(str(p) for p in cache.rglob("*")) == before


def test_stops_at_target_least_recently_used_first(cache):
    a = make_field(cache, 50, 5, 30 * DAY)        # 4 x 1000 bytes of cutouts each
    b = make_field(cache, 60, 5, 10 * DAY)
    c = make_field(cache, 70, 5, 20 * DAY)
    r = go(cache, free=0, c=cfg(min_free=1, target_free=6000))               # one field is not enough, two are
    assert gone(a) and gone(c) and exists(b)
    assert r["deleted_fields"] == 2 and r["free_after"] >= 6000


def test_deletes_everything_deletable_when_target_unreachable(cache):
    a = make_field(cache, 50, 5, 30 * DAY, wise="aaaaaaaaaaaa", record=True)
    b = make_field(cache, 60, 5, 10 * DAY)
    r = go(cache, free=0)
    assert gone(a) and gone(b) and not a["wise_dir"].exists()
    assert r["deleted_fields"] == 2 and r["failed_files"] == 0 and r["freed_bytes"] > 8000


def test_last_opened_record_beats_file_age(cache):
    a = make_field(cache, 50, 5, 30 * DAY, record=True)
    b = make_field(cache, 60, 5, 20 * DAY)
    _age(a["other"][-1], 3 * DAY)                 # a was opened 3 days ago: now b is the least recently used
    go(cache, free=0, c=cfg(min_free=1, target_free=3000))
    assert gone(b) and exists(a)


# ---------- what survives ----------

def test_protected_field_survives_even_when_oldest(cache):
    demo = make_field(cache, 10.0, 10.0, 300 * DAY, wise="bbbbbbbbbbbb", record=True)
    near = make_field(cache, 10.015, 10.015, 200 * DAY)       # the same object opened by slightly other coordinates
    visitor = make_field(cache, 50, 5, 2 * DAY)
    r = go(cache, free=0)
    assert exists(demo) and exists(near) and gone(visitor)
    assert r["by_status"]["protected"]["fields"] == 2


def test_seed_fields_are_protected(cache, tmp_path):
    f = make_field(cache, 80, -30, 300 * DAY)
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "x.ecsv").write_text(f["other"][0].read_text())
    visitor = make_field(cache, 50, 5, 2 * DAY)
    go(cache, free=0, seed=seed)
    assert exists(f) and gone(visitor)


def test_cutouts_of_seed_field_without_cache_query_are_kept(cache, tmp_path):
    f = make_field(cache, 80, -30, 300 * DAY, gaia=False)     # built from seed/queries: no query in the cache
    seed = tmp_path / "seed"
    seed.mkdir()
    f["other"][0].rename(seed / "x.ecsv")
    go(cache, free=0, seed=seed, c=cfg(orphans=True))
    assert all(p.exists() for p in f["cutouts"])


def test_shared_cutouts_survive(cache):
    shared = [_url(f"2025W17_4B_0001_{i}D3", 50, 5) for i in range(3)]
    old = make_field(cache, 50, 5, 30 * DAY, urls=shared + [_url("2025W17_4B_0002_1D3", 50, 5)])
    kept = make_field(cache, 50.5, 5, 30 * DAY, urls=shared)
    for p in kept["other"]:
        _age(p, HOUR * 2)                         # opened two hours ago
    go(cache, free=0)
    assert exists(kept)                           # including the three files the deleted field also listed
    assert not old["cutouts"][3].exists() and not any(p.exists() for p in old["other"])


def test_recent_fields_survive(cache):
    used = make_field(cache, 50, 5, 30 * DAY, record=True)
    _age(used["other"][-1], 5 * HOUR)             # record touched 5 h ago
    building = make_field(cache, 60, 5, 30 * DAY)
    _age(building["cutouts"][0], 600)             # a file written 10 minutes ago
    old = make_field(cache, 70, 5, 30 * DAY)
    r = go(cache, free=0)
    assert exists(used) and exists(building) and gone(old)
    assert r["by_status"]["recent"]["fields"] == 2


def test_fresh_file_blocks_even_with_no_keep_window(cache):
    building = make_field(cache, 60, 5, 30 * DAY)
    _age(building["cutouts"][0], 600)
    r = go(cache, free=0, c=cfg(keep_s=0))
    assert exists(building) and r["by_status"]["fresh"]["fields"] == 1


def test_field_being_opened_right_now_survives(cache):
    old = make_field(cache, 50, 5, 30 * DAY)
    cleanup.note_active(50.0001, 5.0001)          # a request for it has just started (nothing on disk changes)
    go(cache, free=0)
    assert exists(old)


def test_field_opened_between_scan_and_delete_survives(cache):
    old = make_field(cache, 50, 5, 30 * DAY)
    p = cleanup.plan(cache, cfg(), PROTECTED, seed=None)
    assert len(p["steps"]) == 1
    cleanup.note_active(50, 5)
    assert cleanup._delete_group(p["steps"][0], cache, cfg(), time.time()) == (0, 0, 0)
    assert exists(old)


def test_reports_index_skies_untouched(cache):
    (cache / "reports.db").write_bytes(b"sqlite" * 100)
    (cache / "reports.db-wal").write_bytes(b"wal")
    for d, name in (("index", "spherex_index.npz"), ("index/parts", "dr1_x.npz"), ("skies", "50_5_0p1_pass_None_abc.pkl")):
        (cache / d).mkdir(parents=True, exist_ok=True)
        (cache / d / name).write_bytes(b"i" * 5000)
        _age(cache / d / name, 400 * DAY)
    _age(cache / "reports.db", 400 * DAY)
    old = make_field(cache, 50, 5, 30 * DAY)
    r = go(cache, free=0, c=cfg(orphans=True))
    assert gone(old)
    assert (cache / "reports.db").read_bytes() == b"sqlite" * 100 and (cache / "reports.db-wal").exists()
    assert (cache / "index" / "spherex_index.npz").exists() and (cache / "index" / "parts" / "dr1_x.npz").exists()
    assert (cache / "skies" / "50_5_0p1_pass_None_abc.pkl").exists()
    assert r["unknown_count"] == 0                # known things, not "unrecognised"


def test_unknown_files_untouched_and_reported(cache):
    old = make_field(cache, 50, 5, 30 * DAY)
    strange = [cache / "cutouts" / "half.tmp", cache / "cutouts" / "notes.txt", cache / "queries" / "broken.ecsv",
               cache / "gaia" / "something.ecsv", cache / "fields" / "bad.json", cache / "stray.bin",
               cache / "newfolder" / "data.bin", cache / "wise" / "cccccccccccc" / "meta.json",
               cache / "wise" / "cccccccccccc" / "extra.fits", cache / "wise" / "nometa000000" / "00.png"]
    for p in strange:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"?" * 10)
        _age(p, 100 * DAY)
    orphan = cache / "cutouts" / "2025W17_4B_9999_1D3_0123456789ab.fits"      # a cutout no query lists
    orphan.write_bytes(b"o" * 10)
    _age(orphan, 100 * DAY)
    r = go(cache, free=0)
    assert gone(old) and all(p.exists() for p in strange) and orphan.exists()
    names = {n.replace("\\", "/") for n, _ in r["unknown"]}
    assert {"cutouts/half.tmp", "queries/broken.ecsv", "stray.bin", "newfolder", "wise/cccccccccccc",
            "wise/nometa000000", "fields/bad.json", "gaia/something.ecsv"} <= names
    assert r["orphan_files"] == 1


def test_orphan_cutouts_only_when_switched_on(cache):
    make_field(cache, 50, 5, 30 * DAY)
    old_orphan = cache / "cutouts" / "2025W17_4B_9999_1D3_0123456789ab.fits"
    new_orphan = cache / "cutouts" / "2025W17_4B_9999_2D3_0123456789ac.fits"
    for p, age in ((old_orphan, 100 * DAY), (new_orphan, 2 * HOUR)):
        p.write_bytes(b"o" * 10)
        _age(p, age)
    go(cache, free=0, c=cfg(orphans=True))
    assert not old_orphan.exists() and new_orphan.exists()
    (cache / "queries" / "broken.ecsv").write_text("???")      # an unreadable query: its cutouts look like orphans
    old_orphan.write_bytes(b"o" * 10)
    _age(old_orphan, 100 * DAY)
    go(cache, free=0, c=cfg(orphans=True))
    assert old_orphan.exists()


# ---------- safety rails ----------

def test_paths_outside_cache_are_refused(cache, tmp_path):
    outside = tmp_path / "outside.fits"
    outside.write_bytes(b"keep")
    (cache / "index").mkdir()
    (cache / "index" / "a.fits").write_bytes(b"keep")
    (cache / "reports.db").write_bytes(b"keep")
    (cache / "cutouts" / "sub").mkdir()
    (cache / "cutouts" / "sub" / "deep.fits").write_bytes(b"keep")
    (cache / "cutouts" / "x.txt").write_bytes(b"keep")
    bad = [(outside, "cutouts"), (cache / "cutouts" / ".." / ".." / "outside.fits", "cutouts"),
           (cache / "index" / "a.fits", "index"), (cache / "index" / "a.fits", "cutouts"),
           (cache / "cutouts" / ".." / "index" / "a.fits", "cutouts"), (cache / "reports.db", "cutouts"),
           (cache / "cutouts" / ".." / "reports.db", "cutouts"), (cache / "cutouts" / "sub" / "deep.fits", "cutouts"),
           (cache / "cutouts" / "x.txt", "cutouts"), (cache / "cutouts", "cutouts")]
    for path, sub in bad:
        with pytest.raises(ValueError):
            cleanup._unlink(path, cache, sub, (".fits",))
    assert outside.read_bytes() == b"keep" and (cache / "index" / "a.fits").exists()
    assert (cache / "reports.db").exists() and (cache / "cutouts" / "sub" / "deep.fits").exists()
    ok = cache / "cutouts" / "a.fits"
    ok.write_bytes(b"12345")
    assert cleanup._unlink(ok, cache, "cutouts", (".fits",)) == 5 and not ok.exists()


def test_record_cannot_point_outside_cutouts(cache):
    (cache / "reports.db").write_bytes(b"keep")
    (cache / "index").mkdir()
    (cache / "index" / "spherex_index.npz").write_bytes(b"keep")
    (cache / "fields").mkdir()
    rec = cache / "fields" / "50.0000_+5.0000_0.1.json"
    rec.write_text(json.dumps({"ra": 50, "dec": 5, "size_deg": 0.1,
                               "cutouts": ["../reports.db", "..\\index\\spherex_index.npz", "/etc/passwd", "reports.db"]}))
    _age(rec, 30 * DAY)
    r = go(cache, free=0)
    assert not rec.exists() and r["failed_files"] == 0
    assert (cache / "reports.db").exists() and (cache / "index" / "spherex_index.npz").exists()


def test_symlinks_are_not_followed(cache, tmp_path):
    target = tmp_path / "precious.fits"
    target.write_bytes(b"keep")
    link = cache / "cutouts" / "2025W17_4B_9999_1D3_0123456789ab.fits"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available here")
    go(cache, free=0, c=cfg(orphans=True, keep_s=0, fresh_s=0))
    assert target.read_bytes() == b"keep" and link.is_symlink()


def test_refuses_wrong_folder(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "cutouts" in cleanup.run(empty, cfg(), dry_run=False, protected=PROTECTED)["refused"]
    assert cleanup.run(tmp_path / "missing", cfg(), dry_run=False, protected=PROTECTED)["refused"]
    src = tmp_path / "src"
    (src / "cutouts").mkdir(parents=True)
    (src / "main.py").write_text("")
    assert "source tree" in cleanup.run(src, cfg(), dry_run=False, protected=PROTECTED)["refused"]
    assert "source tree" in cleanup.run(src, cfg(), dry_run=True, protected=PROTECTED)["refused"]


def test_refuses_without_protected_lists(cache, monkeypatch):
    old = make_field(cache, 50, 5, 30 * DAY)
    assert "no protected" in go(cache, free=0, protected=[])["refused"]
    monkeypatch.setattr(cleanup, "_PROVIDERS", [])
    cleanup.protect("warm", [("Demo", 10.0, 10.0)], required=True)
    cleanup.protect("surprise", lambda: [], required=True)          # e.g. index.html changed and the list is gone
    r = go(cache, free=0, protected=None)
    assert "surprise" in r["refused"] and exists(old)
    cleanup.protect("surprise", lambda: 1 / 0, required=True)
    assert "surprise" in go(cache, free=0, protected=None)["refused"] and exists(old)
    cleanup.protect("surprise", lambda: [{"name": "S", "ra": 20.0, "dec": -20.0}], required=True)
    assert go(cache, free=0, protected=None)["refused"] is None and gone(old)


def test_protected_lists_are_collected(monkeypatch):
    mod = types.ModuleType("fake_main")
    mod.WARM_FIELDS = [("Barnard's Star", 269.4463, 4.76783)]
    mod.FAMOUS = [("Crab", 83.63, 22.01), ("Integer", 200, 20)]            # any UPPERCASE (name, ra, dec) list
    mod.DIFF_POS = [1.0, 2.0, 3.0]
    mod.lower = [("no", 1.0, 2.0)]
    mod.NOT_SKY = [("x", 500.0, 2.0)]
    assert sorted(f[0] for f in cleanup.fields_in_module(mod)) == ["Barnard's Star", "Crab", "Integer"]
    monkeypatch.setattr(cleanup, "_PROVIDERS", [])
    monkeypatch.setattr(cleanup, "PINNED_FIELDS", [("Pinned", 1.0, 2.0)])
    cleanup.protect("lists", mod, required=True)
    mod.WARM_FIELDS.append(("Later", 5.0, 6.0))                            # added after registration
    pos, problems = cleanup.protected_fields()
    assert not problems and {(1.0, 2.0), (269.4463, 4.76783), (83.63, 22.01), (5.0, 6.0)} <= set(pos)


def test_main_registers_its_lists():
    src = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "cleanup.protect(" in src and "cleanup.track(_load_sky)" in src and "cleanup.start()" in src


# ---------- "last opened" records ----------

class _Coord:
    def __init__(self, ra, dec):
        self.ra = types.SimpleNamespace(deg=ra)
        self.dec = types.SimpleNamespace(deg=dec)


def test_note_used_writes_record_and_never_raises(cache):
    urls = [_url(f"2025W17_4B_0001_{i}D3", 50, 5) for i in range(3)]
    sky = {"coord": _Coord(50.0, 5.0), "size_deg": 0.1, "cutouts": [{"url": u} for u in urls]}
    cleanup.note_used(sky)
    rec = cache / "fields" / "50.0000_+5.0000_0.1.json"
    data = json.loads(rec.read_text())
    assert data["ra"] == 50.0 and sorted(data["cutouts"]) == sorted(cutout_path(u).name for u in urls)
    assert not list((cache / "fields").glob("*.tmp*"))
    _age(rec, 5 * DAY)
    cleanup.note_used(sky)                         # within 10 minutes: no disk write at all
    assert time.time() - rec.stat().st_mtime > 4 * DAY
    cleanup._NOTED.clear()
    cleanup.note_used(sky)                         # later: the record is touched
    assert time.time() - rec.stat().st_mtime < 60
    cleanup._NOTED.clear()
    cleanup.note_used({"coord": _Coord(50.0, 5.0), "size_deg": 0.1, "cutouts": [{"url": _url("2025W17_4B_0009_1D3", 50, 5)}]})
    assert len(json.loads(rec.read_text())["cutouts"]) == 4     # a fuller build adds files, never drops them
    for junk in (None, {}, {"coord": None}, {"coord": _Coord("x", 1), "size_deg": 0.1}, 42):
        cleanup.note_used(junk)


def test_record_makes_index_built_field_deletable_as_a_whole(cache):
    """A field built from the AWS index has no query file: only the record knows its cutouts."""
    f = make_field(cache, 50, 5, 30 * DAY, query=False, gaia=False, record=True)
    assert cleanup.plan(cache, cfg(), PROTECTED, seed=None)["inventory"]["orphans"] == {}
    go(cache, free=0)
    assert gone(f)


def test_track_wraps_loader(cache):
    calls = []

    def load(ra, dec, fov, cadence="pass"):
        calls.append((ra, dec, fov, cadence))
        if ra < 0:
            raise RuntimeError("archive down")
        return {"coord": _Coord(ra, dec), "size_deg": fov, "cutouts": []}

    wrapped = cleanup.track(load)
    assert wrapped(50.0, 5.0, 0.1, cadence="visit")["size_deg"] == 0.1 and calls == [(50.0, 5.0, 0.1, "visit")]
    assert (cache / "fields" / "50.0000_+5.0000_0.1.json").exists() and (50.0, 5.0) in cleanup._ACTIVE
    with pytest.raises(RuntimeError):
        wrapped(-1.0, 5.0, 0.1)


# ---------- reading queries, background job, status ----------

def test_wise_centre_is_the_cutout_not_the_tile():
    ra, dec = cleanup.wise_centre({"CRVAL1": 268.86057, "CRVAL2": 4.543333, "CRPIX1": 830.5, "CRPIX2": -228.5,
                                   "NAXIS1": 131, "NAXIS2": 131})        # the real answer for Barnard's Star
    assert cleanup.sep_deg(ra, dec, 269.4463, 4.76783) < 0.01
    with pytest.raises(Exception):
        cleanup.wise_centre({"CRVAL1": 1.0, "CRVAL2": 2.0})


def test_query_reader_matches_astropy(tmp_path):
    from astropy.table import Table
    urls = [_url(f"2025W17_4B_0001_{i}D3", 126.4325, -51.01) for i in range(5)]
    t = Table(rows=[[u, f"o{i}", 60000.5 + i] for i, u in enumerate(urls)], names=["url", "obs_id", "time"])
    t.meta.update(ra=126.4325, dec=-51.01, size_deg=0.1)
    p = tmp_path / "q.ecsv"
    t.write(p)
    q = cleanup.read_query(p)
    assert (q["ra"], q["dec"], q["size_deg"]) == (126.4325, -51.01, 0.1)
    assert q["cutouts"] == {cutout_path(u).name for u in urls}
    (tmp_path / "bad.ecsv").write_text("not a table")
    assert cleanup.read_query(tmp_path / "bad.ecsv") is None


def test_seed_queries_in_git_are_readable():
    seeds = sorted((ROOT / "seed" / "queries").glob("*.ecsv"))
    assert seeds and all(cleanup.read_query(p) and cleanup.read_query(p)["cutouts"] for p in seeds)


def test_enabled_switch(monkeypatch):
    for k in [k for k in os.environ if k.startswith("RAILWAY_")]:
        monkeypatch.delenv(k)
    monkeypatch.delenv("SKYWATCH_CLEANUP", raising=False)
    assert cleanup.enabled() is False              # a developer's machine: off unless asked
    monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
    assert cleanup.enabled() is True
    monkeypatch.setenv("SKYWATCH_CLEANUP", "0")
    assert cleanup.enabled() is False and cleanup.start() is False
    monkeypatch.delenv("RAILWAY_ENVIRONMENT")
    monkeypatch.setenv("SKYWATCH_CLEANUP", "1")
    assert cleanup.enabled() is True


def test_background_check_never_raises_and_reports(cache, monkeypatch):
    monkeypatch.setattr(cleanup, "_PROVIDERS", [])
    monkeypatch.setattr(cleanup, "_STATE", dict(cleanup._STATE, deletable_bytes=None))
    cleanup.protect("warm", [("Demo", 10.0, 10.0)], required=True)
    old = make_field(cache, 50, 5, 30 * DAY)
    monkeypatch.setenv("SKYWATCH_CLEANUP_MIN_FREE_GB", "0")                 # never "low" on this machine
    cleanup.check()
    s = cleanup.status()
    assert exists(old) and s["deletable_fields"] == 1 and s["deletable_bytes"] > 4000 and s["last_run"] is None
    json.dumps(s)                                                           # goes into /api/health as is
    monkeypatch.setenv("SKYWATCH_CLEANUP_MIN_FREE_GB", "1e9")               # always "low": the real thing
    monkeypatch.setenv("SKYWATCH_CLEANUP_TARGET_FREE_GB", "2e9")
    cleanup.check()
    s = cleanup.status()
    assert gone(old) and s["last_freed_bytes"] > 4000 and s["last_deleted_fields"] == 1 and s["last_run"]
    monkeypatch.setattr(cleanup, "plan", lambda *a, **k: 1 / 0)
    assert cleanup.check() is None and "error" in cleanup.status()["note"]
