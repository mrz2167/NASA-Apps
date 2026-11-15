#!/usr/bin/env python3
# AI disclosure: parts of this file were written with AI assistance (Claude, Anthropic), then reviewed and tested by the LUMEN team. See README, "AI use".
"""Pre-show check: is the live site ready for the demo? Prints READY in green or a list of what's wrong.

    python scripts/preflight.py                      # production
    python scripts/preflight.py http://localhost:8000
    python scripts/preflight.py --quick              # skip the lens maps (slowest part)

Read-only: GET requests only, nothing is written on the server. Standard library only, so it runs on any
laptop with Python 3.9+. Exit code 0 = ready, 1 = something to fix, 2 = ready with warnings.

What it checks:
  * /api/health: ok, warm-up finished (state demo_ready or done), free disk, IRSA / AWS-index state, WISE
  * the page, the training levels (training.json + level 1 and level 5 frames) and the Russian translation
  * timing of the demo calls: /api/frames, /api/object and /api/lens on the demo fields, and one WISE frames
    call. A warm field answers in about a second; a slow answer means the field is not cached and the
    audience would wait.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

PROD = "https://nasa-skywatch-production.up.railway.app"

# (name, ra, dec): demo cases plus one famous object; the same fields main.py warms up on start
FIELDS = [
    ("WISE 0855-0714", 133.75977, -7.24218),
    ("CRBR 2422.8-3423", 246.8525, -24.6842),
    ("Epsilon Indi B", 331.09639, -56.80057),
    ("Orion Nebula (M42)", 83.8221, -5.3911),
]
WISE_FIELD = ("WISE 0855-0714 (WISE frames)", 133.75977, -7.24218)

# seconds: under WARN is "flies"; under FAIL is usable but slow (warning); above FAIL is a problem
LIMITS = {"frames": (5, 30), "object": (6, 20), "lens": (8, 30), "wise": (10, 45), "static": (3, 15)}
MIN_FREE_GB = 1.0

try:
    sys.stdout.reconfigure(errors="replace")      # a Windows console in cp1251 can't print ✓ and ✗
except AttributeError:
    pass
try:
    "✓✗".encode(sys.stdout.encoding or "ascii")
    MARKS = {"ok": "✓", "warn": "!", "fail": "✗"}
except (UnicodeEncodeError, LookupError):
    MARKS = {"ok": "+", "warn": "!", "fail": "x"}
TTY = sys.stdout.isatty() or os.environ.get("FORCE_COLOR")
if os.name == "nt" and TTY:
    os.system("")                                 # turns on ANSI colours in the Windows console
C = {"g": "\033[32m", "y": "\033[33m", "r": "\033[31m", "b": "\033[1m", "0": "\033[0m"} if TTY else \
    dict.fromkeys("gyrb0", "")


def fetch(base: str, path: str, params: dict | None = None, timeout: float = 120) -> tuple[int, bytes, float]:
    """GET base+path -> (HTTP status or 0 on a network error, body, seconds)."""
    url = base + path + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, headers={"User-Agent": "lumen-preflight/1"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), time.time() - t
    except urllib.error.HTTPError as e:
        return e.code, e.read(), time.time() - t
    except Exception as e:                        # timeout, DNS, TLS, connection refused
        return 0, str(e).encode(), time.time() - t


def detail(body: bytes) -> str:
    try:
        d = json.loads(body).get("detail")
        return str(d)[:140] if d else ""
    except Exception:
        return body[:140].decode(errors="replace")


class Report:
    def __init__(self):
        self.problems: list[str] = []
        self.warnings: list[str] = []
        self.ctx = ""                                 # field name, so the summary says where it broke

    def line(self, mark: str, text: str):
        colour = {"ok": "g", "warn": "y", "fail": "r"}[mark]
        sign = MARKS[mark]
        print(f"  {C[colour]}{sign}{C['0']} {text}", flush=True)

    def ok(self, text):
        self.line("ok", text)

    def warn(self, text):
        self.warnings.append(self.ctx + text)
        self.line("warn", text)

    def fail(self, text):
        self.problems.append(self.ctx + text)
        self.line("fail", text)

    def timed(self, kind: str, what: str, secs: float, extra: str = ""):
        warn, fail = LIMITS[kind]
        msg = f"{what}: {secs:.1f} s{extra}"
        if secs <= warn:
            self.ok(msg)
        elif secs <= fail:
            self.warn(msg + f" (slow: over {warn} s, the field may not be cached)")
        else:
            self.fail(msg + f" (too slow: over {fail} s)")


def check_health(base: str, rep: Report) -> None:
    print(f"{C['b']}Server{C['0']}")
    code, body, secs = fetch(base, "/api/health", timeout=30)
    if code == 0:
        rep.fail(f"/api/health: no answer ({body.decode(errors='replace')[:120]})")
        return
    try:
        h = json.loads(body)
    except ValueError:
        rep.fail(f"/api/health: HTTP {code}, not JSON")
        return
    if code == 200 and h.get("ok"):
        rep.ok(f"/api/health ok ({secs:.1f} s), {h.get('cached_fields', '?')} fields in memory")
    else:
        rep.fail(f"/api/health: HTTP {code}" + (f": {'; '.join(h.get('problems') or [])}" if h.get("problems") else ""))

    w = h.get("warm") or {}
    state = w.get("state")
    if state == "done":
        rep.ok(f"warm-up done: {w.get('demo_done')}/{w.get('demo_total')} demo fields, "
               f"{w.get('surprise_ready', '?')} 'Surprise me' fields ready")
    elif state == "demo_ready":
        rep.ok(f"demo fields ready ({w.get('demo_done')}/{w.get('demo_total')}); 'Surprise me' still loading")
    elif state == "running":
        rep.fail(f"warm-up still running: {w.get('demo_done', 0)}/{w.get('demo_total', '?')} demo fields "
                 "(just deployed? wait a minute and run again)")
    else:
        rep.warn(f"warm-up state: {state!r}")

    disk = h.get("disk") or {}
    free = disk.get("free_gb")
    if free is None:
        rep.fail("disk: not readable")
    elif free < MIN_FREE_GB:
        rep.fail(f"disk: only {free} GB free")
    else:
        rep.ok(f"disk: {free} GB free")

    a = h.get("archive") or {}
    if a.get("irsa") == "ok":
        rep.ok("IRSA archive answering")
    elif a.get("aws_index"):
        rep.warn("IRSA archive down: new fields come from our own AWS index (demo fields are cached)")
    elif a:
        rep.fail("IRSA archive down and no AWS index: only cached fields will open")
    wise = h.get("wise") or {}
    if wise and not wise.get("ok"):
        rep.warn(f"WISE frames (WiseView) unavailable: {wise.get('last_error') or 'down'}")


def check_static(base: str, rep: Report) -> None:
    print(f"{C['b']}Page and files{C['0']}")
    code, body, secs = fetch(base, "/", timeout=30)
    if code == 200 and b"LUMEN" in body:
        rep.timed("static", "home page", secs)
    else:
        rep.fail(f"home page: HTTP {code or 'no answer'}")

    code, body, secs = fetch(base, "/levels/training.json", timeout=30)
    try:
        levels = json.loads(body) if code == 200 else None
    except ValueError:
        levels = None
    if not isinstance(levels, dict) or not levels:
        rep.fail(f"training levels: /levels/training.json HTTP {code or 'no answer'}")
    else:
        missing = [k for k in ("1", "5") if k not in levels]
        (rep.fail if missing else rep.ok)(f"training.json: levels {', '.join(sorted(levels))}"
                                          + (f" (missing {', '.join(missing)})" if missing else ""))
    for png in ("l1_0.png", "l1_1.png", "l5_0.png", "l5_1.png"):
        code, body, secs = fetch(base, "/levels/" + png, timeout=30)
        if code == 200 and body[:8] == b"\x89PNG\r\n\x1a\n":
            rep.ok(f"level frame {png} ({len(body) // 1024} KB)")
        else:
            rep.fail(f"level frame {png}: HTTP {code or 'no answer'}")

    code, body, _ = fetch(base, "/i18n/ru.json", timeout=30)
    try:
        n = len(json.loads(body)) if code == 200 else 0
    except ValueError:
        n = 0
    (rep.ok if n else rep.fail)(f"Russian translation: {n} strings" if n else f"/i18n/ru.json: HTTP {code or 'no answer'}")


def check_field(base: str, rep: Report, name: str, ra: float, dec: float, lens: bool) -> None:
    print(f"{C['b']}{name}{C['0']}  ({ra}, {dec})")
    rep.ctx = name + ": "
    pos = {"ra": ra, "dec": dec, "fov": 0.1}
    code, body, secs = fetch(base, "/api/frames", {**pos, "detector": 4, "mode": "blink", "cadence": "pass"},
                             timeout=240)
    if code != 200:
        rep.fail(f"frames: HTTP {code or 'no answer'} after {secs:.0f} s {detail(body)}".rstrip())
        return                                    # object and lens use the same field: no point timing them
    try:
        n = len(json.loads(body).get("frames") or [])
    except ValueError:
        n = 0
    if n < 2:
        rep.fail(f"frames: only {n} frame(s), nothing to blink")
    else:
        rep.timed("frames", "frames", secs, f", {n} frames")

    code, body, secs = fetch(base, "/api/object", {"ra": ra, "dec": dec}, timeout=120)
    if code != 200:
        rep.fail(f"object card: HTTP {code or 'no answer'} after {secs:.0f} s {detail(body)}".rstrip())
    else:
        try:
            o = json.loads(body)
        except ValueError:
            o = {}
        who = o.get("name") or o.get("catalog") or o.get("status") or "?"
        cat = o.get("catalog_status")
        rep.timed("object", "object card", secs, f" ({who}" + (f", catalogue {cat}" if cat and cat != "ok" else "") + ")")

    if lens:
        code, body, secs = fetch(base, "/api/lens", pos, timeout=180)
        if code != 200:
            rep.fail(f"molecule lens: HTTP {code or 'no answer'} after {secs:.0f} s {detail(body)}".rstrip())
        else:
            rep.timed("lens", "molecule lens", secs)


def check_wise(base: str, rep: Report) -> None:
    name, ra, dec = WISE_FIELD
    print(f"{C['b']}{name}{C['0']}")
    rep.ctx = "WISE: "
    code, body, secs = fetch(base, "/api/frames", {"ra": ra, "dec": dec, "fov": 0.1, "source": "wise"}, timeout=120)
    if code != 200:
        # WISE frames come from a third-party service: the SPHEREx demo still works without them
        rep.warn(f"WISE frames: HTTP {code or 'no answer'} after {secs:.0f} s {detail(body)}".rstrip())
        return
    try:
        n = len(json.loads(body).get("frames") or [])
    except ValueError:
        n = 0
    rep.timed("wise", "WISE frames", secs, f", {n} frames")


def main() -> int:
    ap = argparse.ArgumentParser(description="Is the LUMEN site ready for the demo?")
    ap.add_argument("base", nargs="?", default=os.environ.get("SKYWATCH_URL", PROD), help=f"site URL (default {PROD})")
    ap.add_argument("--quick", action="store_true", help="skip the molecule lens maps")
    args = ap.parse_args()
    base = args.base.rstrip("/")
    t0 = time.time()
    print(f"Pre-show check of {base}\n")

    rep = Report()
    check_health(base, rep)
    check_static(base, rep)
    for name, ra, dec in FIELDS:
        check_field(base, rep, name, ra, dec, lens=not args.quick)
    check_wise(base, rep)

    print(f"\nChecked in {time.time() - t0:.0f} s.")
    if rep.problems:
        print(f"{C['r']}{C['b']}NOT READY{C['0']}: {len(rep.problems)} problem(s)")
        for p in rep.problems:
            print(f"  {C['r']}{MARKS['fail']}{C['0']} {p}")
        for w in rep.warnings:
            print(f"  {C['y']}{MARKS['warn']}{C['0']} {w}")
        return 1
    if rep.warnings:
        print(f"{C['g']}{C['b']}READY{C['0']} with {len(rep.warnings)} warning(s):")
        for w in rep.warnings:
            print(f"  {C['y']}{MARKS['warn']}{C['0']} {w}")
        return 2
    print(f"{C['g']}{C['b']}READY{C['0']}: everything answers, the demo fields are warm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
