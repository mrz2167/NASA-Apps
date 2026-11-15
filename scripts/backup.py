#!/usr/bin/env python3
# AI disclosure: parts of this file were written with AI assistance (Claude, Anthropic), then reviewed and tested by the LUMEN team. See README, "AI use".
"""Скачать резервную копию с живого сайта: отчёты (reports.db) и кэш (cache.tar).

    BACKUP_KEY=... python3 scripts/backup.py              # отчёты + кэш в backups/ГГГГ-ММ-ДД/
    BACKUP_KEY=... python3 scripts/backup.py reports      # только отчёты (секунды)
    BACKUP_KEY=... python3 scripts/backup.py cache        # только кэш (~1 ГБ, минуты)

Переменные: BACKUP_KEY (обязательно), SKYWATCH_URL (по умолчанию живой сайт), BACKUP_DIR (по умолчанию backups/),
KEEP_REPORTS (сколько дней хранить отчёты, 60), KEEP_CACHE (сколько копий кэша, 3).

Каждая копия проверяется: база — PRAGMA integrity_check, tar — читается целиком. Битая копия не сохраняется,
а скрипт завершается с ошибкой (код 1), чтобы это было видно в cron/launchd.
Восстановление: reports.db -> положить на диск сервера в cache/reports.db; cache.tar -> `tar xf cache.tar`
в папке приложения (получится cache/…), на Railway: `railway volume files upload`.
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import time
from pathlib import Path

URL = os.environ.get("SKYWATCH_URL", "https://nasa-skywatch-production.up.railway.app").rstrip("/")
KEY = os.environ.get("BACKUP_KEY", "")
ROOT = Path(os.environ.get("BACKUP_DIR", Path(__file__).resolve().parent.parent / "backups"))
KEEP_REPORTS = int(os.environ.get("KEEP_REPORTS", "60"))
KEEP_CACHE = int(os.environ.get("KEEP_CACHE", "3"))


def fetch(path: str, dest: Path, max_time: int, check) -> object:
    """curl -> временный файл -> проверка check(tmp) -> только тогда заменить dest.

    Неудачная попытка (сеть, неверный ключ, битый файл) не трогает уже сохранённую копию.
    curl стримит большие файлы на диск и сам разбирается с сертификатами macOS.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    cmd = ["curl", "-fsS", "--retry", "3", "--retry-delay", "10", "--max-time", str(max_time),
           "-o", str(tmp), "-G", f"{URL}{path}", "--data-urlencode", f"key={KEY}"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{path}: {r.stderr.strip() or 'curl exit ' + str(r.returncode)}")
    try:
        info = check(tmp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dest)
    return info


def check_db(p: Path) -> int:
    with sqlite3.connect(p) as con:
        ok = con.execute("PRAGMA integrity_check").fetchone()[0]
        if ok != "ok":
            raise RuntimeError(f"reports.db integrity_check: {ok}")
        return con.execute("SELECT COUNT(*) FROM finds").fetchone()[0]


def check_tar(p: Path) -> tuple[int, int]:
    n = size = 0
    with tarfile.open(p, "r:") as t:
        for m in t:
            n += 1
            size += m.size
    if n == 0:
        raise RuntimeError("cache.tar is empty")
    return n, size


def prune(pattern: str, keep: int) -> None:
    files = sorted(ROOT.glob(pattern))
    for f in files[:-keep] if keep > 0 else []:
        f.unlink(missing_ok=True)
        if f.parent != ROOT and not any(f.parent.iterdir()):
            f.parent.rmdir()


def main() -> int:
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what not in ("all", "reports", "cache"):
        print(__doc__)
        return 2
    if not KEY:
        print("Set BACKUP_KEY (the server's BACKUP_KEY, or REPORTS_KEY if BACKUP_KEY is not set).", file=sys.stderr)
        return 2
    day = ROOT / time.strftime("%Y-%m-%d")
    day.mkdir(parents=True, exist_ok=True)
    failed = False

    if what in ("all", "reports"):
        dest = day / "reports.db"
        try:
            n = fetch("/api/admin/backup/reports.db", dest, 300, check_db)
            print(f"reports.db: {n} reports, {dest.stat().st_size / 1e3:.0f} KB -> {dest}")
        except Exception as e:
            print(f"reports.db FAILED: {e}", file=sys.stderr)
            failed = True
        prune("*/reports.db", KEEP_REPORTS)

    if what in ("all", "cache"):
        dest = day / "cache.tar"
        free = shutil.disk_usage(ROOT).free
        if free < 5e9:
            print(f"cache.tar skipped: only {free / 1e9:.1f} GB free here", file=sys.stderr)
            failed = True
        else:
            try:
                n, size = fetch("/api/admin/backup/cache.tar", dest, 3600, check_tar)
                print(f"cache.tar: {n} files, {size / 1e9:.2f} GB -> {dest}")
            except Exception as e:
                print(f"cache.tar FAILED: {e}", file=sys.stderr)
                failed = True
            prune("*/cache.tar", KEEP_CACHE)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
