# AI disclosure: parts of this file were written with AI assistance (Claude, Anthropic), then reviewed and tested by the LUMEN team. See README, "AI use".
"""Резервные копии с сервера: отчёты пользователей и кэш SPHEREx/WISE с постоянного диска Railway.

    GET /api/admin/backup?key=KEY              что есть: размеры, число файлов, число отчётов
    GET /api/admin/backup/reports.db?key=KEY   согласованная копия базы отчётов (SQLite backup API)
    GET /api/admin/backup/cache.tar?key=KEY    весь кэш одним tar-потоком (без reports.db — она отдельно)

Ключ — BACKUP_KEY (если не задан, REPORTS_KEY). Без ключа на сервере резервные копии выключены.
Скачивает их scripts/backup.py (вручную или по расписанию) и .github/workflows/backup.yml (отчёты, раз в день).
"""
from __future__ import annotations

import hmac
import os
import sqlite3
import tarfile
import tempfile
import threading
import time
from contextlib import closing

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask

from data import CACHE
import reports

router = APIRouter()


def _key() -> str:
    return os.environ.get("BACKUP_KEY") or os.environ.get("REPORTS_KEY", "")


def _require(key: str | None) -> None:
    k = _key()
    if not k:
        raise HTTPException(403, "Backups are off: set BACKUP_KEY on the server.")
    if not key or not hmac.compare_digest(key, k):
        raise HTTPException(401, "Wrong or missing key.")


def _cache_files():
    db = reports.DB_PATH.resolve()
    for p in sorted(CACHE.rglob("*")):
        if p.is_file() and not p.name.endswith((".tmp", "-journal", "-wal", "-shm")) and p.resolve() != db:
            yield p


@router.get("/api/admin/backup")
def backup_info(key: str | None = None):
    _require(key)
    files = list(_cache_files())
    n_reports = None
    if reports.DB_PATH.exists():
        try:
            with closing(sqlite3.connect(reports.DB_PATH)) as con:
                n_reports = con.execute("SELECT COUNT(*) FROM finds").fetchone()[0]
        except sqlite3.Error:
            n_reports = None
    return {"reports_db": str(reports.DB_PATH), "reports": n_reports,
            "reports_db_bytes": reports.DB_PATH.stat().st_size if reports.DB_PATH.exists() else 0,
            "cache_dir": str(CACHE), "cache_files": len(files), "cache_bytes": sum(p.stat().st_size for p in files)}


@router.get("/api/admin/backup/reports.db")
def backup_reports(key: str | None = None):
    """Копия через SQLite backup API: целая и согласованная, даже если в этот момент приходит отчёт."""
    _require(key)
    with closing(reports._db()):          # создаёт пустую базу, если отчётов ещё не было
        pass
    fd, tmp = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    with closing(sqlite3.connect(reports.DB_PATH)) as src, closing(sqlite3.connect(tmp)) as dst:
        src.backup(dst)
    name = time.strftime("reports-%Y%m%d-%H%M%S.db", time.gmtime())
    return FileResponse(tmp, media_type="application/vnd.sqlite3", filename=name,
                        background=BackgroundTask(os.remove, tmp))


@router.get("/api/admin/backup/cache.tar")
def backup_cache(key: str | None = None):
    """Весь кэш потоком tar (без сжатия: FITS и PNG почти не сжимаются, а CPU на сервере один)."""
    _require(key)
    rfd, wfd = os.pipe()

    def writer():
        with os.fdopen(wfd, "wb") as w:
            try:
                with tarfile.open(fileobj=w, mode="w|") as tar:
                    for p in _cache_files():
                        try:
                            tar.add(p, arcname=str(p.relative_to(CACHE.parent)))
                        except (FileNotFoundError, PermissionError):
                            continue          # файл удалили/переписали во время копии — пропускаем
            except BrokenPipeError:
                pass                          # клиент прервал скачивание

    threading.Thread(target=writer, name="backup-tar", daemon=True).start()

    def chunks():
        with os.fdopen(rfd, "rb") as r:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                yield b

    name = time.strftime("cache-%Y%m%d-%H%M%S.tar", time.gmtime())
    return StreamingResponse(chunks(), media_type="application/x-tar",
                             headers={"Content-Disposition": f'attachment; filename="{name}"'})
