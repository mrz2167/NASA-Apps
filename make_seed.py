"""Поиск IRSA для всех полей сайта -> seed/queries/ (в git): сервер собирает их и без IRSA.

    python make_seed.py            # все поля: кейсы и «Try» (main.WARM_FIELDS) + «Surprise me» (web/index.html)
    python make_seed.py --check    # только показать, каких файлов не хватает

Нужен работающий IRSA TAP. Поля 0.1° и 0.2° (выбор «Field of view» на сайте; 0.05° вырезается из 0.1°).
"""
from __future__ import annotations

import os
import shutil
import sys
import time

os.environ.setdefault("SKYWATCH_WARM", "0")
import main                                     # noqa: E402  списки полей ровно те, что у сервера
from data import CACHE, SEED, _key, find_images  # noqa: E402

SIZES = (0.1, 0.2)


def fields():
    seen, out = set(), []
    for name, ra, dec in list(main.WARM_FIELDS) + main._surprise_fields():
        k = (round(ra, 4), round(dec, 4))
        if k not in seen:
            seen.add(k)
            out.append((name, ra, dec))
    return out


def seed_name(ra, dec, size):
    return f"{_key(round(ra, 6), round(dec, 6), size)}.ecsv"


def main_() -> int:
    check = "--check" in sys.argv
    SEED.mkdir(parents=True, exist_ok=True)
    todo = [(n, ra, dec, s) for n, ra, dec in fields() for s in SIZES if not (SEED / seed_name(ra, dec, s)).exists()]
    print(f"{len(fields())} fields × {len(SIZES)} sizes; missing {len(todo)}")
    if check:
        for n, ra, dec, s in todo:
            print(f"  missing: {n} {s}°")
        return 1 if todo else 0
    failed = 0
    for n, ra, dec, s in todo:
        t = time.time()
        try:
            tab = find_images(ra, dec, s, use_cache=False)      # всегда свежий ответ IRSA
            src = CACHE / "queries" / seed_name(ra, dec, s)
            shutil.copy(src, SEED / src.name)
            print(f"  {n} {s}°: {len(tab)} images, {time.time() - t:.0f} s")
        except Exception as e:
            failed += 1
            print(f"  {n} {s}°: FAILED {str(e)[:120]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main_())
