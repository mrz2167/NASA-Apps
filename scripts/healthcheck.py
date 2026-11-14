#!/usr/bin/env python3
"""Проверка живого сайта с сообщением в Telegram: раз в 5 минут из cron/launchd на любом включённом компьютере.

    TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... python3 scripts/healthcheck.py

Проверяет: /api/health (200 и ok=true; 503 значит сломана база отчётов или кончается диск), главную страницу
и /reports. Пишет в Telegram только когда состояние меняется: «упал» и «снова работает», плюс напоминание,
если сбой длится дольше часа. Состояние — в файле STATE_FILE (по умолчанию ~/.lumen-healthcheck.json).
WISE (сторонний WiseView) только предупреждение: сообщение одно, без тревоги.

Без TELEGRAM_* просто печатает результат (код выхода 1 при сбое) — удобно для проверки руками.
cron (каждые 5 минут):  */5 * * * * cd /путь/к/NASA-Apps && TELEGRAM_BOT_TOKEN=… TELEGRAM_CHAT_ID=… python3 scripts/healthcheck.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

URL = os.environ.get("SKYWATCH_URL", "https://nasa-skywatch-production.up.railway.app").rstrip("/")
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")
STATE = Path(os.environ.get("STATE_FILE", Path.home() / ".lumen-healthcheck.json"))
REMIND_S = 3600


def get(path: str) -> tuple[int, str]:
    """curl (сертификаты macOS работают из коробки): (HTTP-код или 0 при сетевой ошибке, тело)."""
    r = subprocess.run(["curl", "-sS", "--max-time", "30", "--retry", "2", "--retry-delay", "5",
                        "-w", "\n%{http_code}", URL + path], capture_output=True, text=True)
    if r.returncode != 0:
        return 0, r.stderr.strip()
    body, _, code = r.stdout.rpartition("\n")
    return int(code or 0), body


def check() -> tuple[list[str], list[str]]:
    problems, warnings = [], []
    code, body = get("/api/health")
    if code == 0:
        problems.append(f"/api/health: no answer ({body[:120]})")
    else:
        try:
            h = json.loads(body)
        except ValueError:
            h = {}
        if code != 200 or not h.get("ok"):
            problems.append(f"/api/health: HTTP {code}" + (f" — {'; '.join(h.get('problems', []))}" if h.get("problems") else ""))
        if h.get("wise") and not h["wise"].get("ok"):
            warnings.append("WISE frames unavailable (WiseView is down); SPHEREx still works")
    for path in ("/", "/reports"):
        code, _ = get(path)
        if code != 200:
            problems.append(f"{path}: HTTP {code or 'no answer'}")
    return problems, warnings


def telegram(text: str) -> None:
    if not (TOKEN and CHAT):
        return
    subprocess.run(["curl", "-sS", "--max-time", "20", f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                    "-d", f"chat_id={CHAT}", "-d", f"text={quote(text, safe='')}"],
                   capture_output=True)


def main() -> int:
    try:
        st = json.loads(STATE.read_text())
    except (OSError, ValueError):
        st = {"down_since": None, "last_alert": 0, "wise_warned": False}
    problems, warnings = check()
    now = time.time()
    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(now))

    if problems:
        if st["down_since"] is None:
            st["down_since"] = now
            telegram(f"🔴 LUMEN is DOWN ({stamp})\n{URL}\n" + "\n".join(problems))
            st["last_alert"] = now
        elif now - st["last_alert"] > REMIND_S:
            mins = int((now - st["down_since"]) / 60)
            telegram(f"🔴 LUMEN still down for {mins} min\n" + "\n".join(problems))
            st["last_alert"] = now
    elif st["down_since"] is not None:
        mins = int((now - st["down_since"]) / 60)
        telegram(f"🟢 LUMEN is back after {mins} min ({stamp})")
        st["down_since"] = None

    if warnings and not st.get("wise_warned"):
        telegram("🟡 " + "\n".join(warnings))
        st["wise_warned"] = True
    elif not warnings and st.get("wise_warned"):
        telegram("🟢 WISE frames work again")
        st["wise_warned"] = False

    STATE.write_text(json.dumps(st))
    print(f"{stamp} " + ("DOWN: " + "; ".join(problems) if problems else "ok") + (f" | {'; '.join(warnings)}" if warnings else ""))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
