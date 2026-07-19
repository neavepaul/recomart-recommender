"""Small dependency-free progress display for long-running pipeline work."""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from typing import Iterator


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


class Progress:
    """A tqdm-like counter that works in terminals and captured logs."""

    def __init__(self, label: str, total: int | None = None,
                 unit: str = "rows", min_interval: float = 2.0):
        self.label = label
        self.total = total
        self.unit = unit
        self.min_interval = min_interval
        self.started = self.last_print = time.monotonic()
        self.count = 0
        self._tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
        self._printed = False
        self._render(force=True)

    def update(self, count: int, force: bool = False) -> None:
        self.count = count
        self._render(force)

    def advance(self, amount: int = 1) -> None:
        self.update(self.count + amount)

    def _render(self, force: bool = False, complete: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_print < self.min_interval:
            return
        elapsed = max(now - self.started, 0.001)
        rate = self.count / elapsed
        if self.total:
            fraction = min(self.count / self.total, 1.0)
            filled = round(fraction * 20)
            bar = "#" * filled + "-" * (20 - filled)
            amount = f"[{bar}] {fraction:6.1%} | {self.count:,}/{self.total:,} {self.unit}"
        else:
            spinner = "|/-\\"[int(now * 5) % 4]
            amount = f"{spinner} {self.count:,} {self.unit}"
        suffix = f" | {rate:,.0f} {self.unit}/s | {_duration(elapsed)}"
        if complete:
            suffix += " | done"
        line = f"{self.label}: {amount}{suffix}"
        if self._tty:
            print("\r" + line.ljust(100), end="\n" if complete else "", file=sys.stderr, flush=True)
        else:
            print(line, file=sys.stderr, flush=True)
        self.last_print = now
        self._printed = True

    def close(self, count: int | None = None) -> None:
        if count is not None:
            self.count = count
        self._render(force=True, complete=True)


@contextmanager
def sqlite_activity(db, label: str, callback_steps: int = 100_000) -> Iterator[None]:
    """Show liveness while SQLite performs a long statement or index build."""
    progress = Progress(label, unit="SQLite steps", min_interval=3.0)

    def tick() -> int:
        progress.advance(callback_steps)
        return 0

    db.set_progress_handler(tick, callback_steps)
    try:
        yield
    finally:
        db.set_progress_handler(None, 0)
        progress.close()

