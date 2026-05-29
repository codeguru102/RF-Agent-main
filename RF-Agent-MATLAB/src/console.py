from __future__ import annotations

import itertools
import sys
import threading
import time
from typing import Optional


class Spinner:
    def __init__(self, message: str, *, enabled: bool = True, interval: float = 0.1):
        self.message = message
        self.enabled = enabled
        self.interval = interval
        self._frames = itertools.cycle("|/-\\")
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._active = False

    def __enter__(self) -> "Spinner":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is not None:
            self.fail()
        else:
            self.stop(clear=True)

    def start(self) -> None:
        if self._active:
            return
        self._active = True
        if not self.enabled:
            return
        if not sys.stdout.isatty():
            print(f"[..] {self.message}", flush=True)
            return
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def update(self, message: str) -> None:
        with self._lock:
            self.message = message
        if not self.enabled:
            return
        if not sys.stdout.isatty():
            print(f"[..] {message}", flush=True)

    def succeed(self, message: Optional[str] = None) -> None:
        self.stop(clear=True)
        print_success(message or self.message)

    def fail(self, message: Optional[str] = None) -> None:
        self.stop(clear=True)
        print_error(message or self.message)

    def stop(self, *, clear: bool = False) -> None:
        if not self._active:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._active = False
        if clear and self.enabled and sys.stdout.isatty():
            sys.stdout.write("\r" + " " * (len(self.message) + 8) + "\r")
            sys.stdout.flush()

    def _spin(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                message = self.message
            sys.stdout.write(f"\r[{next(self._frames)}] {message}")
            sys.stdout.flush()
            time.sleep(self.interval)


def print_info(message: str) -> None:
    print(f"[info] {message}")


def print_success(message: str) -> None:
    print(f"[ok] {message}")


def print_warning(message: str) -> None:
    print(f"[warn] {message}")


def print_error(message: str) -> None:
    print(f"[fail] {message}")
