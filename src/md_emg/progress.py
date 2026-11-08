from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

from rich.console import Console


class RunReporter:
    """Small console + file reporter safe for the main process.

    GridSearchCV worker processes are intentionally quiet; the reporter logs at
    the experiment/unit level so parallel output remains readable.
    """

    def __init__(self, log_path: Path, verbose: bool = True):
        self.console = Console()
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self._lock = threading.Lock()

    def _write(self, level: str, message: str, *, style: str | None = None):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {level:<7} {message}"
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            if self.verbose:
                if style:
                    self.console.print(f"[{style}]{line}[/{style}]")
                else:
                    self.console.print(line)

    def info(self, message: str):
        self._write("INFO", message)

    def step(self, message: str):
        self._write("RUN", message, style="cyan")

    def success(self, message: str):
        self._write("DONE", message, style="green")

    def warning(self, message: str):
        self._write("WARN", message, style="yellow")
