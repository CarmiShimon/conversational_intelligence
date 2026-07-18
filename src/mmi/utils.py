"""Small shared helpers: logging, timing, JSON caching, CLI bootstrap."""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


_LOG = get_logger("mmi.utils")

# Wall-clock seconds per stage name, populated by stage_timer(). Used to report
# latency/resource notes in the final pipeline metadata (see schemas.py).
STAGE_TIMINGS: dict[str, float] = {}


def reset_stage_timings() -> None:
    """Clear STAGE_TIMINGS. Call at the start of each pipeline run.

    Without this, timings accumulate across every run_pipeline() call made
    in the same process (e.g. batch scripts, tests, notebooks), so a later
    run's reported stage_seconds would silently include earlier runs' time.
    """
    STAGE_TIMINGS.clear()


@contextmanager
def stage_timer(name: str) -> Iterator[None]:
    """Log wall-clock time for a pipeline stage and record it in STAGE_TIMINGS."""
    _LOG.info("[%s] started", name)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        STAGE_TIMINGS[name] = STAGE_TIMINGS.get(name, 0.0) + dt
        _LOG.info("[%s] finished in %.1fs", name, dt)


def write_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    return path


def read_json(path: str | Path) -> Optional[Any]:
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def configure_utf8_stdio() -> None:
    """Reconfigure stdout/stderr to UTF-8.

    Some Windows consoles use a legacy codepage (e.g. cp1255) that cannot
    encode the Unicode progress bars printed by EasyOCR / gdown / tqdm, which
    would otherwise crash the run. Shared by both CLI entry points
    (``mmi.run`` and ``mmi.evaluate``).
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


def load_dotenv_if_available() -> None:
    """Load a ``.env`` file into the environment, if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        pass
