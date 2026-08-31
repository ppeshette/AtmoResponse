"""The local data directory for downloaded scenes, LUT archives, and reference data.

These files persist between runs and are reused, not scratch. Nothing deletes
them automatically. Set ``ATMORESPONSE_DATA`` to place the directory where you
want it, and remove it yourself to reclaim the space.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATA_DIR = Path.home() / "atmoresponse_data"


def default_data_dir() -> Path:
    """The data directory, from ``ATMORESPONSE_DATA`` or the default under home."""

    configured = os.environ.get("ATMORESPONSE_DATA")
    return Path(configured).expanduser() if configured else DEFAULT_DATA_DIR


def resolve_data_dir(data_dir: str | Path | None) -> Path:
    """Normalize a caller-supplied data directory, or fall back to the default."""

    return Path(data_dir).expanduser() if data_dir is not None else default_data_dir()
