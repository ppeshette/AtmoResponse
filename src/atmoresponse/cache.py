"""Cache configuration for public data access."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CacheConfig:
    """Local cache locations used by data-access code."""

    root: Path

    @classmethod
    def default(cls) -> "CacheConfig":
        """Build a cache config from the environment or the user profile."""

        configured = os.environ.get("ATMORESPONSE_CACHE")
        if configured:
            return cls(Path(configured).expanduser())
        return cls(Path.home() / ".cache" / "atmoresponse")

    def child(self, *parts: str) -> Path:
        """Return a path inside the configured cache root."""

        return self.root.joinpath(*parts)

