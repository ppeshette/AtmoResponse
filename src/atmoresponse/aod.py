"""External aerosol optical depth references."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import Enum

from .cache import CacheConfig


class AodSource(str, Enum):
    """Supported external AOD source names."""

    AERONET = "aeronet"
    GOES = "goes"
    VIIRS = "viirs"
    MERRA2 = "merra2"


@dataclass(frozen=True)
class AodQuery:
    """Location and time for an AOD550 lookup."""

    latitude: float
    longitude: float
    when: dt.datetime
    max_distance_km: float | None = None
    max_dt_minutes: float | None = None


@dataclass(frozen=True)
class AodEstimate:
    """One external AOD550 estimate with provenance."""

    value: float
    source: AodSource
    independence: str
    distance_km: float
    dt_minutes: float
    detail: str

    @property
    def separation_km(self) -> float:
        """Space and time separation combined as an equivalent distance."""

        km_per_minute = 25.0 / 30.0
        return float((self.distance_km**2 + (self.dt_minutes * km_per_minute) ** 2) ** 0.5)

    @property
    def outward_caveat(self) -> str | None:
        """Return the public caveat required for assimilated references."""

        if self.source is AodSource.MERRA2:
            return (
                "MERRA-2 assimilates observing networks and should be described as a "
                "reanalysis reference, not an independent measurement."
            )
        return None


def resolve_aod(
    query: AodQuery,
    sources: tuple[AodSource, ...] = (
        AodSource.AERONET,
        AodSource.GOES,
        AodSource.VIIRS,
        AodSource.MERRA2,
    ),
    cache: CacheConfig | None = None,
) -> AodEstimate | None:
    """Resolve the best available external AOD550 estimate."""

    _ = (query, sources, cache)
    raise NotImplementedError("External AOD resolution has not been ported into AtmoResponse yet.")

