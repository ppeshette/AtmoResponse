"""Aerosol optical depth reference infrastructure."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Sequence

import numpy as np

from .cache import CacheConfig

CO_LOCATED_KM = 25.0
REGIONAL_KM = 100.0
KM_PER_MINUTE = 25.0 / 30.0


class AodSource(str, Enum):
    """Supported AOD reference source names."""

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

        return float((self.distance_km**2 + (self.dt_minutes * KM_PER_MINUTE) ** 2) ** 0.5)

    @property
    def tier(self) -> str:
        """Coarse representativeness label from the combined separation."""

        if self.separation_km <= CO_LOCATED_KM:
            return "co-located"
        if self.separation_km <= REGIONAL_KM:
            return "regional"
        return "distant"

    @property
    def outward_caveat(self) -> str | None:
        """Return the public caveat required for assimilated references."""

        if self.source is AodSource.MERRA2:
            return (
                "MERRA-2 assimilates observing networks and should be described as a "
                "reanalysis reference, not an independent measurement."
            )
        return None


@dataclass(frozen=True)
class AodSummary:
    """Representative AOD from a scene product or selected pixel group."""

    value: float
    statistic: str
    count: int
    mean: float
    std: float
    minimum: float
    maximum: float
    detail: str


AodProvider = Callable[[AodQuery, CacheConfig | None], AodEstimate | None]


def expected_error(reference_value: float) -> float:
    """Conventional satellite AOD expected-error envelope."""

    return 0.05 + 0.15 * reference_value


def agrees(retrieved_aod: float, reference: AodEstimate) -> bool:
    """Return whether a retrieved AOD is inside the reference expected-error envelope."""

    return abs(retrieved_aod - reference.value) <= expected_error(reference.value) + 1e-12


def summarize_aod(
    values,
    valid_mask=None,
    *,
    detail: str = "",
) -> AodSummary:
    """Summarize finite AOD values with a median representative value."""

    array = np.asarray(values, dtype="f8")
    valid = np.isfinite(array)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != array.shape:
            raise ValueError(f"valid_mask shape {mask.shape} does not match values shape {array.shape}")
        valid &= mask

    selected = array[valid]
    if selected.size == 0:
        raise ValueError("no valid AOD values to summarize")

    return AodSummary(
        value=float(np.median(selected)),
        statistic="median",
        count=int(selected.size),
        mean=float(np.mean(selected)),
        std=float(np.std(selected)),
        minimum=float(np.min(selected)),
        maximum=float(np.max(selected)),
        detail=detail,
    )


def gather_aod(
    query: AodQuery,
    providers: Mapping[AodSource, AodProvider],
    sources: Sequence[AodSource] = (
        AodSource.AERONET,
        AodSource.GOES,
        AodSource.VIIRS,
        AodSource.MERRA2,
    ),
    cache: CacheConfig | None = None,
) -> list[AodEstimate]:
    """Return every AOD reference that resolves, preserving requested source order."""

    estimates: list[AodEstimate] = []
    for source in sources:
        provider = providers.get(source)
        if provider is None:
            continue
        estimate = provider(query, cache)
        if estimate is not None:
            estimates.append(estimate)
    return estimates


def best_aod(estimates: Sequence[AodEstimate]) -> AodEstimate | None:
    """Pick the AOD reference a scene-level check should use."""

    if not estimates:
        return None

    aeronet = [estimate for estimate in estimates if estimate.source is AodSource.AERONET]
    if aeronet:
        return min(aeronet, key=lambda estimate: estimate.separation_km)

    return min(estimates, key=lambda estimate: estimate.separation_km)


def resolve_aod(
    query: AodQuery,
    sources: tuple[AodSource, ...] = (
        AodSource.AERONET,
        AodSource.GOES,
        AodSource.VIIRS,
        AodSource.MERRA2,
    ),
    cache: CacheConfig | None = None,
    providers: Mapping[AodSource, AodProvider] | None = None,
) -> AodEstimate | None:
    """Resolve the best available external AOD550 estimate."""

    if providers is None:
        raise NotImplementedError("AOD resolution needs configured source providers.")

    return best_aod(gather_aod(query, providers=providers, sources=sources, cache=cache))
