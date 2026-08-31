"""AERONET AOD provider for AtmoResponse."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import requests

from .aod import AodEstimate, AodQuery, AodSource, REGIONAL_KM
from .geo import haversine_km

SITE_LIST_URL = "https://aeronet.gsfc.nasa.gov/aeronet_locations_v3.txt"
DATA_URL = "https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3"
DEFAULT_MAX_DT_MINUTES = 180.0


@dataclass(frozen=True)
class AeronetSite:
    """One AERONET site location."""

    name: str
    longitude: float
    latitude: float


def parse_site_list(text: str) -> list[AeronetSite]:
    """Parse AERONET's public location list."""

    sites: list[AeronetSite] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            longitude = float(parts[1])
            latitude = float(parts[2])
        except ValueError:
            continue
        sites.append(AeronetSite(parts[0], longitude=longitude, latitude=latitude))
    return sites


def fetch_site_list(session: requests.Session | None = None) -> list[AeronetSite]:
    """Fetch AERONET V3 site locations."""

    session = session or requests.Session()
    response = session.get(SITE_LIST_URL, timeout=30)
    response.raise_for_status()
    return parse_site_list(response.text)


def query_aod_rows(
    site: str,
    start: dt.date,
    end: dt.date,
    *,
    level: str = "AOD15",
    avg: int = 10,
    session: requests.Session | None = None,
) -> list[dict[str, str]]:
    """Query AERONET AOD rows for one site over an inclusive date range."""

    session = session or requests.Session()
    params = {
        "site": site,
        "year": start.year,
        "month": start.month,
        "day": start.day,
        "year2": end.year,
        "month2": end.month,
        "day2": end.day,
        level: 1,
        "AVG": avg,
        "if_no_html": 1,
    }
    response = session.get(DATA_URL, params=params, timeout=30)
    response.raise_for_status()
    return parse_aod_rows(response.text)


def parse_aod_rows(text: str) -> list[dict[str, str]]:
    """Parse rows returned by AERONET's ``print_web_data_v3`` endpoint."""

    lines = [line for line in text.splitlines() if line]
    for index, line in enumerate(lines):
        if "Date(dd:mm:yyyy)" not in line or "Time(hh:mm:ss)" not in line:
            continue
        header = line.split(",")
        return [dict(zip(header, row.split(","))) for row in lines[index + 1 :]]
    return []


def row_datetime(row: Mapping[str, str]) -> dt.datetime:
    """Return the observation timestamp from one parsed AERONET AOD row."""

    day, month, year = row["Date(dd:mm:yyyy)"].split(":")
    hour, minute, second = row["Time(hh:mm:ss)"].split(":")
    return dt.datetime(
        int(year),
        int(month),
        int(day),
        int(hour),
        int(minute),
        int(second),
    )


def nearest_in_time(
    rows: Sequence[Mapping[str, str]],
    target: dt.datetime,
) -> tuple[Mapping[str, str], float]:
    """Return the row nearest ``target`` and its absolute separation in seconds."""

    if not rows:
        raise ValueError("no AERONET rows to match")

    target = target.replace(tzinfo=None)
    best = min(rows, key=lambda row: abs((row_datetime(row) - target).total_seconds()))
    return best, abs((row_datetime(best) - target).total_seconds())


def available_aod_bands(row: Mapping[str, str]) -> list[tuple[float, float]]:
    """Return available non-fill AOD bands as ``(wavelength_nm, value)`` pairs."""

    bands: list[tuple[float, float]] = []
    for key, value in row.items():
        if not key.startswith("AOD_") or not key.endswith("nm"):
            continue
        try:
            wavelength = float(key[len("AOD_") : -len("nm")])
            aod = float(value)
        except ValueError:
            continue
        if aod > -900:
            bands.append((wavelength, aod))
    return sorted(bands)


def aod_at(row: Mapping[str, str], wavelength_nm: float = 550.0) -> float | None:
    """Interpolate AERONET AOD to ``wavelength_nm`` with the Angstrom power law."""

    bands = available_aod_bands(row)
    if len(bands) < 2:
        return None

    below = [band for band in bands if band[0] <= wavelength_nm]
    above = [band for band in bands if band[0] >= wavelength_nm]
    if below and above and below[-1][0] == above[0][0]:
        return above[0][1]

    low = below[-1] if below else bands[0]
    high = above[0] if above else bands[-1]
    if low[0] == high[0]:
        return low[1]

    alpha = -math.log(high[1] / low[1]) / math.log(high[0] / low[0])
    return low[1] * (wavelength_nm / low[0]) ** (-alpha)


def candidate_sites(
    query: AodQuery,
    sites: Sequence[AeronetSite],
) -> list[tuple[AeronetSite, float]]:
    """Return AERONET sites inside the query distance window, nearest first."""

    max_distance_km = query.max_distance_km or REGIONAL_KM
    candidates = [
        (
            site,
            haversine_km(query.latitude, query.longitude, site.latitude, site.longitude),
        )
        for site in sites
    ]
    return sorted(
        [(site, distance_km) for site, distance_km in candidates if distance_km <= max_distance_km],
        key=lambda item: item[1],
    )


def from_aeronet(
    query: AodQuery,
    data_dir: str | Path | None = None,
    *,
    session: requests.Session | None = None,
    sites: Sequence[AeronetSite] | None = None,
    level: str = "AOD15",
    avg: int = 10,
) -> AodEstimate | None:
    """Return the nearest usable AERONET AOD550 estimate for a query.

    ``data_dir`` is accepted for the provider protocol and unused: AERONET is
    queried live and nothing is written to the data directory.
    """

    _ = data_dir
    session = session or requests.Session()
    sites = list(sites) if sites is not None else fetch_site_list(session)
    max_dt_minutes = query.max_dt_minutes or DEFAULT_MAX_DT_MINUTES
    when = query.when.replace(tzinfo=None)
    day = when.date()

    for site, distance_km in candidate_sites(query, sites):
        rows = query_aod_rows(site.name, day, day, level=level, avg=avg, session=session)
        if not rows:
            continue
        row, seconds = nearest_in_time(rows, when)
        dt_minutes = seconds / 60.0
        if dt_minutes > max_dt_minutes:
            continue
        value = aod_at(row, 550.0)
        if value is None:
            continue
        return AodEstimate(
            value=float(value),
            source=AodSource.AERONET,
            independence="measurement",
            distance_km=float(distance_km),
            dt_minutes=float(dt_minutes),
            detail=f"site {site.name}",
        )

    return None


def aeronet_providers(
    *,
    session: requests.Session | None = None,
    sites: Sequence[AeronetSite] | None = None,
    level: str = "AOD15",
    avg: int = 10,
):
    """Return a provider mapping containing the AERONET source."""

    def provider(query: AodQuery, data_dir: str | Path | None = None) -> AodEstimate | None:
        return from_aeronet(
            query,
            data_dir,
            session=session,
            sites=sites,
            level=level,
            avg=avg,
        )

    return {AodSource.AERONET: provider}
