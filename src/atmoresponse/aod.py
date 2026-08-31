"""Aerosol optical depth reference infrastructure."""

from __future__ import annotations

import datetime as dt
import importlib
import math
import re
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .downloads import download_file
from .geo import haversine_km
from .storage import resolve_data_dir


def _require(module: str):
    """Import an optional dependency, or raise with the install hint.

    GOES, VIIRS, and MERRA-2 retrieval need packages that ship in the ``live``
    extra rather than the base install.
    """

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"the {module} package is required for this AOD provider. Install it with "
            'pip install "atmoresponse[live]"'
        ) from exc

CO_LOCATED_KM = 25.0
REGIONAL_KM = 100.0
KM_PER_MINUTE = 25.0 / 30.0
DEFAULT_MAX_DT_MINUTES = 180.0

VIIRS_PRODUCTS = ("AERDT_L2_VIIRS_SNPP", "AERDT_L2_VIIRS_NOAA20")
VIIRS_MIN_QA = 2

MERRA2_PRODUCT = "M2T1NXAER"

GOES_SATELLITES = {
    "goes19": -75.2,
    "goes16": -75.2,
    "goes18": -137.2,
    "goes17": -137.2,
}
GOES_PRODUCT = "ABI-L2-AODF"
GOES_MAX_VZA = 60.0
GOES_MAX_DQF = 1
GOES_BUCKET = "https://noaa-{sat}.s3.amazonaws.com/"


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


AodProvider = Callable[[AodQuery, "str | Path | None"], AodEstimate | None]


def _when_utc_naive(when: dt.datetime) -> dt.datetime:
    if when.tzinfo is None:
        return when
    return when.astimezone(dt.timezone.utc).replace(tzinfo=None)


def _reference_dir(data_dir: str | Path | None) -> Path:
    return resolve_data_dir(data_dir) / "aod_reference"


def _max_dt_minutes(query: AodQuery, default: float) -> float:
    return default if query.max_dt_minutes is None else query.max_dt_minutes


def _within_query_limits(query: AodQuery, estimate: AodEstimate) -> bool:
    if query.max_distance_km is not None and estimate.distance_km > query.max_distance_km:
        return False
    if query.max_dt_minutes is not None and estimate.dt_minutes > query.max_dt_minutes:
        return False
    return True


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
            raise ValueError(
                f"valid_mask shape {mask.shape} does not match values shape {array.shape}"
            )
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


def from_viirs(query: AodQuery, data_dir: str | Path | None = None) -> AodEstimate | None:
    """Return the nearest good-quality VIIRS Dark Target AOD550 pixel."""

    import h5py

    max_dt_minutes = _max_dt_minutes(query, DEFAULT_MAX_DT_MINUTES)
    granule = _fetch_earthdata_granule(VIIRS_PRODUCTS, query, max_dt_minutes, _reference_dir(data_dir))
    if granule is None:
        return None
    path, dt_minutes = granule

    with h5py.File(path, "r") as h5:
        lat = h5["geolocation_data/latitude"][:]
        lon = h5["geolocation_data/longitude"][:]
        ds = h5["geophysical_data/Optical_Depth_Land_And_Ocean"]
        raw = ds[:]
        fill = _attr_scalar(ds.attrs["_FillValue"])
        scale = _attr_scalar(ds.attrs["scale_factor"])
        qa = h5["geophysical_data/Land_Ocean_Quality_Flag"][:]

    d2 = (lat - query.latitude) ** 2 + (lon - query.longitude) ** 2
    r0, c0 = np.unravel_index(np.argmin(d2), d2.shape)
    radius = 15
    rows = slice(max(0, r0 - radius), min(lat.shape[0], r0 + radius + 1))
    cols = slice(max(0, c0 - radius), min(lat.shape[1], c0 + radius + 1))

    valid = (raw[rows, cols] != fill) & (qa[rows, cols] >= VIIRS_MIN_QA)
    if not valid.any():
        return None

    sub_d2 = np.where(
        valid,
        (lat[rows, cols] - query.latitude) ** 2 + (lon[rows, cols] - query.longitude) ** 2,
        np.inf,
    )
    best_row, best_col = np.unravel_index(np.argmin(sub_d2), sub_d2.shape)
    glat = float(lat[rows, cols][best_row, best_col])
    glon = float(lon[rows, cols][best_row, best_col])
    estimate = AodEstimate(
        value=float(raw[rows, cols][best_row, best_col]) * scale,
        source=AodSource.VIIRS,
        independence="measurement",
        distance_km=haversine_km(query.latitude, query.longitude, glat, glon),
        dt_minutes=dt_minutes,
        detail=f"{Path(path).name} QA>={VIIRS_MIN_QA}",
    )
    return estimate if _within_query_limits(query, estimate) else None


def from_merra2(query: AodQuery, data_dir: str | Path | None = None) -> AodEstimate | None:
    """Return MERRA-2 total aerosol extinction AOD550 for the nearest cell and hour."""

    xr = _require("xarray")

    path = _fetch_merra2(query, _reference_dir(data_dir))
    if path is None:
        return None

    when = _when_utc_naive(query.when)
    with xr.open_dataset(path) as ds:
        target = np.datetime64(when)
        time_index = int(np.argmin(np.abs(ds.time.values - target)))
        lat_index = int(np.argmin(np.abs(ds.lat.values - query.latitude)))
        lon_index = int(np.argmin(np.abs(ds.lon.values - query.longitude)))
        value = float(ds["TOTEXTTAU"].values[time_index, lat_index, lon_index])
        glat = float(ds.lat.values[lat_index])
        glon = float(ds.lon.values[lon_index])
        dt_minutes = abs(float((ds.time.values[time_index] - target) / np.timedelta64(1, "m")))

    estimate = AodEstimate(
        value=value,
        source=AodSource.MERRA2,
        independence="assimilated",
        distance_km=haversine_km(query.latitude, query.longitude, glat, glon),
        dt_minutes=dt_minutes,
        detail=f"{MERRA2_PRODUCT} cell {glat:.2f},{glon:.2f}",
    )
    return estimate if _within_query_limits(query, estimate) else None


def view_zenith_angle(latitude: float, longitude: float, satellite_longitude: float) -> float:
    """Return the local view zenith angle to a geostationary satellite."""

    radius_km = 6371.0
    height_km = 35786.0
    gamma = math.acos(
        max(
            -1.0,
            min(
                1.0,
                math.cos(math.radians(latitude))
                * math.cos(math.radians(longitude - satellite_longitude)),
            ),
        )
    )
    return math.degrees(
        gamma
        + math.atan(
            radius_km * math.sin(gamma)
            / (radius_km + height_km - radius_km * math.cos(gamma))
        )
    )


def goes_candidates(latitude: float, longitude: float) -> list[tuple[str, float]]:
    """Return GOES satellites that can see the point, best geometry first."""

    scored = []
    for satellite, satellite_longitude in GOES_SATELLITES.items():
        vza = view_zenith_angle(latitude, longitude, satellite_longitude)
        if vza <= GOES_MAX_VZA:
            scored.append((round(vza, 1), -int(satellite[4:]), satellite, vza))
    return [(satellite, vza) for _, _, satellite, vza in sorted(scored)]


def from_goes(query: AodQuery, data_dir: str | Path | None = None) -> AodEstimate | None:
    """Return NOAA GOES ABI full-disk AOD550 from the nearest usable pixel."""

    Dataset = _require("netCDF4").Dataset

    max_dt_minutes = _max_dt_minutes(query, 30.0)
    when = _when_utc_naive(query.when)
    for satellite, vza in goes_candidates(query.latitude, query.longitude):
        granule = _fetch_goes_granule(
            satellite,
            when,
            max_dt_minutes,
            _reference_dir(data_dir),
        )
        if granule is None:
            continue
        path, dt_minutes = granule
        with Dataset(path) as ds:
            projection = ds["goes_imager_projection"]
            scan_angles = _abi_scan_angles(
                query.latitude,
                query.longitude,
                float(projection.longitude_of_projection_origin),
                float(projection.semi_major_axis),
                float(projection.semi_minor_axis),
                float(projection.perspective_point_height) + float(projection.semi_major_axis),
            )
            if scan_angles is None:
                continue
            xi = int(np.abs(ds["x"][:] - scan_angles[0]).argmin())
            yi = int(np.abs(ds["y"][:] - scan_angles[1]).argmin())
            radius = 8
            rows = slice(max(yi - radius, 0), yi + radius + 1)
            cols = slice(max(xi - radius, 0), xi + radius + 1)
            aod = np.ma.filled(ds["AOD"][rows, cols].astype("f8"), np.nan)
            dqf = np.ma.filled(ds["DQF"][rows, cols].astype("f8"), 99)

        valid = np.isfinite(aod) & (dqf <= GOES_MAX_DQF)
        if not valid.any():
            continue
        yy, xx = np.mgrid[
            rows.start : rows.start + aod.shape[0],
            cols.start : cols.start + aod.shape[1],
        ]
        d2 = np.where(valid, (yy - yi) ** 2 + (xx - xi) ** 2, np.inf)
        selected = np.unravel_index(np.argmin(d2), d2.shape)
        step_km = 2.0 / max(np.cos(np.radians(vza)), 0.2)
        estimate = AodEstimate(
            value=float(aod[selected]),
            source=AodSource.GOES,
            independence="measurement",
            distance_km=float(np.sqrt(d2[selected]) * step_km),
            dt_minutes=dt_minutes,
            detail=f"{satellite.upper()} {GOES_PRODUCT} DQF<={GOES_MAX_DQF}, VZA {vza:.0f} deg",
        )
        if _within_query_limits(query, estimate):
            return estimate
    return None


def _abi_scan_angles(
    latitude: float,
    longitude: float,
    satellite_longitude: float,
    r_eq: float,
    r_pol: float,
    h: float,
) -> tuple[float, float] | None:
    lat_r = math.radians(latitude)
    dlon = math.radians(longitude - satellite_longitude)
    e2 = 1.0 - (r_pol**2) / (r_eq**2)
    phi_c = math.atan((r_pol**2) / (r_eq**2) * math.tan(lat_r))
    rc = r_pol / math.sqrt(1.0 - e2 * math.cos(phi_c) ** 2)
    sx = h - rc * math.cos(phi_c) * math.cos(dlon)
    sy = -rc * math.cos(phi_c) * math.sin(dlon)
    sz = rc * math.sin(phi_c)
    if h * (h - sx) < sy**2 + (r_eq**2 / r_pol**2) * sz**2:
        return None
    return (
        math.asin(-sy / math.sqrt(sx**2 + sy**2 + sz**2)),
        math.atan(sz / sx),
    )


def _fetch_goes_granule(
    satellite: str,
    when: dt.datetime,
    max_dt_minutes: float,
    reference_dir: Path,
) -> tuple[Path, float] | None:
    out = reference_dir / "goes" / satellite / when.strftime("%Y%m%d_%H")
    best = None
    for key, stamp in _list_goes_keys(satellite, when, max_dt_minutes):
        dt_minutes = abs((stamp - when).total_seconds()) / 60.0
        if dt_minutes <= max_dt_minutes and (best is None or dt_minutes < best[1]):
            best = (key, dt_minutes)
    if best is None:
        return None

    out.mkdir(parents=True, exist_ok=True)
    path = out / best[0].split("/")[-1]
    if not path.exists():
        download_file(GOES_BUCKET.format(sat=satellite) + best[0], path)
    return path, best[1]


def _list_goes_keys(
    satellite: str,
    when: dt.datetime,
    max_dt_minutes: float,
) -> list[tuple[str, dt.datetime]]:
    import requests
    import xml.etree.ElementTree as et

    namespace = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    window = dt.timedelta(minutes=max_dt_minutes)
    hours = {
        (when + offset).strftime("%Y/%j/%H")
        for offset in (-window, dt.timedelta(0), window)
    }
    out = []
    for hour in sorted(hours):
        response = requests.get(
            GOES_BUCKET.format(sat=satellite),
            params={"list-type": "2", "prefix": f"{GOES_PRODUCT}/{hour}/"},
            timeout=120,
        )
        if not response.ok:
            continue
        keys = et.fromstring(response.content).iter(namespace + "Key")
        for key in (element.text for element in keys):
            if key is None:
                continue
            match = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})", key)
            if match:
                year, doy, hour, minute, second = (int(group) for group in match.groups())
                stamp = dt.datetime(year, 1, 1, hour, minute, second) + dt.timedelta(days=doy - 1)
                out.append((key, stamp))
    return out


def _fetch_merra2(query: AodQuery, reference_dir: Path) -> Path | None:
    earthaccess = _require("earthaccess")

    when = _when_utc_naive(query.when)
    day = when.strftime("%Y-%m-%d")
    out = reference_dir / "merra2" / day.replace("-", "")
    hit = sorted(out.glob("MERRA2_*.nc4"))
    if hit:
        return hit[0]

    earthaccess.login(strategy="netrc")
    results = earthaccess.search_data(short_name=MERRA2_PRODUCT, temporal=(day, day))
    if not results:
        return None
    out.mkdir(parents=True, exist_ok=True)
    files = earthaccess.download(results[:1], str(out))
    return Path(files[0]) if files else None


def _fetch_earthdata_granule(
    short_names: Sequence[str],
    query: AodQuery,
    max_dt_minutes: float,
    reference_dir: Path,
) -> tuple[Path, float] | None:
    earthaccess = _require("earthaccess")

    when = _when_utc_naive(query.when)
    day = when.strftime("%Y-%m-%d")
    key = f"{day.replace('-', '')}_{query.latitude:.1f}_{query.longitude:.1f}"
    window = dt.timedelta(minutes=max_dt_minutes)
    bbox = (
        query.longitude - 0.5,
        query.latitude - 0.5,
        query.longitude + 0.5,
        query.latitude + 0.5,
    )

    best = None
    for short_name in short_names:
        out = reference_dir / short_name.lower() / key
        hit = sorted(out.glob("*.nc"))
        if not hit:
            earthaccess.login(strategy="netrc")
            results = earthaccess.search_data(
                short_name=short_name,
                bounding_box=bbox,
                temporal=(
                    (when - window).isoformat(),
                    (when + window).isoformat(),
                ),
            )
            if not results:
                continue
            out.mkdir(parents=True, exist_ok=True)
            hit = [Path(path) for path in earthaccess.download(results, str(out))]
        for path in hit:
            dt_minutes = _granule_offset_minutes(path, when)
            if dt_minutes is not None and (best is None or dt_minutes < best[1]):
                best = (Path(path), dt_minutes)
    return best


def _granule_offset_minutes(path: str | Path, when: dt.datetime) -> float | None:
    """Return minutes from ``when`` for a file with an ``.AYYYYDDD.HHMM.`` stamp."""

    parts = Path(path).name.split(".")
    for index, part in enumerate(parts):
        if part.startswith("A") and len(part) == 8 and part[1:].isdigit():
            try:
                stamp = dt.datetime.strptime(part[1:] + parts[index + 1], "%Y%j%H%M")
            except (ValueError, IndexError):
                return None
            return abs((stamp - when.replace(tzinfo=None)).total_seconds()) / 60.0
    return None


def _attr_scalar(value) -> float:
    return float(np.asarray(value).ravel()[0])


def default_providers(
    sources: Sequence[AodSource] = (
        AodSource.AERONET,
        AodSource.GOES,
        AodSource.VIIRS,
        AodSource.MERRA2,
    ),
) -> dict[AodSource, AodProvider]:
    """Return the built-in provider mapping for the requested sources."""

    providers: dict[AodSource, AodProvider] = {}
    requested = set(sources)
    if AodSource.AERONET in requested:
        from .aeronet import aeronet_providers

        providers.update(aeronet_providers())
    if AodSource.GOES in requested:
        providers[AodSource.GOES] = from_goes
    if AodSource.VIIRS in requested:
        providers[AodSource.VIIRS] = from_viirs
    if AodSource.MERRA2 in requested:
        providers[AodSource.MERRA2] = from_merra2
    return providers


def gather_aod(
    query: AodQuery,
    providers: Mapping[AodSource, AodProvider] | None = None,
    sources: Sequence[AodSource] = (
        AodSource.AERONET,
        AodSource.GOES,
        AodSource.VIIRS,
        AodSource.MERRA2,
    ),
    data_dir: str | Path | None = None,
    *,
    strict: bool = False,
) -> list[AodEstimate]:
    """Return every AOD reference that resolves, preserving requested source order."""

    if providers is None:
        providers = default_providers(sources)

    estimates: list[AodEstimate] = []
    for source in sources:
        provider = providers.get(source)
        if provider is None:
            continue
        try:
            estimate = provider(query, data_dir)
        except Exception as exc:
            if strict:
                raise RuntimeError(f"{source.value} AOD provider failed") from exc
            warnings.warn(f"{source.value} AOD provider skipped: {exc}", stacklevel=2)
            continue
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
    data_dir: str | Path | None = None,
    providers: Mapping[AodSource, AodProvider] | None = None,
    strict: bool = False,
) -> AodEstimate | None:
    """Resolve the best available external AOD550 estimate."""

    if providers is None:
        providers = default_providers(sources)

    return best_aod(
        gather_aod(query, providers=providers, sources=sources, data_dir=data_dir, strict=strict)
    )
