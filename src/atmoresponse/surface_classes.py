"""Source-neutral scene-footprint surface classification."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from .catalog import SceneRecord

WORLDCOVER_URL_TEMPLATE = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
)

WORLDCOVER_CLASS_NAMES = {
    10: "tree",
    20: "shrub",
    30: "grass",
    40: "cropland",
    50: "built",
    60: "bare",
    70: "snow_ice",
    80: "water",
    90: "wetland",
    95: "mangrove",
    100: "moss",
}

DEFAULT_TAG_RULES = {
    "water": (("water", "wetland", "mangrove"), 0.08),
    "cropland": (("cropland",), 0.15),
    "built": (("built",), 0.10),
    "bare": (("bare",), 0.15),
    "forest": (("tree",), 0.25),
}


@dataclass(frozen=True)
class LandCoverSample:
    """Land-cover class counts sampled under one scene footprint."""

    counts: Mapping[int, int]
    expected_pixels: float | None = None
    missing_tiles: int = 0


@dataclass(frozen=True)
class SurfaceClassification:
    """Surface fractions and reusable capability tags for one scene."""

    scene: SceneRecord
    fractions: Mapping[str, float]
    tags: tuple[str, ...]
    valid: bool
    pixel_count: int
    expected_pixels: float | None = None
    coverage_fraction: float | None = None
    missing_tiles: int = 0
    properties: Mapping[str, object] = field(default_factory=dict)


def worldcover_tiles_for_bbox(bbox: tuple[float, float, float, float]) -> tuple[str, ...]:
    """Return ESA WorldCover tile ids touched by a lon/lat bbox."""

    minx, miny, maxx, maxy = bbox
    tiles = []
    for lat0 in range(int(math.floor(miny / 3) * 3), int(math.floor(maxy / 3) * 3) + 3, 3):
        for lon0 in range(int(math.floor(minx / 3) * 3), int(math.floor(maxx / 3) * 3) + 3, 3):
            ns = f"N{lat0:02d}" if lat0 >= 0 else f"S{-lat0:02d}"
            ew = f"E{lon0:03d}" if lon0 >= 0 else f"W{-lon0:03d}"
            tiles.append(ns + ew)
    return tuple(tiles)


def geometry_area_km2(geometry) -> float:
    """Approximate lon/lat geometry area in square kilometers."""

    minx, miny, maxx, maxy = geometry.bounds
    lat0 = math.radians((miny + maxy) / 2)
    return geometry.area * (111.320**2) * math.cos(lat0)


def fractions_from_counts(
    counts: Mapping[int, int],
    class_names: Mapping[int, str] = WORLDCOVER_CLASS_NAMES,
) -> dict[str, float]:
    """Convert integer land-cover class counts into named fractions."""

    named_counts: dict[str, int] = {}
    for class_id, count in counts.items():
        if class_id == 0 or count <= 0:
            continue
        name = class_names.get(class_id, f"class_{class_id}")
        named_counts[name] = named_counts.get(name, 0) + int(count)

    total = sum(named_counts.values())
    if total == 0:
        return {}
    return {name: count / total for name, count in named_counts.items()}


def tags_from_fractions(
    fractions: Mapping[str, float],
    tag_rules: Mapping[str, tuple[Sequence[str], float]] = DEFAULT_TAG_RULES,
) -> tuple[str, ...]:
    """Apply reusable surface tags to named class fractions."""

    tags = []
    for tag, (classes, threshold) in tag_rules.items():
        if sum(fractions.get(name, 0.0) for name in classes) >= threshold:
            tags.append(tag)
    return tuple(tags)


def classify_scene_surface(
    scene: SceneRecord,
    sample: LandCoverSample,
    tag_rules: Mapping[str, tuple[Sequence[str], float]] = DEFAULT_TAG_RULES,
    min_coverage_fraction: float = 0.85,
) -> SurfaceClassification:
    """Classify one Tanager or EMIT scene footprint from land-cover counts."""

    fractions = fractions_from_counts(sample.counts)
    pixel_count = sum(int(count) for class_id, count in sample.counts.items() if class_id != 0 and count > 0)
    coverage = None
    if sample.expected_pixels is not None and sample.expected_pixels > 0:
        coverage = pixel_count / sample.expected_pixels

    valid = sample.missing_tiles == 0
    if coverage is not None:
        valid = valid and coverage >= min_coverage_fraction

    return SurfaceClassification(
        scene=scene,
        fractions=fractions,
        tags=tags_from_fractions(fractions, tag_rules),
        valid=valid,
        pixel_count=pixel_count,
        expected_pixels=sample.expected_pixels,
        coverage_fraction=coverage,
        missing_tiles=sample.missing_tiles,
        properties={"landcover_source": "ESA WorldCover 2021 v200"},
    )


def sample_worldcover(
    geometry,
    source_template: str = WORLDCOVER_URL_TEMPLATE,
) -> LandCoverSample:
    """Sample ESA WorldCover classes under a lon/lat scene footprint."""

    import numpy as np
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.windows import from_bounds

    minx, miny, maxx, maxy = geometry.bounds
    counts: dict[int, int] = {}
    missing_tiles = 0
    for tile in worldcover_tiles_for_bbox((minx, miny, maxx, maxy)):
        try:
            with rasterio.open(source_template.format(tile=tile)) as ds:
                left = max(minx, ds.bounds.left)
                bottom = max(miny, ds.bounds.bottom)
                right = min(maxx, ds.bounds.right)
                top = min(maxy, ds.bounds.top)
                if left >= right or bottom >= top:
                    continue
                window = from_bounds(left, bottom, right, top, ds.transform)
                values = ds.read(1, window=window)
                if values.size == 0:
                    continue
                inside = ~geometry_mask([geometry], values.shape, ds.window_transform(window), invert=False)
                class_ids, class_counts = np.unique(values[inside], return_counts=True)
                for class_id, count in zip(class_ids.tolist(), class_counts.tolist()):
                    if class_id == 0:
                        continue
                    counts[int(class_id)] = counts.get(int(class_id), 0) + int(count)
        except rasterio.errors.RasterioIOError:
            missing_tiles += 1

    return LandCoverSample(
        counts=counts,
        expected_pixels=geometry_area_km2(geometry) * 10_000,
        missing_tiles=missing_tiles,
    )


def classify_scene_with_worldcover(
    scene: SceneRecord,
    sampler: Callable[[object], LandCoverSample] = sample_worldcover,
) -> SurfaceClassification:
    """Classify one scene using ESA WorldCover sampled from its geometry."""

    if scene.geometry is None:
        raise ValueError("scene geometry is required for surface classification")
    return classify_scene_surface(scene, sampler(scene.geometry))
