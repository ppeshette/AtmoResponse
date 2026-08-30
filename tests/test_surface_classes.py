import datetime as dt

import pytest
from shapely.geometry import box

from atmoresponse.catalog import SceneRecord
from atmoresponse.surface_classes import (
    LandCoverSample,
    classify_scene_surface,
    classify_scene_with_worldcover,
    fractions_from_counts,
    tags_from_fractions,
    worldcover_tiles_for_bbox,
)


def _scene(source="tanager"):
    geometry = box(-119.0, 34.0, -118.0, 35.0)
    return SceneRecord(
        scene_id=f"{source}-scene",
        acquired=dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc),
        bbox=tuple(geometry.bounds),
        source=source,
        geometry=geometry,
    )


def test_fractions_from_counts_names_worldcover_classes_and_drops_nodata():
    fractions = fractions_from_counts({0: 99, 40: 30, 50: 20, 80: 50})

    assert fractions == {
        "cropland": 0.3,
        "built": 0.2,
        "water": 0.5,
    }


def test_tags_from_fractions_applies_capability_thresholds():
    tags = tags_from_fractions({
        "water": 0.05,
        "wetland": 0.03,
        "cropland": 0.16,
        "built": 0.09,
        "tree": 0.24,
    })

    assert tags == ("water", "cropland")


def test_classify_scene_surface_is_source_neutral():
    for source in ("tanager", "emit"):
        result = classify_scene_surface(
            _scene(source),
            LandCoverSample(counts={80: 80, 50: 20}, expected_pixels=100, missing_tiles=0),
        )

        assert result.scene.source == source
        assert result.tags == ("water", "built")
        assert result.valid is True
        assert result.coverage_fraction == 1.0
        assert result.properties["landcover_source"] == "ESA WorldCover 2021 v200"


def test_classify_scene_surface_marks_missing_or_low_coverage_invalid():
    scene = _scene()

    missing = classify_scene_surface(scene, LandCoverSample(counts={80: 100}, missing_tiles=1))
    low_coverage = classify_scene_surface(scene, LandCoverSample(counts={80: 10}, expected_pixels=100))

    assert missing.valid is False
    assert low_coverage.valid is False


def test_worldcover_tiles_for_bbox_uses_esa_three_degree_grid():
    assert worldcover_tiles_for_bbox((-1.0, 1.0, 4.0, 5.0)) == (
        "N00W003",
        "N00E000",
        "N00E003",
        "N03W003",
        "N03E000",
        "N03E003",
    )


def test_classify_scene_with_worldcover_uses_scene_geometry():
    scene = _scene("emit")
    seen = []

    def sampler(geometry):
        seen.append(geometry.bounds)
        return LandCoverSample(counts={60: 90, 80: 10}, expected_pixels=100)

    result = classify_scene_with_worldcover(scene, sampler=sampler)

    assert seen == [scene.geometry.bounds]
    assert result.tags == ("water", "bare")


def test_classify_scene_with_worldcover_requires_geometry():
    scene = SceneRecord(
        scene_id="missing-geometry",
        acquired=dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc),
        bbox=(-119.0, 34.0, -118.0, 35.0),
    )

    with pytest.raises(ValueError, match="geometry"):
        classify_scene_with_worldcover(scene, sampler=lambda geometry: LandCoverSample({}))
