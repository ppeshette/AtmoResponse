import pytest

from atmoresponse.geo import haversine_km


def test_haversine_km_is_zero_for_same_point():
    assert haversine_km(34.0, -118.0, 34.0, -118.0) == 0.0


def test_haversine_km_returns_great_circle_distance():
    assert haversine_km(34.0, -118.0, 35.0, -118.0) == pytest.approx(111.195, abs=0.001)
