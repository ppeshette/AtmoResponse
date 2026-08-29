import datetime as dt

import pytest

from atmoresponse.aod import (
    AodEstimate,
    AodQuery,
    AodSource,
    agrees,
    best_aod,
    expected_error,
    gather_aod,
    resolve_aod,
    summarize_aod,
)


QUERY = AodQuery(latitude=34.0, longitude=-118.0, when=dt.datetime(2026, 8, 29, 18))


def estimate(source, value=0.12, distance_km=10.0, dt_minutes=0.0):
    independence = "assimilated" if source is AodSource.MERRA2 else "measurement"
    return AodEstimate(
        value=value,
        source=source,
        independence=independence,
        distance_km=distance_km,
        dt_minutes=dt_minutes,
        detail=f"{source.value} fixture",
    )


def test_aod_estimate_combines_space_and_time_separation():
    ref = estimate(AodSource.GOES, distance_km=25.0, dt_minutes=30.0)

    assert ref.separation_km == pytest.approx(35.355, abs=0.001)
    assert ref.tier == "regional"


def test_aod_estimate_labels_coarse_representativeness_tiers():
    assert estimate(AodSource.AERONET, distance_km=20.0).tier == "co-located"
    assert estimate(AodSource.GOES, distance_km=80.0).tier == "regional"
    assert estimate(AodSource.VIIRS, distance_km=120.0).tier == "distant"


def test_merra2_estimate_carries_outward_caveat():
    assert estimate(AodSource.MERRA2).outward_caveat is not None
    assert estimate(AodSource.AERONET).outward_caveat is None


def test_expected_error_agreement_is_two_sided():
    ref = estimate(AodSource.AERONET, value=0.20)

    assert expected_error(ref.value) == pytest.approx(0.08)
    assert agrees(0.12, ref)
    assert agrees(0.28, ref)
    assert not agrees(0.30, ref)


def test_summarize_aod_uses_median_and_carries_distribution_stats():
    summary = summarize_aod([0.1, 0.2, 0.9], detail="fixture")

    assert summary.value == 0.2
    assert summary.statistic == "median"
    assert summary.count == 3
    assert summary.mean == pytest.approx(0.4)
    assert summary.std == pytest.approx(0.3559026)
    assert summary.minimum == 0.1
    assert summary.maximum == 0.9
    assert summary.detail == "fixture"


def test_summarize_aod_ignores_nan_and_optional_invalid_pixels():
    summary = summarize_aod(
        [0.1, float("nan"), 0.3, 0.9],
        valid_mask=[True, True, True, False],
    )

    assert summary.value == 0.2
    assert summary.count == 2


def test_summarize_aod_rejects_empty_or_mismatched_inputs():
    with pytest.raises(ValueError, match="no valid AOD"):
        summarize_aod([float("nan")])

    with pytest.raises(ValueError, match="valid_mask shape"):
        summarize_aod([0.1, 0.2], valid_mask=[True])


def test_gather_aod_preserves_source_order_and_skips_missing_data():
    providers = {
        AodSource.VIIRS: lambda query, cache: estimate(AodSource.VIIRS),
        AodSource.AERONET: lambda query, cache: None,
        AodSource.GOES: lambda query, cache: estimate(AodSource.GOES),
    }

    refs = gather_aod(
        QUERY,
        providers=providers,
        sources=(AodSource.AERONET, AodSource.GOES, AodSource.VIIRS, AodSource.MERRA2),
    )

    assert [ref.source for ref in refs] == [AodSource.GOES, AodSource.VIIRS]


def test_best_aod_prefers_aeronet_when_available():
    refs = [
        estimate(AodSource.GOES, distance_km=1.0, dt_minutes=0.0),
        estimate(AodSource.AERONET, distance_km=40.0, dt_minutes=0.0),
    ]

    assert best_aod(refs).source is AodSource.AERONET


def test_best_aod_uses_separation_when_aeronet_is_absent():
    refs = [
        estimate(AodSource.VIIRS, distance_km=1.0, dt_minutes=180.0),
        estimate(AodSource.GOES, distance_km=20.0, dt_minutes=5.0),
    ]

    assert best_aod(refs).source is AodSource.GOES


def test_resolve_aod_uses_injected_providers():
    providers = {
        AodSource.AERONET: lambda query, cache: estimate(AodSource.AERONET, distance_km=30.0),
        AodSource.GOES: lambda query, cache: estimate(AodSource.GOES, distance_km=1.0),
    }

    ref = resolve_aod(QUERY, providers=providers)

    assert ref.source is AodSource.AERONET


def test_resolve_aod_without_providers_is_explicit():
    with pytest.raises(NotImplementedError, match="AOD resolution"):
        resolve_aod(QUERY)
