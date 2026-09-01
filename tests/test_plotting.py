"""Smoke tests for the sensitivity figure primitives.

These render every panel on the Agg backend and assert only that the call
succeeds and draws something. The figure content is a visual-design concern
covered by review, not by assertion.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from atmoresponse import plotting
from atmoresponse.sensitivity import ReconstructionGap, SensitivityResult

N = 12
SHAPE = (4, 6)


def _make_result(*, labeled: bool) -> SensitivityResult:
    rng = np.random.default_rng(0)
    rows = np.repeat(np.arange(2), 6)
    cols = np.tile(np.arange(6), 2)
    shipped_aod = np.linspace(0.05, 0.35, N)
    at_reference = 0.3 + 0.05 * rng.standard_normal(N)
    delta = 0.02 * rng.standard_normal(N)
    at_shipped = at_reference + delta
    curve_aod = np.linspace(0.02, 0.4, 9)
    selected = [("P05", 0), ("P50", 6), ("P95", 11)]
    curves = {label: (curve_aod - 0.2) * 0.1 for label, _ in selected}

    label_shipped = label_reference = class_changed = None
    if labeled:
        label_shipped = np.array(["burn"] * 8 + ["veg"] * 4, dtype=object)
        label_reference = np.array(["burn"] * 6 + ["veg"] * 6, dtype=object)
        class_changed = label_shipped != label_reference

    return SensitivityResult(
        rows=rows, cols=cols, shape=SHAPE, shipped_aod=shipped_aod,
        cwv_g_cm2=np.ones(N), clamped=np.zeros(N, dtype=bool),
        at_shipped=at_shipped, at_reference=at_reference, delta=delta,
        label_shipped=label_shipped, label_reference=label_reference,
        class_changed=class_changed, curve_aod550=curve_aod, curves=curves,
        selected=selected, reference_aod=0.12, unit="index",
        algorithm_name="demo",
    )


def test_spectral_axis_applies_grid():
    fig, ax = plt.subplots()
    ax.plot([400, 900], [0.1, 0.2])
    plotting.spectral_axis(ax, step=100)
    assert ax.xaxis.get_gridlines()
    plt.close(fig)


def test_curve_panel_with_references():
    result = _make_result(labeled=False)
    fig, ax = plt.subplots()
    plotting.sensitivity_curve_panel(
        ax, result,
        references=[("aeronet", 0.11, "site X"), ("viirs", 0.19, "10 km")],
        unit="index")
    assert ax.lines and ax.get_legend() is not None
    plt.close(fig)


@pytest.mark.parametrize("labeled", [False, True])
def test_full_figure_renders(labeled):
    result = _make_result(labeled=labeled)
    fig = plotting.sensitivity_figure(
        result, references=[("aeronet", 0.11, "site X")], unit="index",
        title="demo figure")
    assert len(fig.axes) >= 3  # three panels plus colourbars
    plt.close(fig)


@pytest.mark.parametrize("labeled", [False, True])
def test_full_figure_with_footprint_and_scoring_region(labeled):
    from dataclasses import replace

    base = _make_result(labeled=labeled)
    footprint = np.zeros(SHAPE, dtype=bool)
    footprint[:, 1:] = True                       # column 0 is outside the swath
    result = replace(base, footprint=footprint)
    region = np.zeros(SHAPE, dtype=bool)
    region[0, :] = True                           # top row is the scored region
    fig = plotting.sensitivity_figure(
        result, unit="index", scoring_region=region, metric_loc="upper right")
    badge = fig.axes[2].texts[-1].get_text()
    assert "in " in badge and "out " in badge
    plt.close(fig)


@pytest.mark.parametrize("kwargs", [{"symlog": True}, {"delta_vlim": (-0.01, 0.05)},
                                    {"delta_vlim": (0.0, 0.05)}])
def test_map_panel_scaling_modes(kwargs):
    result = _make_result(labeled=True)
    fig, ax = plt.subplots()
    plotting.sensitivity_map_panel(ax, result, unit="index", **kwargs)
    assert ax.images
    plt.close(fig)


def test_distribution_and_reconstruction_panels():
    result = _make_result(labeled=False)
    fig, (a, b) = plt.subplots(1, 2)
    plotting.sensitivity_distribution_panel(a, result.at_shipped, result.at_reference,
                                            unit="index")
    gap = ReconstructionGap(gap=result.delta * 1.5, median_abs_gap=0.01, p90_abs_gap=0.03)
    plotting.reconstruction_gap_panel(b, gap, result.delta, unit="index")
    assert a.patches and b.patches
    plt.close(fig)
