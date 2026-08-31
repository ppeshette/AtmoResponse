"""Figure primitives for spectral plots and for the Potential and Realized
Sensitivity results that :mod:`atmoresponse.sensitivity` produces.

Importing this module requires matplotlib (the ``plot`` optional dependency).
It gives every :class:`~atmoresponse.sensitivity.SensitivityResult` one shared
set of panels so that figures stay consistent from one algorithm to the next.
:data:`REFERENCE_STYLES` fixes the colour and linestyle of each reference AOD
source, and :func:`sensitivity_figure` is the standard three panel layout.
"""
import numpy as np
from matplotlib.colors import ListedColormap, Normalize, SymLogNorm, TwoSlopeNorm
from matplotlib.ticker import MultipleLocator

from atmoresponse.aod import expected_error

# One reference-source to (colour, linestyle) mapping, used everywhere a
# reference AOD is drawn so the same source always renders the same way.
REFERENCE_STYLES = {
    "aeronet": ("#24292f", "-"),
    "goes": ("#8250df", "--"),
    "viirs": ("#0969da", ":"),
    "merra2": ("#57606a", "-."),
}

# Distinct from coolwarm's own near-white zero centre. Without it a computed
# near-zero delta and a masked or no-data pixel render identically.
_NODATA_COLOR = "#999999"

# A hue absent from coolwarm and from the usual sequential value-panel maps
# (copper, Oranges, Blues, viridis all skip pure green), so the class-flip
# overlay stays visible whatever the rest of the figure uses.
_FLIP_COLOR = "lime"

_CURVE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]


def spectral_axis(ax, step=100, alpha=0.3, rotation=45, fontsize=8):
    """Apply the standard wavelength-axis convention with gridlines every
    ``step`` nanometres.

    Dense 100 nm ticks are what make a many-hundred-band spectrum readable. A
    feature such as the ferric-iron shoulder near 500 and 650 to 700 nm or the
    chlorophyll features near 665 and 700 nm can then be located by eye.
    """
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.grid(True, which="major", alpha=alpha)
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
        label.set_ha("right")
        label.set_fontsize(fontsize)
    return ax


def sensitivity_curve_panel(ax, result, references=(), *, title="", unit=""):
    """Potential Sensitivity: one swept curve per representative pixel
    (``result.curves``, keyed by percentile label), each marked at its own
    shipped-AOD point with an ``x``, plus every reference AOD in ``references``.

    ``references`` is an iterable of ``(source, value, detail)`` tuples where
    ``source`` is a key into :data:`REFERENCE_STYLES`. Pass every gathered
    reference, not only the one driving the Realized Sensitivity delta, so the
    reader sees the full spread of independent AOD estimates.
    """
    for color, (label, idx) in zip(_CURVE_COLORS, result.selected):
        curve = result.curves[label]
        ax.plot(result.curve_aod550, curve, color=color, marker="o", ms=3, label=label)
        ax.scatter(result.shipped_aod[idx], result.delta[idx], color=color,
                   marker="x", s=42, zorder=4)

    width = expected_error(result.reference_aod)
    ax.axvspan(result.reference_aod - width, result.reference_aod + width,
               color="#d0d7de", alpha=0.35, label="conventional expected-error envelope")
    for source, value, detail in references:
        color, linestyle = REFERENCE_STYLES[source]
        ax.axvline(value, color=color, linestyle=linestyle, lw=1,
                   label=f"{source.upper()} {value:.4f} ({detail})")

    ax.set(title=title or f"{result.algorithm_name} Potential Sensitivity",
           xlabel="assumed AOD550 (Realized Sensitivity marker: x)",
           ylabel="output minus reference-AOD output" + (f" ({unit})" if unit else ""))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6, ncol=2, frameon=False, loc="best")
    return ax


def _nodata_cmap(name):
    """A new colormap based on ``name`` with NaN drawn in :data:`_NODATA_COLOR`."""
    import matplotlib.pyplot as plt

    return plt.get_cmap(name).with_extremes(bad=_NODATA_COLOR)


def _bbox(rows, cols, shape, pad=3):
    """A tight ``(r0, r1, c0, c1)`` crop window around ``rows`` and ``cols``,
    padded by ``pad`` pixels and clipped to ``shape``.

    A result's maps are full-scene-shaped so that ``value_map`` and ``delta_map``
    land in real scene coordinates. For an AOI that is a small fraction of the
    scene, plotting the full canvas would waste most of the panel on no-data
    grey. This crop is display only and the underlying arrays stay full sized.
    """
    r0 = max(int(np.min(rows)) - pad, 0)
    c0 = max(int(np.min(cols)) - pad, 0)
    r1 = min(int(np.max(rows)) + pad + 1, shape[0])
    c1 = min(int(np.max(cols)) + pad + 1, shape[1])
    return r0, r1, c0, c1


def sensitivity_map_panel(ax, result, *, title="", unit="", show_class_changed=True,
                          symlog=False, delta_vlim=None):
    """Realized Sensitivity: the delta map cropped to a tight box around the
    AOI's own pixels, drawn coolwarm and centred on zero at the scene's own 1st
    and 99th percentile symmetric limits.

    Where ``result.class_changed`` marks a label flip, a crisp per-pixel
    translucent :data:`_FLIP_COLOR` fill is overlaid. A continuous algorithm with
    no ``class_changed`` skips the overlay. A per-pixel fill is used rather than a
    contour because contouring a scattered boolean field draws a ring around
    every isolated flipped pixel and reads as noise. A smoothed flip-density
    contour was also rejected because it hides widespread-but-scattered flipping
    behind a threshold and loses the sense of how pervasive the flipping is.

    ``symlog=True`` swaps the linear norm for a symmetric-log one, with the
    linear threshold at the 25th percentile of the absolute delta so the bulk of
    the population still reads near-linearly and only the tail gets the log
    stretch. Use it for an algorithm whose delta spans several orders of
    magnitude, such as a ratio that approaches a near-zero denominator.

    ``delta_vlim=(lo, hi)`` overrides the automatic symmetric limits. A
    near-uniform one-sign delta, which a railed shipped AOD produces, otherwise
    gets crushed into one saturated end by the zero-centred scale. When the given
    range straddles zero the panel keeps zero-centred coolwarm. When it does not,
    the panel switches to a sequential map over ``[lo, hi]`` so residual spatial
    structure stays legible. ``delta_vlim`` is ignored under ``symlog``.
    """
    delta = result.delta_map()
    r0, r1, c0, c1 = _bbox(result.rows, result.cols, delta.shape)
    delta = delta[r0:r1, c0:c1]
    finite = delta[np.isfinite(delta)]
    limit = max(abs(v) for v in np.nanpercentile(finite, (1, 99))) if finite.size else 1e-12
    limit = limit or 1e-12
    cmap = _nodata_cmap("coolwarm")
    if delta_vlim is not None and not symlog:
        lo, hi = float(delta_vlim[0]), float(delta_vlim[1])
        if lo < 0 < hi:
            norm = TwoSlopeNorm(vmin=lo, vcenter=0, vmax=hi)
        else:
            norm = Normalize(vmin=lo, vmax=hi)
            cmap = _nodata_cmap("plasma")
    elif symlog:
        abs_finite = np.abs(finite[finite != 0])
        linthresh = float(np.nanpercentile(abs_finite, 25)) if abs_finite.size else 1e-12
        linthresh = linthresh or 1e-12
        norm = SymLogNorm(linthresh=linthresh, vmin=-limit, vmax=limit)
    else:
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    image = ax.imshow(delta, cmap=cmap, interpolation="none", norm=norm)
    if show_class_changed and result.class_changed is not None:
        changed = result.class_changed_map()[r0:r1, c0:c1]
        fill = np.where(changed, 1.0, np.nan)
        ax.imshow(fill, cmap=ListedColormap([_FLIP_COLOR]), alpha=1.0, interpolation="none")
        title = title or (f"{result.algorithm_name} Realized Sensitivity\n"
                          f"{_FLIP_COLOR}: class changed")
    ax.set(title=title or f"{result.algorithm_name} Realized Sensitivity")
    ax.set_axis_off()
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04,
                       label="LUT at shipped AOD minus LUT at reference AOD"
                             + (f" ({unit})" if unit else ""))
    return ax


def sensitivity_value_panel(ax, result, values, *, title="", unit="", cmap="viridis"):
    """The absolute-output companion to :func:`sensitivity_map_panel`, so a delta
    has a scale to be read against. A delta of ``-0.1`` means something different
    on an output near ``0.3`` than on one near ``2.5``.

    ``values`` is any per-pixel array aligned with ``result.rows`` and
    ``result.cols``, typically ``result.at_shipped`` or ``result.at_reference``.
    It is scattered onto the scene grid and cropped to the AOI bounding box. The
    panel is not centred on zero, so it uses a plain sequential colormap.
    """
    value_map = result.value_map(values)
    r0, r1, c0, c1 = _bbox(result.rows, result.cols, value_map.shape)
    value_map = value_map[r0:r1, c0:c1]
    finite = value_map[np.isfinite(value_map)]
    vmin, vmax = (np.nanpercentile(finite, (1, 99)) if finite.size else (0.0, 1.0))
    image = ax.imshow(value_map, cmap=_nodata_cmap(cmap), interpolation="none",
                      vmin=vmin, vmax=vmax)
    ax.set(title=title or f"{result.algorithm_name} output")
    ax.set_axis_off()
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=unit or "output")
    return ax


def sensitivity_distribution_panel(ax, at_shipped, at_reference, *, unit="", title=""):
    """Overlaid histograms of an algorithm's real output under the two AOD
    assumptions.

    This is the cheapest, least derived grounding view. It shows whether the
    whole scene's output distribution shifts under the AOD assumption or whether
    only a few outliers move. It is the picture that goes with
    :func:`~atmoresponse.sensitivity.variance_fraction`, not a replacement for
    that number.
    """
    shipped = np.asarray(at_shipped, dtype=float)
    reference = np.asarray(at_reference, dtype=float)
    shipped = shipped[np.isfinite(shipped)]
    reference = reference[np.isfinite(reference)]
    bins = np.histogram_bin_edges(np.concatenate((shipped, reference)), bins=60)
    ax.hist(reference, bins=bins, alpha=0.5, color="#57606a", label="at reference AOD")
    ax.hist(shipped, bins=bins, alpha=0.5, color="#0969da", label="at shipped AOD")
    ax.set(title=title or "output distribution: shipped versus reference AOD",
           xlabel=unit or "output", ylabel="pixel count")
    ax.legend(fontsize=8, frameon=False)
    return ax


def reconstruction_gap_panel(ax, gap_result, delta, *, unit="", title=""):
    """Overlaid histograms of the absolute reconstruction gap (the LUT-method
    axis) against the absolute Realized Sensitivity delta (the AOD-assumption
    axis).

    Together they show whether an algorithm's AOD-driven delta is large relative
    to the LUT's own baseline gap against the delivered ISOFIT product, or is
    comparable to it and therefore hard to separate from reconstruction noise.
    """
    abs_gap = np.abs(gap_result.gap)
    abs_gap = abs_gap[np.isfinite(abs_gap)]
    abs_delta = np.abs(np.asarray(delta, dtype=float))
    abs_delta = abs_delta[np.isfinite(abs_delta)]
    bins = np.histogram_bin_edges(np.concatenate((abs_gap, abs_delta)), bins=60)
    ax.hist(abs_gap, bins=bins, alpha=0.5, color="#8250df",
            label="|LUT at shipped minus delivered ISOFIT| (method gap)")
    ax.hist(abs_delta, bins=bins, alpha=0.5, color="#0969da",
            label="|LUT at shipped minus LUT at reference| (Realized Sensitivity)")
    ax.set(title=title or "method gap versus AOD-driven delta",
           xlabel="absolute delta" + (f" ({unit})" if unit else ""), ylabel="pixel count")
    ax.legend(fontsize=8, frameon=False)
    return ax


def sensitivity_figure(result, *, references=(), title="", unit="", value_source="shipped",
                       value_cmap="viridis", figsize=(20, 6), symlog=False, delta_vlim=None):
    """The standard single-algorithm three-panel figure: the Potential
    Sensitivity curves, the algorithm's absolute output at ``value_source``'s
    AOD, then the Realized Sensitivity map.

    The middle panel grounds the delta, since a delta of ``-0.1`` means something
    different on an output near ``0.3`` than on one near ``2.5``. The variance
    fraction, distribution, and reconstruction-gap panels stay separate and
    optional. A substantially different layout should compose
    :func:`sensitivity_curve_panel`, :func:`sensitivity_value_panel`, and
    :func:`sensitivity_map_panel` into its own grid rather than call this.

    ``value_cmap`` forwards to :func:`sensitivity_value_panel`. A SAM angle to
    the target class is smaller for a stronger match, so the default viridis
    (dark at low values) happens to read correctly for a dark-equals-target
    intent, but another target or a caller with the opposite intuition may want
    a different map.
    """
    import matplotlib.pyplot as plt

    values = result.at_shipped if value_source == "shipped" else result.at_reference
    figure, axes = plt.subplots(1, 3, figsize=figsize)
    sensitivity_curve_panel(axes[0], result, references, unit=unit)
    sensitivity_value_panel(axes[1], result, values, unit=unit, cmap=value_cmap,
                            title=f"{result.algorithm_name} output at {value_source} AOD")
    sensitivity_map_panel(axes[2], result, unit=unit, symlog=symlog, delta_vlim=delta_vlim)
    if title:
        figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.94) if title else None)
    return figure
