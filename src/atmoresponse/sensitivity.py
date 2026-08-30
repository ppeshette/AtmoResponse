"""Public entry point: run any reflectance algorithm's Potential/Realized Sensitivity
against a per-sensor atmospheric-response LUT.

The module is layered so Tanager and EMIT share everything but the scene reader
and the LUT:

- ``_run_from_arrays`` is the sensor-agnostic core. Given already-extracted pixel
  arrays, a resolved algorithm, and which LUT to use (``lut_root``/``lut_axes``),
  it runs the two sensitivity questions and assembles the ``SensitivityResult``.
- ``evaluate`` is the source-neutral per-pixel evaluation seam: arrays in,
  results plus a clamp mask out, with an injectable ``correct``.
- ``run_tanager`` and ``run_emit`` are the sensor implementations. Each does its
  own scene-path resolution, array extraction (``tanager_ortho`` or ``emit``),
  and acquisition-date parsing, then calls ``_run_from_arrays`` with its sensor's
  LUT.

Everything is measured against the LUT, never against a live radiative-transfer
call.

Two sensitivity questions:

- **Potential Sensitivity** sweeps a handful of representative pixels across a
  plausible AOD range. It is a property of the algorithm in general, not of this
  scene.
- **Realized Sensitivity** evaluates every masked pixel at exactly two AOD values
  (this scene's own shipped ISOFIT AOD, and an independent reference) and
  differences them. It is a property of this real scene's real delivery. The
  value is a difference between two LUT-corrected reflectances, not a comparison
  against ground truth, so it is "sensitivity", never "error".

Three distinct roles a boolean array can play, deliberately kept separate:

- ``mask`` is a data-quality gate (cloud/cirrus/nodata, land vs water, a
  canopy/bare-soil corroboration test). It is never derived from the thing being
  tested, so it is safe as a processing restriction.
- ``aoi`` is the block actually read off disk. It is also a processing
  restriction, but one drawn from domain knowledge (a mapped fire perimeter, a
  known ore body) risks confounding the algorithm's detection question with the
  ground truth it will later be judged against. Where that circularity matters,
  run scene-wide and use ``scoring_region`` instead.
- ``scoring_region``, on a ``SensitivityResult`` via ``.scored()``, is optional
  and strictly post-hoc. It slices already-computed results for reporting and
  never restricts what the algorithm saw.

An algorithm returns either a bare ``float`` (a continuous score) or a
``LabeledScore`` (a native classifier's assigned label plus its numeric distance,
such as SAM's angle to the nearest library member and its margin over the
second-best). ``group_labels``, if given, maps each raw label to a semantic group
before ``class_changed`` is computed, so a flip between near-duplicate library
entries is not counted the same as a flip that crosses the decision that matters.

``fit``, if given in place of ``algorithm``, runs exactly once on the AOI's raw
*radiance* population, before any AOD assumption is applied, and returns the
``Algorithm`` to use for both AOD sides and every curve point. Fitting in
radiance keeps whatever it derives (for example, in-scene endmember spectra)
independent of the atmosphere error being tested. Exactly one of
``algorithm``/``fit`` is required.

AOD is interpolated by default. Read only at the nearest LUT node
(``node_only=True``) where a documented reason exists. A narrow-differencing
algorithm's sensitivity to sub-node AOD interpolation is the case on record.
Picking node-only without one silently costs precision.

A pixel whose real shipped geometry, CWV, or AOD falls outside the LUT's
tabulated grid is **held at the nearest boundary node** (the ``clamp`` option in
``lut``). It is never masked out and never left to raise, because a dry scene can
sit entirely below the LUT's CWV floor and masking would leave too few pixels for
a Realized Sensitivity map at all. ``SensitivityResult.clamped`` marks which
pixels this touched, and the raw ``shipped_aod``/``cwv_g_cm2`` arrays are
retained, so a caller can disclose the fraction and its cause rather than have it
silently absorbed into the delta.

Single scene only. A two-date algorithm (dNBR) needs a different entry point.
This module assumes a reflectance-domain algorithm throughout. A radiance-domain
one, such as a methane matched filter, has no correction step for the sweep to
act on and cannot be pointed at this module.

Serial by default (``workers=1``). Pass ``workers=N`` to parallelize the
per-pixel Realized Sensitivity pass over a ``multiprocessing.Pool``. That imposes
one constraint: when ``workers > 1``, ``algorithm`` must be a picklable
module-level object (a small callable class with its state as attributes), not a
closure returned by a local factory, because ``multiprocessing`` uses spawn on
Windows and cannot pickle a nested closure. ``workers=1`` still accepts a
closure. The Potential Sensitivity curve sweep stays serial regardless.

The core preloads every shard the real pixel population's geometry could touch in
the main process and installs it into each worker's cache at startup, so a fresh
worker does not start with a cold cache and N processes do not contend for the
same shard files. This is a correctness-neutral optimization: a cache miss still
falls through to the normal lazy read.
"""
from __future__ import annotations

import contextlib
import datetime
import multiprocessing as mp
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import h5py
import numpy as np

from atmoresponse import tanager_ortho as extract
from atmoresponse.aod import expected_error
from atmoresponse.lut import (
    DEFAULT_OZONE_ATM_CM,
    SHARD_ROOT,
    correct_spectrum_batch_from_lut,
    correct_spectrum_from_lut,
    fold_raa_array,
    install_shards,
    load_axes,
    read_shards,
    shard_keys_needed,
)
from atmoresponse.tanager_ortho import GRID

Reflectance = Mapping[float, float]


@dataclass(frozen=True)
class LabeledScore:
    """A native classifier's per-pixel result, such as SAM's angle to its
    assigned library member. ``value`` is what the sensitivity difference acts
    on, ``label`` is the assigned class, and ``margin`` is the gap to the
    second-best assignment (kept as provenance, not used in the delta itself)."""

    value: float
    label: str
    margin: float = float("nan")


Algorithm = Callable[[Reflectance], "float | LabeledScore"]
Fit = Callable[[np.ndarray, np.ndarray], Algorithm]
Mask = Callable[["h5py.File", tuple[int, int, int, int]], np.ndarray]


def _unpack(result) -> tuple[float, str | None]:
    if result is None:
        # ``evaluate``'s permanent-LUT-gap sentinel. Mapping it to NaN lets it
        # flow through the delta and curve arithmetic. A caller recovers the
        # count with ``~np.isfinite(result.delta)``.
        return float("nan"), None
    if isinstance(result, LabeledScore):
        return float(result.value), result.label
    return float(result), None


def _snap_to_nodes(values: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    idx = np.abs(nodes[None, :] - values[:, None]).argmin(axis=1)
    return nodes[idx]


def _group(labels: np.ndarray, group_labels: Mapping[str, str] | None) -> np.ndarray:
    if group_labels is None:
        return labels
    return np.array([group_labels.get(label, label) if label is not None else None
                     for label in labels], dtype=object)


_POOL_STATE = None


def _pool_init(algorithm, correct, use_batch, wl_um, aero_profile, month, day, clamp,
               lut_root, lut_axes, preload):
    global _POOL_STATE
    _POOL_STATE = (algorithm, correct, use_batch, wl_um, aero_profile, month, day, clamp,
                   lut_root, lut_axes)
    if preload:
        install_shards(lut_root, preload)


def _reflectance_list(wl_nm, block, gap):
    """``(n, n_bands)`` reflectance block + its per-pixel gap mask -> the list of
    ``{wavelength_nm: value}`` dicts (or ``None`` for a gapped pixel) the
    algorithm layer consumes."""
    wl_nm = [float(w) for w in wl_nm]
    return [None if gap[i] else dict(zip(wl_nm, block[i])) for i in range(len(block))]


def _evaluate_algorithm(algorithm, reflectances):
    """Evaluate valid reflectances together when the algorithm supports batching."""
    results = [None] * len(reflectances)
    valid_indices = [i for i, reflectance in enumerate(reflectances) if reflectance is not None]
    if not valid_indices:
        return results
    valid = [reflectances[i] for i in valid_indices]
    evaluate_many = getattr(algorithm, "evaluate_many", None)
    evaluated = list(evaluate_many(valid)) if evaluate_many else [algorithm(item) for item in valid]
    if len(evaluated) != len(valid):
        raise ValueError("algorithm.evaluate_many() returned the wrong number of results")
    for i, result in zip(valid_indices, evaluated):
        results[i] = result
    return results


def _pool_evaluate_chunk(chunk):
    aod_chunk, cwv_chunk, sun_z, sun_a, view_z, view_a, radiance_chunk = chunk
    (algorithm, correct, use_batch, wl_um, aero_profile, month, day, clamp,
     lut_root, lut_axes) = _POOL_STATE
    wl_nm = wl_um * 1000.0
    if use_batch:
        block, gap, clamped = correct_spectrum_batch_from_lut(
            sun_z, sun_a, view_z, view_a, month, day, aero_profile,
            aod_chunk, cwv_chunk, wl_um, radiance_chunk,
            root=lut_root, axes=lut_axes, clamp=clamp)
        return _evaluate_algorithm(algorithm, _reflectance_list(wl_nm, block, gap)), clamped
    reflectances = []
    for i in range(len(aod_chunk)):
        try:
            values = correct(
                sun_z=float(sun_z[i]), sun_a=float(sun_a[i]),
                view_z=float(view_z[i]), view_a=float(view_a[i]),
                month=month, day=day, aero_profile=aero_profile, aot550=float(aod_chunk[i]),
                cwv_g_cm2=float(cwv_chunk[i]), wl_um=wl_um, L_obs=radiance_chunk[i],
            )
        except ValueError:
            # A permanent per-cell LUT gap (some aerosol/band/geometry combos are
            # unsolvable). Record the sentinel instead of failing the population.
            reflectances.append(None)
            continue
        reflectances.append(dict(zip((float(wl) for wl in wl_nm), np.asarray(values, dtype=float))))
    return (_evaluate_algorithm(algorithm, reflectances),
            np.zeros(len(aod_chunk), dtype=bool))


def evaluate(
    algorithm: Algorithm, wl_nm: np.ndarray, radiance: np.ndarray,
    geometry: Mapping[str, np.ndarray], aero_profile: str, aod550,
    month: int, day: int, cwv_g_cm2: np.ndarray,
    *, correct: Callable[..., np.ndarray] | None = None, clamp: bool = True,
    lut_root: str = SHARD_ROOT, lut_axes: Mapping | None = None,
    workers: int = 1, chunksize: int = 64, pool: "mp.pool.Pool | None" = None,
) -> tuple[list, np.ndarray]:
    """Evaluate ``algorithm`` at every pixel in ``radiance`` (n_pixels, n_bands).

    ``aod550`` is either one shared value, broadcast to every pixel (the
    reference-AOD side, and every curve point), or a per-pixel array (the
    shipped-AOD side). It is a pure function of already-extracted arrays with no
    scene I/O, so it is the source-neutral seam that ``run_tanager()`` and
    ``run_emit()`` wrap, and what a test injects a fake ``correct`` into.
    ``lut_root``/``lut_axes`` select which LUT the default corrector reads
    (Tanager or EMIT). They are ignored when ``correct`` is injected.

    Returns ``(results, clamped)``. ``results`` is one algorithm output per pixel
    (or ``None`` for a permanent LUT gap). ``clamped`` is the per-pixel mask of
    pixels held at a LUT boundary (all-False on the injected-``correct`` path).

    With ``correct is None`` (the default, the real LUT) the whole chunk is
    corrected in one ``correct_spectrum_batch_from_lut`` call. An injected
    ``correct`` is called once per pixel over the whole band vector.

    An algorithm may optionally provide ``evaluate_many(reflectances)`` to
    evaluate a population together. ``workers > 1`` distributes this loop over a
    ``multiprocessing.Pool``. Pass an already-built ``pool`` to reuse workers
    across calls. ``workers`` is ignored when ``pool`` is given.
    """
    use_batch = correct is None
    scalar_correct = correct_spectrum_from_lut if use_batch else correct
    n = len(radiance)
    aod_per_pixel = np.broadcast_to(np.atleast_1d(aod550), (n,))
    wl_um = np.asarray(wl_nm, dtype=float) / 1000.0
    if workers <= 1 and pool is None:
        results = []
        clamped_parts = []
        for start in range(0, n, chunksize):
            stop = min(start + chunksize, n)
            if use_batch:
                sl = slice(start, stop)
                block, gap, clamped_chunk = correct_spectrum_batch_from_lut(
                    geometry["sun_z"][sl], geometry["sun_a"][sl],
                    geometry["view_z"][sl], geometry["view_a"][sl],
                    month, day, aero_profile, aod_per_pixel[sl], cwv_g_cm2[sl],
                    wl_um, radiance[sl], root=lut_root, axes=lut_axes, clamp=clamp)
                reflectances = _reflectance_list(wl_nm, block, gap)
                clamped_parts.append(clamped_chunk)
            else:
                reflectances = []
                for i in range(start, stop):
                    try:
                        values = scalar_correct(
                            sun_z=float(geometry["sun_z"][i]), sun_a=float(geometry["sun_a"][i]),
                            view_z=float(geometry["view_z"][i]), view_a=float(geometry["view_a"][i]),
                            month=month, day=day, aero_profile=aero_profile,
                            aot550=float(aod_per_pixel[i]), cwv_g_cm2=float(cwv_g_cm2[i]),
                            wl_um=wl_um, L_obs=radiance[i],
                        )
                    except ValueError:
                        reflectances.append(None)
                        continue
                    reflectances.append(dict(zip(
                        (float(wl) for wl in wl_nm), np.asarray(values, dtype=float))))
                clamped_parts.append(np.zeros(stop - start, dtype=bool))
            results.extend(_evaluate_algorithm(algorithm, reflectances))
        clamped = np.concatenate(clamped_parts) if clamped_parts else np.zeros(0, dtype=bool)
        return results, clamped

    chunks = [
        (aod_per_pixel[i:i + chunksize], cwv_g_cm2[i:i + chunksize],
         geometry["sun_z"][i:i + chunksize], geometry["sun_a"][i:i + chunksize],
         geometry["view_z"][i:i + chunksize], geometry["view_a"][i:i + chunksize],
         radiance[i:i + chunksize])
        for i in range(0, n, chunksize)
    ]
    if pool is not None:
        chunk_results = pool.map(_pool_evaluate_chunk, chunks)
    else:
        preload = _preload_for_pool(aero_profile, geometry, cwv_g_cm2, lut_root, lut_axes)
        with _worker_pool(workers, algorithm, scalar_correct, use_batch, wl_um, aero_profile,
                          month, day, clamp, lut_root, lut_axes, preload=preload) as owned_pool:
            chunk_results = owned_pool.map(_pool_evaluate_chunk, chunks)
    results = [result for chunk, _ in chunk_results for result in chunk]
    clamped = (np.concatenate([mask for _, mask in chunk_results]) if chunk_results
               else np.zeros(0, dtype=bool))
    return results, clamped


def _preload_for_pool(aero_profile, geometry, cwv_g_cm2, lut_root=SHARD_ROOT, lut_axes=None):
    """Every LUT shard this pixel population's real geometry could touch, read
    once in the calling (main) process ahead of handing work to a ``Pool``. A
    fresh worker's empty cache is the parallelism bottleneck. Assumes the default
    ozone. Best-effort: returns ``{}`` (a harmless no-op for ``install_shards``)
    if ``aero_profile`` is not a tabulated model or the store is absent."""
    axes = lut_axes if lut_axes is not None else load_axes()
    aerosol_values = axes["axes"]["aerosol"]["values"]
    if aero_profile not in aerosol_values:
        return {}
    aero_idx = aerosol_values.index(aero_profile)
    raa = fold_raa_array(geometry["view_a"], geometry["sun_a"])
    ozone = np.full(np.shape(np.asarray(geometry["sun_z"])), DEFAULT_OZONE_ATM_CM)
    try:
        keys = shard_keys_needed(lut_root, aero_idx, aero_profile, axes,
                                 geometry["sun_z"], geometry["view_z"], raa, cwv_g_cm2, ozone)
        return read_shards(lut_root, keys)
    except OSError:
        # Store absent, or a transient filesystem error. Preloading is a pure
        # optimization: a worker's lazy read raises its own error later if the
        # store is actually needed and missing.
        return {}


def _worker_pool(workers, algorithm, correct, use_batch, wl_um, aero_profile, month, day,
                 clamp, lut_root=SHARD_ROOT, lut_axes=None, preload=None):
    """One ``Pool``, reused across the shipped-AOD and reference-AOD
    ``evaluate()`` calls. A no-op context manager when ``workers <= 1``.
    ``preload`` is installed into every worker's cache at startup via
    ``_pool_init``."""
    if workers <= 1:
        return contextlib.nullcontext(None)
    return mp.Pool(workers, initializer=_pool_init,
                   initargs=(algorithm, correct, use_batch, wl_um, aero_profile, month, day,
                             clamp, lut_root, lut_axes, preload))


@dataclass(frozen=True)
class SensitivityResult:
    """Everything one ``run_tanager``/``run_emit`` call produces: the two answers,
    plus enough to replot or re-slice either without recomputing."""

    rows: np.ndarray
    cols: np.ndarray
    shape: tuple[int, int]
    shipped_aod: np.ndarray
    cwv_g_cm2: np.ndarray
    clamped: np.ndarray
    at_shipped: np.ndarray
    at_reference: np.ndarray
    delta: np.ndarray
    label_shipped: np.ndarray | None
    label_reference: np.ndarray | None
    class_changed: np.ndarray | None
    curve_aod550: np.ndarray
    curves: dict[str, np.ndarray]
    selected: list[tuple[str, int]]
    reference_aod: float
    unit: str
    algorithm_name: str

    def value_map(self, values: np.ndarray) -> np.ndarray:
        """Scatter any per-pixel array aligned with ``rows``/``cols`` back onto
        the scene grid."""
        out = np.full(self.shape, np.nan)
        out[self.rows, self.cols] = values
        return out

    def delta_map(self) -> np.ndarray:
        return self.value_map(self.delta)

    def class_changed_map(self) -> np.ndarray | None:
        if self.class_changed is None:
            return None
        out = np.zeros(self.shape, dtype=bool)
        out[self.rows, self.cols] = self.class_changed
        return out

    def scored(self, scoring_region: np.ndarray) -> "SensitivityResult":
        """Restrict summary statistics to ``scoring_region`` (boolean,
        scene-shaped) without recomputing anything: the post-hoc slice. Never
        used to decide what gets processed, only what gets reported."""
        keep = scoring_region[self.rows, self.cols]
        return replace(
            self, rows=self.rows[keep], cols=self.cols[keep],
            shipped_aod=self.shipped_aod[keep], cwv_g_cm2=self.cwv_g_cm2[keep],
            clamped=self.clamped[keep], at_shipped=self.at_shipped[keep],
            at_reference=self.at_reference[keep], delta=self.delta[keep],
            label_shipped=None if self.label_shipped is None else self.label_shipped[keep],
            label_reference=None if self.label_reference is None else self.label_reference[keep],
            class_changed=None if self.class_changed is None else self.class_changed[keep],
        )


ADDITIVITY_BOUNDS = (0.33, 3.0)


@dataclass(frozen=True)
class VarianceFraction:
    """How much of a scene's real algorithm-output spread is genuine material
    diversity versus just the shipped-vs-reference AOD assumption, computed from
    Realized Sensitivity's own two-AOD-per-pixel data.

    ``atmosphere_fraction`` is NaN when ``reliable`` is False. Report ``coverage``
    instead. ``scene_variance`` and ``atmosphere_variance`` are always populated,
    so the raw ratio stays recoverable."""

    scene_variance: float
    atmosphere_variance: float
    atmosphere_fraction: float
    additivity: float
    reliable: bool
    coverage: float


def variance_fraction(at_reference: np.ndarray, delta: np.ndarray) -> VarianceFraction:
    """``atmosphere_fraction`` = the share of the scene's real output spread you
    would mistake for material diversity if you used the reference AOD instead of
    the shipped one: ``Var(delta) / (Var(at_reference) + Var(delta))``.
    Dimensionless, bounded [0, 1], with no per-pixel division.

    The ratio is unreliable when a few outlier pixels dominate every variance
    term at once (a raw-ratio algorithm crossing a near-zero denominator does
    this), which pins ``atmosphere_fraction`` near 0.5 regardless of the true
    balance. ``additivity = Var(at_reference + delta) / (Var(at_reference) +
    Var(delta))`` detects it: near 1 when ``delta`` and ``at_reference`` are
    roughly independent, far from 1 when they collapse onto one outlier
    population. Outside ``ADDITIVITY_BOUNDS`` the result carries ``reliable=False``
    and ``atmosphere_fraction`` is NaN.

    ``coverage`` = the fraction of pixels whose ``delta`` is within 3*MAD of the
    median. Report it in place of ``atmosphere_fraction`` when ``reliable`` is
    False. Always computed."""
    at_reference = np.asarray(at_reference, dtype=float)
    delta = np.asarray(delta, dtype=float)
    scene_var = float(np.var(at_reference))
    atmo_var = float(np.var(delta))
    total = scene_var + atmo_var
    fraction = atmo_var / total if total > 0 else float("nan")
    additivity = float(np.var(at_reference + delta) / total) if total > 0 else float("nan")
    reliable = bool(ADDITIVITY_BOUNDS[0] <= additivity <= ADDITIVITY_BOUNDS[1])
    mad = float(np.median(np.abs(delta - np.median(delta)))) if delta.size else float("nan")
    coverage = (float(np.mean(np.abs(delta - np.median(delta)) <= 3.0 * mad))
                if mad and np.isfinite(mad) and mad > 0 else float("nan"))
    return VarianceFraction(scene_var, atmo_var,
                            fraction if reliable else float("nan"),
                            additivity, reliable, coverage)


@dataclass(frozen=True)
class ReconstructionGap:
    """How far the LUT correction (at the shipped AOD) diverges from Tanager's own
    delivered ISOFIT reflectance for the same algorithm at the same pixels. Like
    ``SensitivityResult.delta``, ``gap`` is a plain difference between two
    algorithm outputs in the same units, so comparing ``median_abs_gap`` /
    ``p90_abs_gap`` against the same percentiles of ``abs(delta)`` shows whether
    an algorithm's AOD-driven sensitivity stands out against the LUT's own
    baseline reconstruction gap or is lost inside it."""

    gap: np.ndarray
    median_abs_gap: float
    p90_abs_gap: float


def reconstruction_gap(at_shipped: np.ndarray, at_isofit: np.ndarray) -> ReconstructionGap:
    """``at_isofit`` is the algorithm applied directly to Tanager's real
    delivered ``surface_reflectance`` (via ``tanager_ortho.reflectance_at``, no
    LUT involved) at the same pixels ``at_shipped`` was computed for."""
    at_shipped = np.asarray(at_shipped, dtype=float)
    at_isofit = np.asarray(at_isofit, dtype=float)
    gap = at_shipped - at_isofit
    finite = np.isfinite(gap)
    abs_gap = np.abs(gap[finite])
    return ReconstructionGap(gap=gap, median_abs_gap=float(np.median(abs_gap)),
                             p90_abs_gap=float(np.percentile(abs_gap, 90)))


def _select_representative(shipped_aod: np.ndarray,
                           percentiles: Sequence[int]) -> list[tuple[str, int]]:
    finite = np.flatnonzero(np.isfinite(shipped_aod))
    used: set[int] = set()
    selected: list[tuple[str, int]] = []
    for pct in percentiles:
        target = np.percentile(shipped_aod[finite], pct)
        for idx in finite[np.argsort(np.abs(shipped_aod[finite] - target))]:
            if idx not in used:
                used.add(int(idx))
                selected.append((f"P{pct:02d}", int(idx)))
                break
    return selected


def _curve_aod_values(shipped_aod, reference_aod, nodes, node_only, n_samples, extra_curve_aod=()):
    """``extra_curve_aod`` guarantees the sweep spans every value in it (for
    example every gathered AOD reference, not just the one ``reference_aod``
    driving the Realized Sensitivity delta). Without it, a reference plotted far
    from ``reference_aod`` sits off the end of the curve with nothing near it."""
    width = expected_error(reference_aod)
    bounds = [float(np.nanmin(shipped_aod)), float(np.nanmax(shipped_aod)),
              reference_aod - width, reference_aod + width, *extra_curve_aod]
    lower = max(nodes.min(), min(bounds))
    upper = min(nodes.max(), max(bounds))
    if node_only:
        return nodes[(nodes >= lower) & (nodes <= upper)]
    return np.unique(np.concatenate(
        (np.linspace(lower, upper, n_samples), [reference_aod, lower, upper],
         list(extra_curve_aod))))


def _run_from_arrays(
    algorithm: Algorithm,
    wl_nm: np.ndarray,
    radiance: np.ndarray,
    geometry: Mapping[str, np.ndarray],
    shipped_aod: np.ndarray,
    cwv_g_cm2: np.ndarray,
    aero_profile: str,
    reference_aod: float,
    acquisition: datetime.datetime,
    scene_shape: tuple[int, int],
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    group_labels: Mapping[str, str] | None = None,
    node_only: bool = False,
    percentiles: Sequence[int] = (5, 20, 50, 80, 95),
    n_curve_samples: int = 17,
    extra_curve_aod: Sequence[float] = (),
    unit: str = "",
    algorithm_name: str = "",
    lut_root: str = SHARD_ROOT,
    lut_axes: Mapping | None = None,
    correct: Callable[..., np.ndarray] | None = None,
    workers: int = 1,
    chunksize: int = 64,
) -> SensitivityResult:
    """The sensor-agnostic core: everything after a scene's pixels have been
    extracted. ``run_tanager``/``run_emit`` do the sensor-specific loading and
    call this with a resolved ``algorithm``, the pixel arrays, and the LUT the
    scene's sensor was tabulated against (``lut_root``/``lut_axes``).

    Runs ``algorithm`` at the shipped and reference AOD (Realized Sensitivity)
    and sweeps a handful of representative pixels across AOD (Potential
    Sensitivity). ``workers > 1`` parallelizes the two full-population passes.
    The curve sweep stays serial.
    """
    axes = lut_axes if lut_axes is not None else load_axes()
    nodes = np.asarray(axes["axes"]["aod"]["values"], dtype=float)
    shipped_eval = _snap_to_nodes(shipped_aod, nodes) if node_only else shipped_aod
    reference_eval = (float(_snap_to_nodes(np.array([reference_aod]), nodes)[0])
                      if node_only else float(reference_aod))

    use_batch = correct is None
    resolved_correct = correct if correct is not None else correct_spectrum_from_lut
    wl_um = np.asarray(wl_nm, dtype=float) / 1000.0
    preload = (_preload_for_pool(aero_profile, geometry, cwv_g_cm2, lut_root, lut_axes)
               if workers > 1 else None)
    with _worker_pool(workers, algorithm, resolved_correct, use_batch, wl_um, aero_profile,
                      acquisition.month, acquisition.day, True, lut_root, lut_axes,
                      preload=preload) as pool:
        raw_shipped, clamped = evaluate(
            algorithm, wl_nm, radiance, geometry, aero_profile, shipped_eval,
            acquisition.month, acquisition.day, cwv_g_cm2, correct=correct,
            lut_root=lut_root, lut_axes=lut_axes, chunksize=chunksize, pool=pool)
        raw_reference, _ = evaluate(
            algorithm, wl_nm, radiance, geometry, aero_profile, reference_eval,
            acquisition.month, acquisition.day, cwv_g_cm2, correct=correct,
            lut_root=lut_root, lut_axes=lut_axes, chunksize=chunksize, pool=pool)
    at_shipped, label_shipped_list = zip(*(_unpack(r) for r in raw_shipped))
    at_reference, label_reference_list = zip(*(_unpack(r) for r in raw_reference))
    at_shipped = np.asarray(at_shipped, dtype=float)
    at_reference = np.asarray(at_reference, dtype=float)
    delta = at_shipped - at_reference

    has_labels = any(label is not None for label in label_shipped_list)
    label_shipped = label_reference = class_changed = None
    if has_labels:
        label_shipped = _group(np.array(label_shipped_list, dtype=object), group_labels)
        label_reference = _group(np.array(label_reference_list, dtype=object), group_labels)
        class_changed = label_shipped != label_reference

    selected = _select_representative(shipped_aod, percentiles)
    curve_aod_values = _curve_aod_values(shipped_aod, reference_eval, nodes, node_only,
                                         n_curve_samples, extra_curve_aod)
    curves = {}
    for label, idx in selected:
        row_radiance = radiance[idx:idx + 1]
        row_geometry = {name: values[idx:idx + 1] for name, values in geometry.items()}
        row_cwv = cwv_g_cm2[idx:idx + 1]
        anchor_raw, _ = evaluate(algorithm, wl_nm, row_radiance, row_geometry, aero_profile,
                                 reference_eval, acquisition.month, acquisition.day,
                                 row_cwv, correct=correct, lut_root=lut_root, lut_axes=lut_axes,
                                 chunksize=chunksize)
        anchor_value, _ = _unpack(anchor_raw[0])
        curve_values = np.empty(len(curve_aod_values))
        for j, aod in enumerate(curve_aod_values):
            raw, _ = evaluate(algorithm, wl_nm, row_radiance, row_geometry, aero_profile,
                              float(aod), acquisition.month, acquisition.day, row_cwv,
                              correct=correct, lut_root=lut_root, lut_axes=lut_axes,
                              chunksize=chunksize)
            value, _ = _unpack(raw[0])
            curve_values[j] = value - anchor_value
        curves[label] = curve_values

    return SensitivityResult(
        rows=rows, cols=cols, shape=scene_shape, shipped_aod=shipped_aod,
        cwv_g_cm2=cwv_g_cm2, clamped=np.asarray(clamped, dtype=bool),
        at_shipped=at_shipped, at_reference=at_reference, delta=delta,
        label_shipped=label_shipped, label_reference=label_reference, class_changed=class_changed,
        curve_aod550=curve_aod_values, curves=curves, selected=selected,
        reference_aod=reference_aod, unit=unit, algorithm_name=algorithm_name,
    )


def _select_and_mask(valid, aoi):
    """(rows, cols) of the mask-passed pixels in scene coordinates, plus a shape
    check: the part of scene loading that is identical for every sensor."""
    r0, r1, c0, c1 = aoi
    if valid.shape != (r1 - r0, c1 - c0):
        raise ValueError(f"mask returned shape {valid.shape}, expected {(r1 - r0, c1 - c0)} "
                         f"(the AOI's own extent). A mask must return one boolean per AOI pixel.")
    local_rows, local_cols = np.nonzero(valid)
    if len(local_rows) == 0:
        raise ValueError("mask selected zero pixels inside the AOI")
    return local_rows + r0, local_cols + c0


def run_tanager(
    scene_id: str,
    aoi: tuple[int, int, int, int],
    mask: Mask,
    band_targets_nm: Sequence[float],
    reference_aod: float,
    aero_profile: str,
    *,
    algorithm: Algorithm | None = None,
    fit: Fit | None = None,
    group_labels: Mapping[str, str] | None = None,
    node_only: bool = False,
    percentiles: Sequence[int] = (5, 20, 50, 80, 95),
    n_curve_samples: int = 17,
    extra_curve_aod: Sequence[float] = (),
    unit: str = "",
    algorithm_name: str = "",
    cache=None,
    correct: Callable[..., np.ndarray] | None = None,
    workers: int = 1,
    chunksize: int = 64,
) -> SensitivityResult:
    """Potential/Realized Sensitivity for one Tanager scene: run ``algorithm``
    (or the algorithm ``fit`` derives) at every ``mask``-passed pixel in
    ``aoi``. See the module docstring for ``mask`` vs. ``scoring_region``,
    ``fit``'s radiance-domain requirement, and ``node_only``. Exactly one of
    ``algorithm``/``fit`` is required.

    ``cache`` is passed to ``tanager_ortho.scene_paths`` (a ``CacheConfig``, a
    path, or ``None`` for the default cache).
    """
    if (algorithm is None) == (fit is None):
        raise ValueError("pass exactly one of `algorithm` or `fit`")

    sr_path, l1_path = extract.scene_paths(scene_id, cache)
    with h5py.File(sr_path, "r") as sr, h5py.File(l1_path, "r") as l1:
        extract.validate_aoi(sr, aoi)
        valid = mask(sr, aoi)
        rows, cols = _select_and_mask(valid, aoi)
        scene_shape = sr[GRID + "aerosol_optical_depth"].shape

        wl_nm, radiance_block = extract.radiance_at(l1, band_targets_nm, aoi=aoi)
        geometry_block = extract.geometry(l1, aoi=aoi)
        shipped_aod_block = extract.shipped_aod(sr, aoi=aoi)
        cwv_block = extract.column_water_vapour(sr, aoi=aoi)
        radiance = radiance_block[valid]
        geometry = {name: values[valid] for name, values in geometry_block.items()}
        shipped_aod = shipped_aod_block[valid]
        cwv_g_cm2 = cwv_block[valid]

    acquisition = datetime.datetime.strptime(scene_id[:8], "%Y%m%d")
    if fit is not None:
        algorithm = fit(wl_nm, radiance)

    return _run_from_arrays(
        algorithm, wl_nm, radiance, geometry, shipped_aod, cwv_g_cm2, aero_profile,
        reference_aod, acquisition, scene_shape, rows, cols,
        group_labels=group_labels, node_only=node_only, percentiles=percentiles,
        n_curve_samples=n_curve_samples, extra_curve_aod=extra_curve_aod, unit=unit,
        algorithm_name=algorithm_name, correct=correct, workers=workers, chunksize=chunksize,
    )


def run_emit(*args, **kwargs) -> SensitivityResult:
    """Potential/Realized Sensitivity for one EMIT scene against the EMIT LUT.
    The cross-sensor counterpart of ``run_tanager``.

    Not available yet: it needs an ``emit.geometry()`` reader for the L1B OBS
    product and the bundled EMIT LUT (``axes_emit.json`` plus its shard store).
    """
    raise NotImplementedError(
        "run_emit is not available yet: it needs emit.geometry() (L1B OBS product) "
        "and the bundled EMIT LUT axes and shard store.")
