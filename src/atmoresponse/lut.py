"""Atmospheric-response LUT: axis definitions, the shard store, and the algebra
that turns a stored cell into surface reflectance (or back).

This module is the store half. numpy + stdlib only, so it stays importable in a
minimal environment; the 6S runs that fill the table are not part of the public
package.

What a cell holds. Six numbers per (key, aod, band), in two equivalent forms:

- ``xa``/``xb``/``xc`` -- 6S's own radiance-inversion coefficients::

      rho -> L :  y = rho / (1 - xc*rho) ;  L = (y + xb) / xa
      L -> rho :  y = xa*L - xb          ;  rho = y / (1 + xc*y)

- ``path``/``trans``/``sphalb`` -- the same physics in pure reflectance space::

      rho_toa = path + trans * rho / (1 - sphalb * rho)

  Carried alongside because the solver computes them first and because they are
  the date-independent half (see below).

**Earth-Sun distance is not an axis.** It scales TOA solar irradiance by 1/d^2
and nothing else -- it does not touch the atmosphere's optical properties. So
``path``/``trans``/``sphalb`` are entirely date-free; only ``xa``/``xb`` inherit a
date, because they fold the radiance<->reflectance conversion into the
atmospheric inversion. Cells are generated at ``REF_DOY`` and
``normalise_radiance()`` applies the scalar ``(d_obs/d_ref)**2`` at lookup.
"""
from __future__ import annotations

import glob
import itertools
import json
import math
import os
import re
import warnings
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

import numpy as np

# Shards relocate together via LUT_STORE. The public package does not ship the
# shard store itself; point this at a downloaded LUT archive.
LUT_STORE = os.environ.get("LUT_STORE", "lut_store")
SHARD_ROOT = os.path.join(LUT_STORE, "shards")

# Axes that key a shard (what generation parallelises over) vs. axes swept inside
# one shard. Adding a key axis writes new shards and rewrites nothing.
KEY_AXES = ("sza", "vza", "raa", "aerosol", "cwv", "ozone")
CELL_AXES = ("aod", "band")

# The six numbers a cell holds, named once: the solver returns exactly these
# keys, shards store exactly these arrays.
COEF_FIELDS = ("xa", "xb", "xc", "path", "trans", "sphalb")

# Optional, additive per-cell fields beyond COEF_FIELDS -- upward-only
# transmittance for a fire-emission forward model. NEVER required by read_shard;
# an older shard simply lacks them. Check presence (``"trans_up_gas" in shard``)
# before use.
EXTRA_FIELDS = ("trans_up_gas", "trans_up_scatter", "trans_up_ch4", "trans_up_water")

# A cell's status. UNATTEMPTED and FAILED are both NaN; separating them makes
# "what fraction of this shard failed?" answerable. Only STATUS_OK cells are ever
# read back as values.
STATUS_OK = 0
STATUS_UNATTEMPTED = 1
STATUS_FAILED = 2

# Simulation date all cells are generated at; see the module docstring.
REF_DOY = 182

# Matches the solver's ozone default. Duplicated, not imported, to keep this
# module free of the 6S dependency.
DEFAULT_OZONE_ATM_CM = 0.33

# Days elapsed before each month, non-leap; 6S takes month/day, the rest of this
# codebase takes a day of year.
_DAYS_BEFORE_MONTH = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365)


def doy_to_month_day(doy):
    for m in range(12):
        if doy <= _DAYS_BEFORE_MONTH[m + 1]:
            return m + 1, doy - _DAYS_BEFORE_MONTH[m]
    raise ValueError(f"day of year out of range: {doy}")


def month_day_to_doy(month, day):
    return _DAYS_BEFORE_MONTH[month - 1] + day


def earth_sun_distance_au(doy):
    """Earth-Sun distance in AU for a day of year (Spencer 1971 series)."""
    g = 2.0 * math.pi * (doy - 1) / 365.0
    e = (1.00011 + 0.034221 * math.cos(g) + 0.00128 * math.sin(g)
         + 0.000719 * math.cos(2 * g) + 0.000077 * math.sin(2 * g))
    return 1.0 / math.sqrt(e)


def normalise_radiance(L, doy, ref_doy=REF_DOY):
    """Rescale an observed radiance onto the LUT's reference Earth-Sun distance.

    Radiance goes as 1/d^2, so an observation at distance ``d_obs`` is equivalent
    to one at ``d_ref`` after multiplying by ``(d_obs/d_ref)**2``.
    """
    return np.asarray(L, dtype=float) * (earth_sun_distance_au(doy)
                                         / earth_sun_distance_au(ref_doy)) ** 2


def reflectance_from_radiance(xa, xb, xc, L):
    """Observed at-sensor radiance -> Lambertian surface reflectance."""
    y = np.asarray(xa, dtype=float) * np.asarray(L, dtype=float) - np.asarray(xb, dtype=float)
    return y / (1.0 + np.asarray(xc, dtype=float) * y)


def radiance_from_reflectance(xa, xb, xc, rho):
    """Lambertian surface reflectance -> at-sensor radiance (the inverse)."""
    rho = np.asarray(rho, dtype=float)
    y = rho / (1.0 - np.asarray(xc, dtype=float) * rho)
    return (y + np.asarray(xb, dtype=float)) / np.asarray(xa, dtype=float)


def toa_from_surface(path, trans, sphalb, rho):
    """Reflectance-space forward model -- the date-free form of the same cell."""
    rho = np.asarray(rho, dtype=float)
    return (np.asarray(path, dtype=float)
            + np.asarray(trans, dtype=float) * rho / (1.0 - np.asarray(sphalb, dtype=float) * rho))


@dataclass(frozen=True)
class CorrectionCoefficients:
    """One LUT cell's radiance-inversion coefficients, as a convenience wrapper
    around the bare ``xa``/``xb``/``xc`` module functions."""

    xa: float
    xb: float
    xc: float

    def reflectance_from_radiance(self, radiance):
        return reflectance_from_radiance(self.xa, self.xb, self.xc, radiance)

    def radiance_from_reflectance(self, reflectance):
        return radiance_from_reflectance(self.xa, self.xb, self.xc, reflectance)


# --------------------------------------------------------------------------- axes

_BUNDLED_AXES = {"tanager": "axes_tanager.json", "emit": "axes_emit.json"}


@lru_cache(maxsize=None)
def load_axes(path=None, *, sensor="tanager"):
    """Load the axis-definition file (index -> value, per axis).

    ``path=None`` reads the copy bundled with the package for ``sensor``
    (``"tanager"`` or ``"emit"``). Axis value lists are **append-only**: refining
    density adds values at the end, so an existing cell's integer key never
    changes meaning. Index order is therefore not value order, so always sort by
    value at lookup.

    Cached for the process lifetime. Call ``load_axes.cache_clear()`` if a test
    points ``path`` at a file that changes underfoot.
    """
    if path is None:
        text = files("atmoresponse.assets.lut").joinpath(_BUNDLED_AXES[sensor]).read_text()
        return json.loads(text)
    with open(path) as f:
        return json.load(f)


def axis_values(axes, name):
    return np.asarray(axes["axes"][name]["values"], dtype=float)


def band_table(axes):
    """(centre_nm, fwhm_nm) for every band index, as shipped by the sensor."""
    b = axes["axes"]["band"]
    return np.asarray(b["centre_nm"], dtype=float), np.asarray(b["fwhm_nm"], dtype=float)


def nearest_band_index(axes, wl_um):
    """The sensor's fixed band whose centre is closest to ``wl_um``. Never
    interpolated -- the sensor's band grid is a fixed property."""
    centre_nm, _ = band_table(axes)
    return int(np.argmin(np.abs(centre_nm - float(wl_um) * 1000.0)))


# -------------------------------------------------------------------------- store

SHARD_RE = re.compile(r"^shard_(\d+)\.npz$")


def _shard_name(shard_id):
    """Zero-padded to 9 digits; nothing depends on the width. Overflow is safe --
    the name grows a digit and ``SHARD_RE`` still parses it. Do not change the
    padding width on a populated store: the same numeric id at two widths is two
    files claiming one id, which ``scan_shards`` rejects."""
    return f"shard_{shard_id:09d}.npz"


def read_shard(path):
    """Read one shard. EXTRA_FIELDS keys are included only if present in the file
    -- check with ``"trans_up_gas" in shard``, don't assume presence the way
    COEF_FIELDS' presence can be assumed."""
    d = np.load(path, allow_pickle=False)
    key = {str(n): int(i) for n, i in zip(d["key_names"], d["key_idx"])}
    out = {f: d["path_refl" if f == "path" else f] for f in COEF_FIELDS}
    out.update({f: d[f] for f in EXTRA_FIELDS if f in d})
    out.update({"key": key, "aod_idx": d["aod_idx"], "band_idx": d["band_idx"],
                "status": d["status"],
                "provenance": json.loads(str(d["provenance"]))})
    return out


def scan_shards(root=SHARD_ROOT):
    """key tuple -> (shard_id, path), read from the shards themselves.

    This is the record; any manifest is only an index over it.
    """
    found = {}
    seen_ids = {}
    for path in sorted(glob.glob(os.path.join(root, "shard_*.npz"))):
        m = SHARD_RE.match(os.path.basename(path))
        if not m:
            raise ValueError(f"unparseable shard name: {path}")
        shard_id = int(m.group(1))
        if shard_id in seen_ids:
            raise ValueError(
                f"duplicate shard id {shard_id}: {seen_ids[shard_id]} and {path}. "
                f"Two files claim one id -- usually the padding width was changed "
                f"on a populated store. Ids must be unique; resolve by hand.")
        seen_ids[shard_id] = path
        d = np.load(path, allow_pickle=False)
        found[tuple(int(i) for i in d["key_idx"])] = (shard_id, path)
    return found


# ----------------------------------------------------------------------- consumer
#
# Nothing above this line reads more than one shard. This section turns an
# arbitrary (geometry, atmosphere, aod, band) query into interpolated
# coefficients, then into surface reflectance.

# Continuous KEY axes, each bracketed and interpolated at lookup. The ``aerosol``
# axis is excluded: it names a discrete 6S aerosol *model* (not AOD, not a
# continuum), matched exactly and handled separately in ``lookup()``. See that
# function's docstring on why the axis is discrete and what a continuous version
# would take.
GEOMETRY_AXES = ("sza", "vza", "raa", "cwv", "ozone")

# Interpolation is two independent knobs per axis:
#   x-space -- the coordinate a lookup interpolates ALONG (theta vs cos theta;
#              aod vs log aod). Not the same as where the grid points sit.
#   y-space -- whether the coefficient or its log is the interpolated quantity
#              (per COEF_FIELDS field, falling back to linear where a value is
#              not strictly positive -- see ``_lerp_fields``).
# Measured, crossing both knobs, at the grid's own values: sza/vza/raa want
# cos + log; aod wants log + log; cwv wants linear + linear. Never extend one
# axis's verdict to another. ``ozone`` (a single tabulated value) is kept
# linear + linear as the conservative default.
_AXIS_XVAR = {
    "sza": lambda v: math.cos(math.radians(v)),
    "vza": lambda v: math.cos(math.radians(v)),
    "raa": lambda v: math.cos(math.radians(v)),
    "cwv": lambda v: v,
    "ozone": lambda v: v,
    "aod": lambda v: math.log(v),
}
_AXIS_LOG = {"sza": True, "vza": True, "raa": True, "cwv": False, "ozone": False, "aod": True}

# Vectorized twin of ``_AXIS_XVAR`` (same x-space per axis, np-array-safe) for the
# batch lookup path -- kept separate rather than generalizing the scalar dict, the
# same reason ``fold_raa_array`` is separate from ``fold_raa``.
_AXIS_XVAR_NP = {
    "sza": lambda v: np.cos(np.radians(v)),
    "vza": lambda v: np.cos(np.radians(v)),
    "raa": lambda v: np.cos(np.radians(v)),
    "cwv": lambda v: np.asarray(v, dtype=float),
    "ozone": lambda v: np.asarray(v, dtype=float),
    "aod": lambda v: np.log(v),
}

_SHARD_CACHE = {}
_CWV_LEVEL_CACHE = {}
_VZA_LEVEL_CACHE = {}
_SINGLE_CWV_WARNING_KEYS = set()


def clear_shard_cache():
    """Drop cached shard reads. Call this if the store changes underfoot -- new
    shards are being written while this module reads the same root, or a test
    points ``root`` at a scratch store and back."""
    _SHARD_CACHE.clear()
    _CWV_LEVEL_CACHE.clear()
    _VZA_LEVEL_CACHE.clear()
    _SINGLE_CWV_WARNING_KEYS.clear()
    _CLAMP_WARNING_KEYS.clear()


def _sorted_axis(axes, name):
    """[(value, original_index), ...] ascending by value.

    Axis value lists are append-only (see ``load_axes``), so index order is not
    value order -- every bracket has to sort first.
    """
    vals = axes["axes"][name]["values"]
    order = sorted(range(len(vals)), key=lambda i: float(vals[i]))
    return [(float(vals[i]), i) for i in order]


_CLAMP_WARNING_KEYS = set()


def _warn_clamp(name, requested, held_at):
    """One RuntimeWarning per (axis, direction) that a query was held at a
    boundary node. Deduped so a scene-scale population does not emit one warning
    per pixel."""
    key = (name, requested > held_at)
    if key not in _CLAMP_WARNING_KEYS:
        warnings.warn(
            f"{name}={requested} is outside the LUT's tabulated range; held at the "
            f"nearest node {held_at}. clamp=True was requested.",
            RuntimeWarning, stacklevel=3,
        )
        _CLAMP_WARNING_KEYS.add(key)


def _bracket(sorted_pairs, x):
    """(lo_idx, hi_idx, lo_val, hi_val) bracketing x; ``lo_idx == hi_idx`` at an
    exact grid value (or a single-point axis), so interpolation reduces to
    identity there -- required by the rule that narrow-differencing algorithms
    must be read at tabulated AOD values.

    Raises outside the tabulated range: this is the strict backstop. A caller
    that wants a query beyond the grid served rather than refused passes
    ``clamp=True`` to a ``lookup*``/``correct_*`` entry point, which snaps the
    value to the nearest boundary node (a held edge cell, never a linearly
    extrapolated one) and warns before this function is reached.
    """
    x = float(x)
    lo_v, hi_v = sorted_pairs[0][0], sorted_pairs[-1][0]
    if x < lo_v - 1e-9 or x > hi_v + 1e-9:
        raise ValueError(f"{x} is outside the LUT's tabulated range [{lo_v}, {hi_v}]")
    for v, i in sorted_pairs:
        if abs(v - x) < 1e-9:
            return i, i, v, v
    for (v0, i0), (v1, i1) in zip(sorted_pairs, sorted_pairs[1:]):
        if v0 < x < v1:
            return i0, i1, v0, v1
    raise ValueError(f"could not bracket {x}")  # unreachable given the range check


def _lerp_fields(lo, hi, w, log_axis):
    """Blend two coefficient dicts at weight w in [0, 1]. Per COEF_FIELDS field,
    not per cell -- log interpolation is only safe where both endpoints are
    strictly positive, and that can differ field by field."""
    if w == 0.0:
        return lo
    out = {}
    for f in COEF_FIELDS:
        a, b = lo[f], hi[f]
        if log_axis and a > 0 and b > 0:
            out[f] = math.exp(math.log(a) * (1.0 - w) + math.log(b) * w)
        else:
            out[f] = a * (1.0 - w) + b * w
    return out


def _lerp_array_fields(lo, hi, w, log_axis):
    """Array-valued counterpart of ``_lerp_fields`` for a band vector.

    ``w`` may be a scalar (the per-pixel ``lookup_spectrum`` path) or a per-pixel
    weight array broadcastable against the ``(n_pixels, n_bands)`` field arrays
    (the ``lookup_spectrum_batch`` path). The scalar-zero fast path is kept for
    the former; an array ``w`` -- even all-zero -- goes through the blend, which
    is numerically identical at ``w == 0``."""
    if np.ndim(w) == 0 and w == 0.0:
        return lo
    out = {}
    for f in COEF_FIELDS:
        a, b = lo[f], hi[f]
        linear = a * (1.0 - w) + b * w
        if log_axis:
            positive = (a > 0) & (b > 0)
            a_safe = np.where(positive, a, 1.0)
            b_safe = np.where(positive, b, 1.0)
            logarithmic = np.exp(np.log(a_safe) * (1.0 - w) + np.log(b_safe) * w)
            out[f] = np.where(positive, logarithmic, linear)
        else:
            out[f] = linear
    return out


def _cached_shard(root, key_idx):
    """Read (and cache) the shard at an exact KEY_AXES index tuple.

    ``scan_shards`` re-globs every call rather than being cached itself: the
    store may be gaining shards while this module reads it, and a stale directory
    listing would silently hide newly finished geometry. Re-globbing is cheap;
    the shard payloads are what is worth caching.
    """
    key_tuple = tuple(key_idx[a] for a in KEY_AXES)
    cache_key = (root, key_tuple)
    if cache_key not in _SHARD_CACHE:
        found = scan_shards(root)
        if key_tuple not in found:
            raise ValueError(
                f"no shard for key {dict(zip(KEY_AXES, key_tuple))} under {root} "
                f"-- not generated yet ({len(found)} shard(s) on disk)")
        shard = read_shard(found[key_tuple][1])
        shard["_aod_pos"] = {int(a): p for p, a in enumerate(shard["aod_idx"])}
        shard["_band_pos"] = {int(b): p for p, b in enumerate(shard["band_idx"])}
        _SHARD_CACHE[cache_key] = shard
    return _SHARD_CACHE[cache_key]


def _populated_cwv_indices(root, aerosol_idx):
    """CWV levels currently present for one aerosol model in ``root``.

    The directory mtime keeps this cheap for ordinary repeated lookups while
    still noticing a newly written shard. A partly generated second CWV level is
    deliberately visible immediately: lookup must then require its normal
    interpolation corner rather than silently hold the first level.
    """
    marker = os.stat(root).st_mtime_ns
    cache_key = (root, aerosol_idx)
    cached = _CWV_LEVEL_CACHE.get(cache_key)
    if cached is not None and cached[0] == marker:
        return cached[1]

    aerosol_pos = KEY_AXES.index("aerosol")
    cwv_pos = KEY_AXES.index("cwv")
    levels = tuple(sorted({key[cwv_pos] for key in scan_shards(root)
                           if key[aerosol_pos] == aerosol_idx}))
    _CWV_LEVEL_CACHE[cache_key] = (marker, levels)
    return levels


def _populated_vza_indices(root, aerosol_idx, cwv_idx):
    """VZA levels with complete-in-practice shard support for one LUT slice.

    A sparse run can generate only a VZA subset while retaining planned
    intermediate axis values. Those planned values must bracket across the
    populated subset rather than demand a nonexistent exact shard.
    """
    marker = os.stat(root).st_mtime_ns
    cache_key = (root, aerosol_idx, cwv_idx)
    cached = _VZA_LEVEL_CACHE.get(cache_key)
    if cached is not None and cached[0] == marker:
        return cached[1]

    aerosol_pos = KEY_AXES.index("aerosol")
    cwv_pos = KEY_AXES.index("cwv")
    vza_pos = KEY_AXES.index("vza")
    levels = tuple(sorted({key[vza_pos] for key in scan_shards(root)
                           if key[aerosol_pos] == aerosol_idx and key[cwv_pos] == cwv_idx}))
    _VZA_LEVEL_CACHE[cache_key] = (marker, levels)
    return levels


def _geometry_specs(root, aero_idx, aerosol, axes, query, clamp=False):
    """(name, lo_idx, hi_idx, weight, is_log) for each of GEOMETRY_AXES at one query point.

    cwv and vza both bracket against only the levels actually POPULATED for this
    aerosol model / cwv slice, never the full declared axis. The axes file is
    append-only, so an abandoned planned point stays declared forever with zero
    shards, and bracketing against it would raise on every query that does not
    land exactly on a populated node. One shared helper (used by both ``lookup``
    and ``lookup_spectrum``) so the two cannot diverge.

    ``clamp`` is threaded to each axis's ``_axis_spec``. The single-populated-CWV
    branch already holds out-of-range CWV at the one level regardless of
    ``clamp``.
    """
    cwv_indices = _populated_cwv_indices(root, aero_idx)
    if not cwv_indices:
        raise ValueError(f"no shards generated yet for aerosol {aerosol!r}")
    cwv_values = axes["axes"]["cwv"]["values"]
    if len(cwv_indices) == 1:
        level = cwv_indices[0]
        level_value = cwv_values[level]
        if abs(float(query["cwv"]) - float(level_value)) > 1e-9:
            warning_key = (root, aero_idx, level)
            if warning_key not in _SINGLE_CWV_WARNING_KEYS:
                warnings.warn(
                    f"LUT has only CWV={level_value} populated for aerosol {aerosol!r}; "
                    f"holding requested CWV={query['cwv']} at that level rather than "
                    "interpolating a missing bracket.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                _SINGLE_CWV_WARNING_KEYS.add(warning_key)
        cwv_spec = ("cwv", level, level, 0.0, _AXIS_LOG["cwv"])
    else:
        cwv_pairs = [(float(cwv_values[i]), i) for i in cwv_indices]
        cwv_spec = ("cwv", *_axis_spec(axes, "cwv", query["cwv"], cwv_pairs, clamp=clamp))

    cwv_query_indices = {cwv_spec[1], cwv_spec[2]}
    vza_sets = [set(_populated_vza_indices(root, aero_idx, idx)) for idx in cwv_query_indices]
    vza_levels = tuple(sorted(set.intersection(*vza_sets))) if vza_sets else ()

    specs = []
    for name in GEOMETRY_AXES:
        if name == "cwv":
            specs.append(cwv_spec)
        elif name == "vza" and vza_levels:
            values = axes["axes"]["vza"]["values"]
            pairs = [(float(values[index]), index) for index in vza_levels]
            specs.append((name, *_axis_spec(axes, name, query[name], pairs, clamp=clamp)))
        else:
            specs.append((name, *_axis_spec(axes, name, query[name], clamp=clamp)))
    return specs


def shard_keys_needed(root, aero_idx, aerosol, axes, sza, vza, raa, cwv, ozone):
    """Every KEY_AXES shard combination a population of geometry queries could
    touch -- the cartesian product of each non-degenerate axis's bracket corners
    (``_geometry_specs``), unioned over every query. Read-only: never reads a
    shard payload, so this is cheap to run over a whole scene's per-pixel
    geometry.

    Built for the sensitivity module's multi-worker preload: a fresh worker
    process's ``_SHARD_CACHE`` starts empty, and letting each one discover shards
    independently both loses the warm-cache speedup a serial run gets on
    spatially-coherent pixels and adds disk contention. For a single
    scene/AOI/aerosol the real touched set is typically a handful of shards.
    Callers should read the result with ``read_shards()`` in the main process
    once and hand it to each worker via ``install_shards()``.

    Query arrays are rounded before deduplication (3 dp for sza/vza/raa, 4 for
    cwv/ozone): real per-pixel geometry varies continuously, so without rounding
    almost every pixel would count as a distinct query even though most bracket
    identically.
    """
    sza = np.asarray(sza, dtype=float)
    vza = np.asarray(vza, dtype=float)
    raa = np.asarray(raa, dtype=float)
    cwv = np.asarray(cwv, dtype=float)
    ozone = np.broadcast_to(np.asarray(ozone, dtype=float), sza.shape)
    combos = {
        (round(float(s), 3), round(float(v), 3), round(float(r), 3),
         round(float(c), 4), round(float(o), 4))
        for s, v, r, c, o in zip(sza, vza, raa, cwv, ozone)
    }
    keys = set()
    for s, v, r, c, o in combos:
        try:
            specs = _geometry_specs(root, aero_idx, aerosol, axes,
                                    {"sza": s, "vza": v, "raa": r, "cwv": c, "ozone": o})
        except ValueError:
            # Out of the tabulated range, or nothing generated yet for this
            # aerosol -- best-effort preload, so skip rather than abort. The real
            # per-pixel lookup still raises its own clear error later if this
            # combination turns out to be genuinely needed and genuinely invalid.
            continue
        axis_options = [{lo_i, hi_i} for _name, lo_i, hi_i, _w, _log in specs]
        for combo in itertools.product(*axis_options):
            values = dict(zip(GEOMETRY_AXES, combo))
            values["aerosol"] = aero_idx
            keys.add(tuple(values[a] for a in KEY_AXES))
    return keys


def read_shards(root, key_tuples):
    """Read every shard named by ``key_tuples`` (KEY_AXES order) fresh from disk,
    independent of ``_SHARD_CACHE`` -- returns a plain ``{key_tuple: shard}`` dict
    a caller can hand to another process's cache via ``install_shards()``. A key
    with no shard on disk is silently skipped (this function is a preload
    optimization, never load-bearing for correctness)."""
    found = scan_shards(root)
    out = {}
    for key_tuple in key_tuples:
        if key_tuple not in found:
            continue
        shard = read_shard(found[key_tuple][1])
        shard["_aod_pos"] = {int(a): p for p, a in enumerate(shard["aod_idx"])}
        shard["_band_pos"] = {int(b): p for p, b in enumerate(shard["band_idx"])}
        out[key_tuple] = shard
    return out


def install_shards(root, shards):
    """Populate this process's ``_SHARD_CACHE`` from an already-read shard dict
    (``read_shards``'s output) -- the worker side of preloading, called once per
    worker at startup instead of each worker discovering shards lazily from disk.
    A no-op for any key already cached."""
    for key_tuple, shard in shards.items():
        _SHARD_CACHE.setdefault((root, key_tuple), shard)


def _shard_cell(shard, aod_idx, band_idx):
    """One (aod_idx, band_idx) cell's coefficients from an already-read shard.

    Raises rather than returning the stored NaN for an unattempted or failed
    cell -- a silently propagated NaN would be indistinguishable from "this band
    happens to be near zero" once it reaches an algorithm.
    """
    if aod_idx not in shard["_aod_pos"] or band_idx not in shard["_band_pos"]:
        raise ValueError(f"shard for key {shard['key']} does not cover "
                         f"aod_idx={aod_idx}, band_idx={band_idx}")
    i, j = shard["_aod_pos"][aod_idx], shard["_band_pos"][band_idx]
    status = int(shard["status"][i, j])
    if status != STATUS_OK:
        raise ValueError(f"cell (aod_idx={aod_idx}, band_idx={band_idx}) at key "
                         f"{shard['key']} is not usable (status={status}: "
                         + ("unattempted" if status == STATUS_UNATTEMPTED else "6S failed")
                         + ")")
    return {f: float(shard[f][i, j]) for f in COEF_FIELDS}


def _combine_axes(leaf, specs, interpolate=_lerp_fields):
    """Recursively interpolate one KEY axis at a time.

    ``specs`` is a list of (name, lo_idx, hi_idx, weight, is_log) for the axes
    still to resolve; ``leaf(idx_dict)`` returns the coefficient dict once all of
    them are pinned to a single index. This is the standard multilinear
    extension of the per-axis rule. Treat the composed error as bounded by the
    worst single axis involved, not verified tighter.
    """
    if not specs:
        return leaf({})
    name, lo_i, hi_i, w, log_axis = specs[0]
    rest = specs[1:]
    if lo_i == hi_i:
        return _combine_axes(lambda idx, _lo=lo_i: leaf({name: _lo, **idx}), rest, interpolate)
    c_lo = _combine_axes(lambda idx, _lo=lo_i: leaf({name: _lo, **idx}), rest, interpolate)
    c_hi = _combine_axes(lambda idx, _hi=hi_i: leaf({name: _hi, **idx}), rest, interpolate)
    return interpolate(c_lo, c_hi, w, log_axis)


def _axis_spec(axes, name, value, sorted_pairs=None, clamp=False):
    """(lo_idx, hi_idx, weight, is_log) for one axis at one query value -- the
    single place ``_AXIS_XVAR``/``_AXIS_LOG`` get applied, so a KEY axis and the
    CELL axis (aod) interpolate through identical code. ``weight`` is computed in
    the axis's own x-space, never in the raw value.

    ``clamp=True`` snaps a query beyond the axis's own range to the nearest
    boundary value (held edge cell, ``weight`` 0) and warns, instead of letting
    ``_bracket`` raise.
    """
    pairs = sorted_pairs or _sorted_axis(axes, name)
    if clamp:
        lo_bound, hi_bound = pairs[0][0], pairs[-1][0]
        if value < lo_bound - 1e-9 or value > hi_bound + 1e-9:
            held = lo_bound if value < lo_bound else hi_bound
            _warn_clamp(name, value, held)
            value = held
    lo_i, hi_i, lo_v, hi_v = _bracket(pairs, value)
    if lo_i == hi_i:
        return lo_i, hi_i, 0.0, _AXIS_LOG[name]
    xvar = _AXIS_XVAR[name]
    w = (xvar(value) - xvar(lo_v)) / (xvar(hi_v) - xvar(lo_v))
    return lo_i, hi_i, w, _AXIS_LOG[name]


def lookup(sza, vza, raa, aerosol, cwv, ozone, aod, band_idx, root=SHARD_ROOT, axes=None,
           clamp=False):
    """Interpolated {xa, xb, xc, path, trans, sphalb} at an arbitrary point
    inside the LUT's tabulated grid.

    ``aerosol`` selects a named 6S aerosol *model* (e.g. "Maritime",
    "Continental") and must match a tabulated name exactly. The axis is a
    discrete set of named models, not a continuum, so it is matched rather than
    bracketed. The models are themselves fixed mixtures of a few fundamental
    components, so a continuous aerosol axis is possible in principle by
    tabulating custom component blends -- a possible future expansion, not what
    this axis is today. ``band_idx`` is exact too (the sensor's fixed band grid
    is never interpolated -- resolve a wavelength with ``nearest_band_index``).
    Every other argument brackets and interpolates.

    Raises ValueError on a coordinate outside the tabulated range (unless
    ``clamp=True``, which holds it at the nearest boundary node and warns), an
    ungenerated shard, or an unsolved cell -- never returns a silent NaN.
    """
    axes = axes or load_axes()
    aerosol_values = axes["axes"]["aerosol"]["values"]
    if aerosol not in aerosol_values:
        raise ValueError(f"aerosol {aerosol!r} is not a tabulated model: {aerosol_values}")
    aero_idx = aerosol_values.index(aerosol)

    query = {"sza": sza, "vza": vza, "raa": raa, "cwv": cwv, "ozone": ozone}
    specs = _geometry_specs(root, aero_idx, aerosol, axes, query, clamp=clamp)
    aod_lo, aod_hi, aod_w, aod_log = _axis_spec(axes, "aod", aod, clamp=clamp)

    def leaf(idx):
        key = {**idx, "aerosol": aero_idx}
        shard = _cached_shard(root, key)
        c_lo = _shard_cell(shard, aod_lo, band_idx)
        c_hi = c_lo if aod_hi == aod_lo else _shard_cell(shard, aod_hi, band_idx)
        return _lerp_fields(c_lo, c_hi, aod_w, log_axis=aod_log)

    return _combine_axes(leaf, specs)


def lookup_spectrum(sza, vza, raa, aerosol, cwv, ozone, aod, band_idx,
                    root=SHARD_ROOT, axes=None, clamp=False):
    """Interpolated coefficients for multiple exact band indices.

    Geometry and atmosphere remain scalar: this accelerates one pixel's full
    spectrum without averaging over the real pixel-to-pixel angle fields. It is
    equivalent to calling ``lookup`` once per supplied index. ``clamp`` behaves
    as in ``lookup``.
    """
    axes = axes or load_axes()
    band_idx = np.asarray(band_idx, dtype=int)
    if band_idx.ndim != 1:
        raise ValueError("band_idx must be one-dimensional")
    if np.any(band_idx < 0) or np.any(band_idx >= len(axes["axes"]["band"]["values"])):
        raise ValueError("band_idx contains an out-of-range band")

    aerosol_values = axes["axes"]["aerosol"]["values"]
    if aerosol not in aerosol_values:
        raise ValueError(f"aerosol {aerosol!r} is not a tabulated model: {aerosol_values}")
    aero_idx = aerosol_values.index(aerosol)
    query = {"sza": sza, "vza": vza, "raa": raa, "cwv": cwv, "ozone": ozone}
    specs = _geometry_specs(root, aero_idx, aerosol, axes, query, clamp=clamp)
    aod_lo, aod_hi, aod_w, aod_log = _axis_spec(axes, "aod", aod, clamp=clamp)

    def cell(shard, aod_idx):
        positions = np.array([shard["_band_pos"].get(int(index), -1) for index in band_idx])
        if np.any(positions < 0):
            missing = band_idx[positions < 0].tolist()
            raise ValueError(f"shard is missing band indices {missing}")
        row = shard["_aod_pos"].get(aod_idx)
        if row is None:
            raise ValueError(f"shard is missing aod_idx={aod_idx}")
        status = shard["status"][row, positions]
        if np.any(status != STATUS_OK):
            bad = band_idx[status != STATUS_OK].tolist()
            raise ValueError(f"LUT has unusable cells at aod_idx={aod_idx}, bands={bad}")
        return {f: shard[f][row, positions] for f in COEF_FIELDS}

    def leaf(idx):
        shard = _cached_shard(root, {**idx, "aerosol": aero_idx})
        c_lo = cell(shard, aod_lo)
        c_hi = c_lo if aod_hi == aod_lo else cell(shard, aod_hi)
        return _lerp_array_fields(c_lo, c_hi, aod_w, aod_log)

    return _combine_axes(leaf, specs, interpolate=_lerp_array_fields)


def fold_raa(view_a, sun_a):
    """Two absolute azimuths -> the LUT's relative-azimuth axis, in [0, 180].

    Cells are generated at solar_azimuth=0; the (solar_a=0, view_a=raa)
    reduction was verified exact for absolute-azimuth pairs sharing a relative
    angle, which is what makes this fold valid rather than an approximation.
    """
    d = abs(float(view_a) - float(sun_a)) % 360.0
    return 360.0 - d if d > 180.0 else d


def fold_raa_array(view_a, sun_a):
    """Vectorized twin of ``fold_raa`` -- same formula, array-safe. Kept separate
    rather than generalizing ``fold_raa`` itself: that function's scalar contract
    is relied on elsewhere, and ``np.where`` on a scalar condition returns a 0-d
    array, not a plain float."""
    d = np.abs(np.asarray(view_a, dtype=float) - np.asarray(sun_a, dtype=float)) % 360.0
    return np.where(d > 180.0, 360.0 - d, d)


def correct_from_lut(sun_z, sun_a, view_z, view_a, month, day, aero_profile,
                     aot550, cwv_g_cm2, wl_um, L_obs, ozone=DEFAULT_OZONE_ATM_CM,
                     root=SHARD_ROOT, axes=None, clamp=False):
    """LUT-backed radiance -> surface reflectance for one band, one pixel.

    ``aero_profile`` is a LUT axis-value string (e.g. "Maritime"). Two things a
    real 6S run gets for free that this does explicitly: relative azimuth (the
    LUT keys on raa -- ``fold_raa``) and Earth-Sun distance (cells are generated
    at ``REF_DOY``; ``normalise_radiance`` rescales the observed radiance onto it
    before inversion). ``clamp`` behaves as in ``lookup``.
    """
    axes = axes or load_axes()
    raa = fold_raa(view_a, sun_a)
    doy = month_day_to_doy(month, day)
    L_ref = float(normalise_radiance(L_obs, doy))
    band_idx = nearest_band_index(axes, wl_um)
    c = lookup(sun_z, view_z, raa, aero_profile, cwv_g_cm2, ozone, aot550,
               band_idx, root=root, axes=axes, clamp=clamp)
    return float(reflectance_from_radiance(c["xa"], c["xb"], c["xc"], L_ref))


def correct_array_from_lut(sun_z, sun_a, view_z, view_a, month, day, aero_profile,
                           aot550, cwv_g_cm2, wl_um, L_obs, ozone=DEFAULT_OZONE_ATM_CM,
                           root=SHARD_ROOT, axes=None, clamp=False):
    """``correct_from_lut``, vectorized over many pixels sharing one
    geometry/aod/band.

    ``L_obs`` is array-like; every other argument is scalar. ``lookup()``'s
    coefficients don't depend on ``L_obs``, so they are computed once here rather
    than once per pixel. Not a fit when geometry genuinely varies per pixel --
    there, use ``correct_spectrum_batch_from_lut``. ``clamp`` behaves as in
    ``lookup``.
    """
    axes = axes or load_axes()
    raa = fold_raa(view_a, sun_a)
    doy = month_day_to_doy(month, day)
    L_ref = normalise_radiance(np.asarray(L_obs, dtype=float), doy)
    band_idx = nearest_band_index(axes, wl_um)
    c = lookup(sun_z, view_z, raa, aero_profile, cwv_g_cm2, ozone, aot550,
               band_idx, root=root, axes=axes, clamp=clamp)
    return reflectance_from_radiance(c["xa"], c["xb"], c["xc"], L_ref)


def correct_spectrum_from_lut(sun_z, sun_a, view_z, view_a, month, day, aero_profile,
                              aot550, cwv_g_cm2, wl_um, L_obs,
                              ozone=DEFAULT_OZONE_ATM_CM, root=SHARD_ROOT, axes=None,
                              clamp=False):
    """Correct one pixel's multi-band spectrum with vectorized LUT lookup.

    ``wl_um`` and ``L_obs`` must be matching one-dimensional arrays. The pixel
    retains its own geometry and atmospheric inputs; only the independent-band
    coefficient lookup and algebra are vectorized. ``clamp`` behaves as in
    ``lookup``.
    """
    axes = axes or load_axes()
    wavelengths = np.asarray(wl_um, dtype=float)
    radiance = np.asarray(L_obs, dtype=float)
    if wavelengths.ndim != 1 or radiance.ndim != 1 or wavelengths.shape != radiance.shape:
        raise ValueError("wl_um and L_obs must be matching one-dimensional arrays")
    raa = fold_raa(view_a, sun_a)
    doy = month_day_to_doy(month, day)
    band_idx = np.array([nearest_band_index(axes, wavelength) for wavelength in wavelengths])
    c = lookup_spectrum(sun_z, view_z, raa, aero_profile, cwv_g_cm2, ozone, aot550,
                        band_idx, root=root, axes=axes, clamp=clamp)
    L_ref = normalise_radiance(radiance, doy)
    return reflectance_from_radiance(c["xa"], c["xb"], c["xc"], L_ref)


# ------------------------------------------ batch (per-pixel-geometry) lookup
#
# ``lookup_spectrum`` / ``correct_spectrum_from_lut`` vectorize over BANDS for
# one pixel; ``correct_array_from_lut`` vectorizes over PIXELS but only for one
# shared geometry. This layer covers the remaining case: N pixels each with
# their OWN geometry/CWV/AOD, full spectrum. Pixels are grouped by their discrete
# interpolation-bracket signature (which grid cell each axis falls in) and each
# group runs the same ``_combine_axes`` recursion the scalar path uses, with the
# per-axis blend weight carried as a per-pixel array.


def _axis_spec_batch(axes, name, values, sorted_pairs=None, clamp=False):
    """Vectorized ``_axis_spec``: (lo_idx[], hi_idx[], weight[], out_of_range[])
    for an array of query values on one axis. Same bracketing rule and x-space as
    the scalar form. Raises if ANY value is outside the range, unless
    ``clamp=True``, which snaps those values to the nearest boundary, warns once,
    and flags them in ``out_of_range``."""
    pairs = _sorted_axis(axes, name) if sorted_pairs is None else sorted(sorted_pairs)
    grid_v = np.array([v for v, _ in pairs], dtype=float)
    grid_i = np.array([i for _, i in pairs], dtype=int)
    values = np.asarray(values, dtype=float)
    lo_bound, hi_bound = grid_v[0], grid_v[-1]
    outside = (values < lo_bound - 1e-9) | (values > hi_bound + 1e-9)
    if np.any(outside):
        if not clamp:
            bad = np.unique(values[outside])[:3].tolist()
            raise ValueError(f"{name} value(s) {bad} outside the LUT's tabulated range "
                             f"[{lo_bound}, {hi_bound}]")
        below = values < lo_bound - 1e-9
        above = values > hi_bound + 1e-9
        if np.any(below):
            _warn_clamp(name, float(values[below].min()), lo_bound)
        if np.any(above):
            _warn_clamp(name, float(values[above].max()), hi_bound)
        values = np.clip(values, lo_bound, hi_bound)
    if len(grid_v) == 1:
        z = np.zeros(len(values), dtype=int)
        return grid_i[z], grid_i[z], np.zeros(len(values)), outside

    pos = np.clip(np.searchsorted(grid_v, values, side="right") - 1, 0, len(grid_v) - 2)
    lo_idx, hi_idx = grid_i[pos], grid_i[pos + 1]
    lo_val, hi_val = grid_v[pos], grid_v[pos + 1]
    xvar = _AXIS_XVAR_NP[name]
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = (xvar(values) - xvar(lo_val)) / (xvar(hi_val) - xvar(lo_val))

    on_lo = np.abs(values - lo_val) < 1e-9
    on_hi = np.abs(values - hi_val) < 1e-9
    on_node = on_lo | on_hi
    node_idx = np.where(on_hi, hi_idx, lo_idx)
    lo_idx = np.where(on_node, node_idx, lo_idx)
    hi_idx = np.where(on_node, node_idx, hi_idx)
    weight = np.where(on_node, 0.0, weight)
    return lo_idx, hi_idx, weight, outside


def _geometry_specs_batch(root, aero_idx, aerosol, axes, sza, vza, raa, cwv, ozone,
                          clamp=False):
    """Per-pixel ``(name, lo_idx[], hi_idx[], weight[], is_log)`` for each
    GEOMETRY_AXES entry plus a per-pixel ``out_of_range`` mask -- the array
    counterpart of ``_geometry_specs``, mirroring its populated-CWV/VZA-levels
    logic and its one-level-CWV RuntimeWarning. ``clamp`` is threaded to each
    axis's ``_axis_spec_batch``."""
    n = len(sza)
    cwv = np.asarray(cwv, dtype=float)
    cwv_indices = _populated_cwv_indices(root, aero_idx)
    if not cwv_indices:
        raise ValueError(f"no shards generated yet for aerosol {aerosol!r}")
    cwv_values = axes["axes"]["cwv"]["values"]
    geo_oor = np.zeros(n, dtype=bool)

    if len(cwv_indices) == 1:
        level = cwv_indices[0]
        level_value = float(cwv_values[level])
        held = np.abs(cwv - level_value) > 1e-9
        if np.any(held):
            geo_oor |= held  # a single-level hold is a boundary hold, disclose it too
            warning_key = (root, aero_idx, level)
            if warning_key not in _SINGLE_CWV_WARNING_KEYS:
                warnings.warn(
                    f"LUT has only CWV={level_value} populated for aerosol {aerosol!r}; "
                    "holding requested CWV at that level rather than interpolating a "
                    "missing bracket.", RuntimeWarning, stacklevel=3)
                _SINGLE_CWV_WARNING_KEYS.add(warning_key)
        cwv_lo = cwv_hi = np.full(n, level, dtype=int)
        cwv_w = np.zeros(n)
        cwv_query_indices = {level}
    else:
        cwv_pairs = [(float(cwv_values[i]), i) for i in cwv_indices]
        cwv_lo, cwv_hi, cwv_w, cwv_oor = _axis_spec_batch(axes, "cwv", cwv, cwv_pairs, clamp=clamp)
        geo_oor |= cwv_oor
        cwv_query_indices = set(cwv_lo.tolist()) | set(cwv_hi.tolist())

    vza_sets = [set(_populated_vza_indices(root, aero_idx, idx)) for idx in cwv_query_indices]
    vza_levels = tuple(sorted(set.intersection(*vza_sets))) if vza_sets else ()

    per_axis = {"sza": np.asarray(sza, dtype=float), "vza": np.asarray(vza, dtype=float),
                "raa": np.asarray(raa, dtype=float),
                "ozone": np.full(n, float(ozone))}
    specs = []
    for name in GEOMETRY_AXES:
        if name == "cwv":
            specs.append(("cwv", cwv_lo, cwv_hi, cwv_w, _AXIS_LOG["cwv"]))
        elif name == "vza" and vza_levels:
            values = axes["axes"]["vza"]["values"]
            pairs = [(float(values[i]), i) for i in vza_levels]
            lo, hi, w, oor = _axis_spec_batch(axes, "vza", per_axis["vza"], pairs, clamp=clamp)
            geo_oor |= oor
            specs.append(("vza", lo, hi, w, _AXIS_LOG["vza"]))
        else:
            lo, hi, w, oor = _axis_spec_batch(axes, name, per_axis[name], clamp=clamp)
            geo_oor |= oor
            specs.append((name, lo, hi, w, _AXIS_LOG[name]))
    return specs, geo_oor


def lookup_spectrum_batch(sza, vza, raa, aerosol, cwv, ozone, aod, band_idx,
                          root=SHARD_ROOT, axes=None, clamp=False):
    """``lookup_spectrum`` vectorized over many pixels that each carry their OWN
    geometry/CWV/AOD.

    sza/vza/raa/cwv/aod are equal-length 1-D arrays (one value per pixel); ozone
    and aerosol are scalar; band_idx is the shared 1-D exact-band vector. Returns
    ``({field: (n_pixels, n_bands)}, clamped (n,))`` where ``clamped`` marks
    pixels whose geometry/AOD was held at a boundary node (``clamp=True``) or
    whose CWV was held at the only populated level.

    A permanent per-cell LUT gap (a documented 6S-unsolvable physics corner, not
    a bug) marks *that pixel's* whole row NaN and leaves every other pixel
    finite; it does not raise. A genuinely missing shard / band / aod-row still
    raises, exactly as the scalar path does."""
    axes = axes or load_axes()
    band_idx = np.asarray(band_idx, dtype=int)
    if band_idx.ndim != 1:
        raise ValueError("band_idx must be one-dimensional")
    if np.any(band_idx < 0) or np.any(band_idx >= len(axes["axes"]["band"]["values"])):
        raise ValueError("band_idx contains an out-of-range band")
    aerosol_values = axes["axes"]["aerosol"]["values"]
    if aerosol not in aerosol_values:
        raise ValueError(f"aerosol {aerosol!r} is not a tabulated model: {aerosol_values}")
    aero_idx = aerosol_values.index(aerosol)

    sza = np.asarray(sza, dtype=float)
    vza = np.asarray(vza, dtype=float)
    raa = np.asarray(raa, dtype=float)
    cwv = np.asarray(cwv, dtype=float)
    aod = np.asarray(aod, dtype=float)
    n = sza.shape[0]
    if not (vza.shape[0] == raa.shape[0] == cwv.shape[0] == aod.shape[0] == n):
        raise ValueError("sza/vza/raa/cwv/aod must be equal-length 1-D arrays")
    n_bands = band_idx.shape[0]

    geo_specs, geo_oor = _geometry_specs_batch(root, aero_idx, aerosol, axes,
                                               sza, vza, raa, cwv, ozone, clamp=clamp)
    aod_lo_all, aod_hi_all, aod_w_all, aod_oor = _axis_spec_batch(axes, "aod", aod, clamp=clamp)
    clamped = geo_oor | aod_oor
    n_aod_axis = len(axes["axes"]["aod"]["values"])

    out = {f: np.full((n, n_bands), np.nan) for f in COEF_FIELDS}
    gap = np.zeros(n, dtype=bool)

    sig = np.stack([np.concatenate([lo[:, None], hi[:, None]], axis=1)
                    for _n, lo, hi, _w, _l in geo_specs], axis=1).reshape(n, -1)
    _uniq, inv = np.unique(sig, axis=0, return_inverse=True)
    inv = inv.reshape(-1)

    for g in range(len(_uniq)):
        members = np.flatnonzero(inv == g)
        group_specs = [(name, int(lo[members][0]), int(hi[members][0]),
                        w[members][:, None], is_log)
                       for name, lo, hi, w, is_log in geo_specs]
        aod_lo = aod_lo_all[members]
        aod_hi = aod_hi_all[members]
        aod_w = aod_w_all[members][:, None]

        def leaf(pinned, _members=members, _aod_lo=aod_lo, _aod_hi=aod_hi, _aod_w=aod_w):
            shard = _cached_shard(root, {**pinned, "aerosol": aero_idx})
            positions = np.array([shard["_band_pos"].get(int(b), -1) for b in band_idx])
            if np.any(positions < 0):
                raise ValueError(f"shard {shard['key']} is missing band indices "
                                 f"{band_idx[positions < 0].tolist()}")
            pos_map = np.full(n_aod_axis, -1)
            for a_i, p in shard["_aod_pos"].items():
                pos_map[a_i] = p
            rows_lo, rows_hi = pos_map[_aod_lo], pos_map[_aod_hi]
            if np.any(rows_lo < 0) or np.any(rows_hi < 0):
                raise ValueError(f"shard {shard['key']} is missing an aod bracket row")
            status = shard["status"]
            bad_lo = status[rows_lo[:, None], positions[None, :]] != STATUS_OK
            bad_hi = status[rows_hi[:, None], positions[None, :]] != STATUS_OK
            gap[_members] |= (bad_lo | bad_hi).any(axis=1)
            c_lo = {f: np.where(bad_lo, np.nan, shard[f][rows_lo[:, None], positions[None, :]])
                    for f in COEF_FIELDS}
            c_hi = {f: np.where(bad_hi, np.nan, shard[f][rows_hi[:, None], positions[None, :]])
                    for f in COEF_FIELDS}
            return _lerp_array_fields(c_lo, c_hi, _aod_w, _AXIS_LOG["aod"])

        blended = _combine_axes(leaf, group_specs, interpolate=_lerp_array_fields)
        for f in COEF_FIELDS:
            out[f][members] = blended[f]

    for f in COEF_FIELDS:
        out[f][gap] = np.nan
    return out, clamped


def correct_spectrum_batch_from_lut(sun_z, sun_a, view_z, view_a, month, day, aero_profile,
                                    aot550, cwv_g_cm2, wl_um, L_obs,
                                    ozone=DEFAULT_OZONE_ATM_CM, root=SHARD_ROOT, axes=None,
                                    clamp=False):
    """``correct_spectrum_from_lut`` vectorized over many pixels with per-pixel
    geometry/CWV/AOD (``lookup_spectrum_batch`` + the reflectance inversion).

    sun_z/sun_a/view_z/view_a/aot550/cwv_g_cm2 are equal-length 1-D arrays;
    wl_um is the shared 1-D band-wavelength vector; L_obs is (n_pixels, n_bands);
    month/day/ozone scalar. Returns ``(reflectance (n, n_bands), gap (n,), clamped
    (n,))``: ``gap`` marks pixels whose bracket hit a permanent per-cell LUT gap
    (their reflectance row is all-NaN); ``clamped`` marks pixels whose
    geometry/AOD was held at a boundary node (``clamp=True``) or whose CWV was
    held at the only populated level."""
    axes = axes or load_axes()
    wl_um = np.asarray(wl_um, dtype=float)
    L_obs = np.asarray(L_obs, dtype=float)
    if wl_um.ndim != 1:
        raise ValueError("wl_um must be one-dimensional (the shared band vector)")
    if L_obs.ndim != 2 or L_obs.shape[1] != wl_um.shape[0]:
        raise ValueError("L_obs must be (n_pixels, n_bands) matching wl_um")
    raa = fold_raa_array(view_a, sun_a)
    doy = month_day_to_doy(month, day)
    band_idx = np.array([nearest_band_index(axes, w) for w in wl_um])
    c, clamped = lookup_spectrum_batch(sun_z, view_z, raa, aero_profile, cwv_g_cm2, ozone,
                                       aot550, band_idx, root=root, axes=axes, clamp=clamp)
    L_ref = normalise_radiance(L_obs, doy)
    reflectance = reflectance_from_radiance(c["xa"], c["xb"], c["xc"], L_ref)
    gap = ~np.isfinite(reflectance).all(axis=1)
    return reflectance, gap, clamped
