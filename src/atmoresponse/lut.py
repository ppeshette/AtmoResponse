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
import json
import math
import os
import re
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

@lru_cache(maxsize=None)
def load_axes(path=None):
    """Load the axis-definition file (index -> value, per axis).

    ``path=None`` reads the copy bundled with the package. Axis value lists are
    **append-only**: refining density adds values at the end, so an existing
    cell's integer key never changes meaning. Index order is therefore not value
    order -- always sort by value at lookup.

    Cached for the process lifetime. Call ``load_axes.cache_clear()`` if a test
    points ``path`` at a file that changes underfoot.
    """
    if path is None:
        text = files("atmoresponse.assets.lut").joinpath("axes.json").read_text()
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
