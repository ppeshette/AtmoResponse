# Known issues

Package-robustness bugs found during the competition submission review. None changes a submitted
number or figure: the annotated notebook filters or masks around each one, and the submitted values
in `src/atmoresponse/assets/figure_values.json` are what the notebook produces. They are listed here
because a user calling the affected functions directly, outside the notebook's guarded path, can
hit wrong or confusing behaviour. Fix order below.

## 1. `variance_fraction` does not drop non-finite pairs (P1)

`src/atmoresponse/sensitivity.py`. `variance_fraction(at_reference, delta)` passes its inputs
straight to `np.var`. A single NaN, for example one LUT-gap pixel, makes `atmosphere_variance`,
`additivity`, `atmosphere_fraction`, and `coverage` all NaN even when the rest of the scene is
valid.

Submission impact: none. Every notebook case that can produce a NaN filters first (the AlOH cell
with `np.isfinite`, the CDOM cell through `_run_cdom` plus `.scored()`). RSI, SAM, and Wynne CI have
no gaps and all report `reliable=True`.

Where it bites: a user who calls `variance_fraction` on a raw `SensitivityResult` that has any LUT
gap gets an all-NaN result with no indication why.

Fix: mask `np.isfinite(at_reference) & np.isfinite(delta)` inside `variance_fraction`, report the
retained fraction, add a test with one NaN.

## 2. Classifier flip count treats a one-sided gap as a class change (P1)

`src/atmoresponse/sensitivity.py`. `class_changed` is `label_shipped != label_reference`. If one
side is a gap (`None`) and the other is a label, the comparison is `True` and the pixel is counted
as a flip.

Submission impact: negligible. The submitted SAM flip fraction is the notebook value, 4.18 percent.
A separate cross-check on a slightly different pixel set gave 4.15 percent, so gap-driven false
flips are not materially inflating the number.

Fix: exclude pixels where either label is `None` from `class_changed` and from the flip-fraction
denominator.

## 3. `from_aeronet` strips the timezone instead of converting, and queries a single day (P2)

`src/atmoresponse/aeronet.py`. `from_aeronet()` calls `when.replace(tzinfo=None)` rather than
converting to UTC the way `from_goes`, `from_viirs`, and `from_merra2` do. It also queries
`day..day`, so a valid observation inside the requested time window but on the other side of
midnight is invisible.

Submission impact: none. The notebook's one live AERONET call uses a naive datetime, where
`replace(tzinfo=None)` is a no-op, and every other reference value in the notebook is a recorded
constant.

Where it bites: a user passing a timezone-aware local datetime gets the wrong matching window.

Fix: `when.astimezone(timezone.utc)` for a tz-aware input, and widen the AERONET query to cover the
full `[when - tolerance, when + tolerance]` span including a day boundary.

## 4. `sample_linear` does not apply the fill-value-to-NaN convention (P2)

`src/atmoresponse/recipes/_spectral.py` and `src/atmoresponse/recipes/mineral.py`.
`sample_linear()` linearly interpolates without the `-9999 -> NaN` guard that `nearest_reflectance()`
applies. On a fill pixel the AlOH recipe returns a finite absurd value (observed 1.000024 and
60601.0 from `-9999` inputs) instead of NaN.

Submission impact: none. `masks.admissible` plus the `nodata_pixels` screen remove fill pixels
before the AlOH recipe runs, which is why the notebook AlOH result is `atmosphere_fraction`
2.6e-5 and not a blow-up.

Where it bites: a user running the recipe on unmasked reflectance.

Fix: apply the same `<= FILL_LIMIT -> NaN` guard in `sample_linear` (or in `validate_spectra`), add
a fixture with a fill pixel.
