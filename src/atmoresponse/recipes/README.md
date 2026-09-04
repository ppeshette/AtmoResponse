# Recipes

Recipe modules hold the small algorithm examples that the report and notebook use.

## Efficient full-spectrum classifiers

For full-spectrum classifiers, avoid per-pixel setup inside the scoring loop. Read the needed band
window as a block, prepare fixed libraries once for the scene wavelength grid, and run classifiers
over arrays or bounded chunks.

The SAM math in `atmoresponse.recipes.sam.sam_angles()` is already vectorized for `(n, bands)`
input, so the important part is keeping wrappers vectorized too. Use
`atmoresponse.recipes.sam.prepare_sam_classifier()` to resample a fixed library once for a scene
wavelength grid, then use its `evaluate_many()` hook for sensitivity runners. The bundled wildfire
example exposes `atmoresponse.recipes.wildfire_sam.prepare_wildfire_sam()` as a thin convenience
wrapper around that generic path. Reserve one-spectrum helpers such as `wildfire_sam_score()` for
one-off checks. Future LUT-backed sensitivity runners should correct whole spectra or chunks in one
call rather than looping over bands.

Two practical performance rules follow from that:

- Parallel workers only help after data are already available to the recipe. If runtime is dominated
  by reading or decompressing image planes, reduce the band window or redundant reads before adding
  workers.
- For many-band SAM runs, atmospheric correction cost scales with pixels times selected bands. Use a
  validated `band_stride` or narrower library window at classifier-preparation time, then correct
  only `selected_wavelengths_nm`.
- Keep vectorized recipes array-native. A per-pixel `dict` interface is convenient for scalar
  formulas, but full-spectrum recipes should not rebuild ordered arrays from per-pixel dictionaries
  in their hot path.
