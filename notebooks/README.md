# Notebooks

`walkthrough.ipynb` is the annotated walkthrough. `walkthrough.py` is its source in the
jupytext percent format (`# %%` cell markers), kept so the notebook reviews cleanly as a
diff. Install the notebook tooling and the run-time extras with:

```bash
python -m pip install "atmoresponse[notebook]"
```

Regenerate the notebook from its source after editing `walkthrough.py`:

```bash
jupytext --to notebook notebooks/walkthrough.py
```

## Status: runs end to end against real scenes

Every run, figure, and value-export cell is live. Each example states its scene, area of
interest, aerosol model, and gathered aerosol optical depth references inline, with the
provenance of each value. Scenes download on first use into the data directory
(`ATMORESPONSE_DATA`, default `~/atmoresponse_data`). The look-up table, about 240 MB,
downloads once in section 1 and is kept in the same place.

## Sections

1. Setup: data directory and look-up table.
2. Finding scenes in Planet's catalog.
3. The reference aerosol optical depth: AERONET, GOES, VIIRS, and MERRA-2.
4. The aerosol model: the four 6S models and why the analyst picks one.
5. Walkthrough A, a continuous output: canopy chlorophyll near Rajanpur, Pakistan.
6. Walkthrough B, a classifier: wildfire ash and char near Malibu, California.
7. A narrow-band water index: cyanobacteria on Lake Ontario.
8. A spectral inversion on the same water: CDOM off Malibu.
9. A shortwave-infrared mineral feature: AlOH depth at Cuprite, Nevada.
10. Collect the figure-ready values.

Sections 5 through 9 are independent after the shared setup, so any one can be run on
its own.

## The shape of a run

```python
result = sensitivity.run_tanager(
    scene_id, aoi, mask, band_targets_nm, reference_aod, aero_profile,
    algorithm=my_algorithm, data_dir=DATA_DIR, lut=LUT_STORE_TANAGER,
)
fig = plotting.sensitivity_figure(result)
```

`recipes.as_algorithm(recipe_function)` adapts a fixed-band recipe to the `algorithm`
argument, and a prepared SAM classifier is passed as `algorithm` directly. The `mask` is
a `masks` helper (`tanager_water`, `tanager_vegetation`, `tanager_land`, `tanager_clear`)
or any function `(sr_h5, aoi) -> bool array`, wrapped in `masks.admissible(...)`, which adds the
cloud, nodata, finite-radiance, and look-up-table-coverage gates.
