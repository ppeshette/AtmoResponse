# AtmoResponse

AtmoResponse traces atmospheric-correction assumptions into Tanager reflectance algorithms.

The repository is the public project package for the Planet Tanager Open Data Competition
submission. It is being assembled as a release artifact: source code, report files, an annotated
notebook, and lightweight reproducibility assets belong here. Scene data, bulky derived products,
local caches, and credentials do not.

## Repository Layout

| Path | Purpose |
|---|---|
| `src/atmoresponse/` | Installable Python package |
| `src/atmoresponse/recipes/` | Published algorithm examples used by the submission |
| `src/atmoresponse/masks.py` | Mask composition helpers for recipe outputs and scene validity gates |
| `tests/` | Offline smoke tests and formula fixtures |
| `notebooks/` | Annotated executable walkthrough |
| `report/` | Memo, method notes, acquisition targets, and figure captions |
| `reproducibility_assets/` | Small manifests and source assets needed to rebuild examples |
| `examples/` | Small runnable entry points and configuration examples |

## Install

Create the clean conda environment:

```bash
conda env create -f environment.yml
conda activate atmoresponse
```

Install the package:

```bash
python -m pip install .
```

Development checks use:

```bash
python -m pip install -e ".[test,live,geo]"
python -m pytest
```

Run the offline quickstart:

```bash
python examples/quickstart.py
```

## Recipe API

Algorithm examples are exposed from `atmoresponse.recipes`. Import formula recipes directly from
that package, and import full-spectrum helpers from their recipe modules:

```python
from atmoresponse.recipes import cyanobacteria_index
from atmoresponse.recipes.sam import prepare_sam_classifier
from atmoresponse.recipes.wildfire_sam import prepare_wildfire_sam
```

For repeated SAM scoring, prepare the fixed library once for a scene wavelength grid, then call
`evaluate_many()` on band-last spectra. Use `selected_wavelengths_nm` when choosing which bands to
read or atmospherically correct upstream.

`atmoresponse.recipes.water` exposes MNDWI as a recipe. `atmoresponse.masks` combines recipe outputs
with cloud, nodata, LUT-coverage, and finite-input validity gates.

## Data Access

Default examples should run against public Planet STAC metadata and public Tanager scene assets.
Some external aerosol references require user-owned credentials. Credentialed paths must fail with
readable setup messages when credentials are absent.

Bundled example spectra are limited to small fixed libraries with visible provenance. See `NOTICE`
and the JSON manifests under `src/atmoresponse/assets/` for source credits and intended-use limits.

AtmoResponse walks Planet's static Tanager STAC catalog directly. It does not require a STAC API
server for catalog search.
EMIT catalog searches use NASA CMR and can target either L2A reflectance or L1B radiance products.

AtmoResponse uses a cache-first workflow. Set `ATMORESPONSE_CACHE` to choose the local cache
location. If the variable is unset, the package uses a platform cache directory under the current
user profile. Scene asset downloads are written to temporary files first, then moved into the cache
when complete.

## Look-up table

Potential and Realized Sensitivity are computed against a precomputed look-up table of
atmospheric-correction coefficients, one table per sensor. Each table holds the 6S
radiative-transfer coefficients for a fixed grid of solar and view geometry, column water vapour,
ozone, aerosol model, aerosol optical depth, and sensor band. A reflectance retrieval at an assumed
AOD is then a table lookup plus a short algebra step rather than a live radiative-transfer call,
which is what makes a full-scene comparison at two AOD values tractable.

The package ships only the axis definitions, `src/atmoresponse/assets/lut/axes_tanager.json` and
`axes_emit.json`. The coefficient store is large and is distributed separately as a per-sensor
archive on Zenodo. Fetch and unpack it with `download_lut`:

```python
from atmoresponse.downloads import download_lut

store = download_lut("tanager")            # into the default cache, idempotent
result = run_tanager(..., lut=store)
```

`download_lut` returns the store directory (the one containing `shards/`). Point
`LUT_STORE_TANAGER` or `LUT_STORE_EMIT` at it instead, or pass it to `run_tanager` or `run_emit`
as `lut=`. The Tanager and EMIT tables are Zenodo records
[10.5281/zenodo.22210933](https://doi.org/10.5281/zenodo.22210933) and
[10.5281/zenodo.22210726](https://doi.org/10.5281/zenodo.22210726); pass `url=` to fetch a
different archive.

The pipeline that generated the tables (the 6S forward-model driver and the shard store layout) is
not part of this release. It may be published separately at a later date. The method itself is
described in the report.

## Current Scaffold

This scaffold defines the public import boundary while the full implementation is assembled.
Shared scene models, a neutral hyperspectral cube, cache-backed downloads, source-neutral scene
asset caching, explicit Tanager and EMIT catalog access, source-neutral surface classification,
Tanager and EMIT adapters, shipped-AOD summaries, AOD reference selection from AERONET, GOES,
VIIRS, and MERRA-2, the LUT consumer layer, the LUT archive download helper (`download_lut`), per-sensor
sensitivity runners (`run_tanager`, `run_emit`), the figure primitives, fixed-library SAM
primitives, the bundled wildfire SAM library, and the RSI/WI, Wynne CI, QAA v6 CDOM, and AlOH
example algorithms are implemented.
GOES uses public NOAA buckets. VIIRS and MERRA-2 use Earthdata-backed live dependencies and
credentials.

## Acknowledgements

AI coding assistants were central to this work, principally Anthropic's Claude Code and OpenAI's
Codex. They supported implementation, code review, and analysis throughout the project, and I am
grateful for their availability as tools for research. I wrote, edited, or reviewed all code published
in this repository.
