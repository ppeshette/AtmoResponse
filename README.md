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

## Data Access

Default examples should run against public Planet STAC metadata and public Tanager scene assets.
Some external aerosol references require user-owned credentials. Credentialed paths must fail with
readable setup messages when credentials are absent.

AtmoResponse walks Planet's static Tanager STAC catalog directly. It does not require a STAC API
server for catalog search.

AtmoResponse uses a cache-first workflow. Set `ATMORESPONSE_CACHE` to choose the local cache
location. If the variable is unset, the package uses a platform cache directory under the current
user profile. Scene asset downloads are written to temporary files first, then moved into the cache
when complete.

## Current Scaffold

This scaffold defines the public import boundary while the full implementation is assembled.
Tanager catalog search, scene asset resolution, cache-backed scene downloads, HDF5 extraction
utilities, and AOD reference-selection infrastructure are implemented. Live aerosol source
providers, LUT storage, and full sensitivity evaluation are named but not yet implemented.
