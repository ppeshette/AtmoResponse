# Bundled assets

Everything the package needs at runtime that is not code, plus the small reference files that let a
reviewer check the walkthrough without running it. It all ships in the wheel. Large scene data,
generated rasters, local caches, and credential files are excluded from the repository.

## `endmembers/`

The Malibu wildfire SAM library, `wildfire_vnir_sam_curated.npz` with its `wildfire_vnir_sam_curated.json`
manifest. A 400 to 1300 nm fixed-library example derived from the USGS Digital Spectral Library
splib06a (Clark et al. 2007) and SAFARI 2000 fire-residue spectra (Landmann and Roy 2004,
doi:10.3334/ORNLDAAC/751). The manifest carries the source credits and the intended-use limit.

`palisades_fire_perimeter.npz` is the January 2025 Palisades fire perimeter from Wildland Fire
Interagency Geospatial Services (WFIGS, public), rasterized onto the Malibu Tanager scene
`20250123_185518_92_4001`. The walkthrough passes it as the wildfire SAM `scoring_region`.

The `.npz` files here are whitelisted past the repository's blanket `*.npz` rule.

## `lut/`

`axes_tanager.json` and `axes_emit.json`, the runtime subset of the look-up table axis definitions.
They carry the append-only value lists, their units, and the band centre and width tables, which is
everything `atmoresponse.lut` reads. The table generation code is not part of this release.

## `figure_values.json`

The values the annotated notebook exports: the atmosphere fraction or flip fraction for every
walkthrough case. The project summary and the figure captions cite it. The notebook rewrites its own
run copy next to itself; this committed copy is the reference.
