# Reproducibility Assets

Small manifests, source spectra, and preparation scripts belong here when they are needed to rebuild
published examples.

Large scene data, generated rasters, local caches, and credential files are excluded from the
repository.

The bundled Malibu wildfire SAM library is packaged under `src/atmoresponse/assets/endmembers/`.
It is a 400-1300 nm fixed-library example derived from USGS Digital Spectral Library splib06a
(Clark et al. 2007) and SAFARI 2000 fire-residue spectra (Landmann & Roy 2004,
doi:10.3334/ORNLDAAC/751). Its JSON manifest carries the source credits and intended-use limit.

The LUT axis definitions under `src/atmoresponse/assets/lut/` are the runtime subset of the axis
files that describe each sensor's look-up table. They carry the append-only value lists, their
units, and the band centre and width tables, which is everything `atmoresponse.lut` reads. The
table generation code is not part of this release. See the README section on the look-up table.
