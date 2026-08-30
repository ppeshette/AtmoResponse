"""Small offline quickstart for the public AtmoResponse recipe API."""

from __future__ import annotations

import numpy as np

from atmoresponse.recipes.agriculture import canopy_chlorophyll_rsi, vegetation_water_indices
from atmoresponse.recipes.cyanobacteria import cyanobacteria_index
from atmoresponse.recipes.mineral import aloh_2200_depth
from atmoresponse.recipes.wildfire_sam import load_wildfire_sam_library, wildfire_sam_score


def main() -> None:
    wavelengths_nm = np.array([
        620.0,
        665.0,
        681.0,
        704.0,
        709.0,
        815.0,
        865.0,
        1240.0,
        1530.0,
        2150.0,
        2200.0,
        2250.0,
    ])
    reflectance = np.array([
        0.020,
        0.026,
        0.023,
        0.030,
        0.030,
        0.120,
        0.140,
        0.070,
        0.050,
        0.160,
        0.120,
        0.170,
    ])

    rsi = canopy_chlorophyll_rsi(reflectance, wavelengths_nm)
    wi_1240, wi_1530 = vegetation_water_indices(reflectance, wavelengths_nm)
    ci, ss_665, cyano_dominant = cyanobacteria_index(reflectance, wavelengths_nm)
    aloh = aloh_2200_depth(reflectance, wavelengths_nm)

    library = load_wildfire_sam_library()
    sam = wildfire_sam_score(library.reflectance[0], library.wavelengths_nm, library)

    print(f"RSI R815/R704: {float(rsi):.3f}")
    print(f"WI R865/R1240: {float(wi_1240):.3f}")
    print(f"WI R865/R1530: {float(wi_1530):.3f}")
    print(f"Wynne CI: {float(ci):.6f}; SS665: {float(ss_665):.6f}; dominant: {bool(cyano_dominant)}")
    print(f"AlOH 2200 depth: {float(aloh):.3f}")
    print(f"Wildfire SAM target label: {sam.label}; angle: {sam.value:.6f} rad")


if __name__ == "__main__":
    main()
