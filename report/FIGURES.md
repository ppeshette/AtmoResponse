# Figures

Each figure is the standard three-panel output of `plotting.sensitivity_figure`: Potential
Sensitivity curves on the left, the algorithm's absolute output in the center, and the Realized
Sensitivity map on the right. On the maps a warm gray marks pixels the sensor imaged but the
analysis mask excluded, and the pale border is outside the ortho frame. The badge on the Realized
Sensitivity panel carries the reported number. The annotated notebook regenerates all four.

## Figure 1. Canopy chlorophyll red-edge ratio, Rajanpur, Pakistan

`figures/rsi.png`. Scene `20250114_062056_92_4001`, acquired January 14, 2025. The Inoue et al.
(2016) `R815 / R704` ratio over 211,873 canopy pixels, corrected at the shipped aerosol optical
depth and at a MERRA-2 reference. Across the scene, 9.86 percent of the ratio's variance is the
aerosol assumption rather than real canopy difference. The Realized Sensitivity map is almost all
one sign, so the ratio shifts coherently across the whole scene rather than at scattered pixels.
This is the most aerosol-exposed of the land-surface algorithms, which a ratio spanning a steep
spectral slope predicts.

## Figure 2. Wildfire Spectral Angle Mapper, Palisades fire, Malibu, California

`figures/sam.png`. Scene `20250123_185518_92_4001`, acquired January 23, 2025, days after the
fire. A Spectral Angle Mapper against a fixed library of 28 burned and unburned spectra, corrected
at the shipped floor value and at a VIIRS reference. The magenta overlay marks pixels that change
between burned and unburned, and the black outline is the WFIGS fire perimeter. The badge reports
the flip fraction for the whole scene, inside the perimeter, and outside: 4.18 percent overall, 5.9
percent inside, 2.4 percent outside. Reassignment concentrates where surfaces are spectrally ambiguous,
which is inside the burn.

## Figure 3. NOAA Cyanobacteria Index, western Lake Ontario

`figures/ci.png`. Scene `20250704_165204_61_4001`, acquired July 4, 2025, over a bloom along the
Rochester shoreline. The Wynne et al. (2008) Cyanobacteria Index, the curvature of the reflectance
spectrum at 681 nm read as its value there minus a baseline from 665 and 709 nm, corrected at the
shipped aerosol optical depth and at a GOES reference. Its variance fraction is 0.33 percent, the
lowest in the set. Subtracting the baseline cancels a smooth spectral shift, so the index barely
moves even though the aerosol error is large.

## Figure 4. QAA v6 dissolved-and-detrital absorption, Malibu, California

`figures/cdom.png`. Same scene as Figure 2, `20250123_185518_92_4001`, over the coastal water. The
QAA v6 semi-analytical inversion (Lee et al. 2002, Lee 2020), corrected at the shipped floor value
and at an AERONET reference. The panel shows `a_dg(443)`, the dissolved-and-detrital absorption at
443 nm, whose variance fraction is 54 percent. Run on the same pixels, the inversion also returns a
phytoplankton absorption at 64 percent and a spectral slope at 2.7 percent. One algorithm produces both a near-immune output, the slope set by a ratio,
and two of the most exposed outputs in the walkthrough, the absorption magnitudes.
