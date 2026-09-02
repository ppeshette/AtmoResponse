# AtmoResponse

*Tracing atmospheric-correction uncertainty into Tanager reflectance algorithms.*

**AtmoResponse measures how far a spectral algorithm's output moves when Tanager's atmospheric correction
is repeated at a different aerosol optical depth. Potential Sensitivity sweeps representative pixels across a
plausible range of aerosol optical depth, characterizing an algorithm's construction independently
of any one scene. Realized Sensitivity corrects every pixel of a delivered scene twice, at the
aerosol optical depth Tanager shipped and at an external value resolved from AERONET, VIIRS, GOES, or
MERRA-2, then differences the two results. The readout depends on how the algorithm is built: the
fraction of pixels reassigned for a classifier, the fraction of the scene where the output stays
defined for a ratio that can diverge, or a variance fraction for a continuous score. Published
algorithms drawn from mineral, vegetation, wildfire, and water remote sensing span more than four
orders of magnitude in variance fraction, from an AlOH feature depth at 0.0026 percent through a
red-edge canopy ratio at 9.86 percent to a semi-analytical CDOM absorption retrieval at 54 percent,
while a burn-severity classifier reassigns 4.18 percent of its pixels.
Few scenes in the open archive sit near the ground instruments that would verify results of this
kind, and ten acquisition targets would close that gap.**

## 1. The problem

Planet's Tanager mission distributes a surface reflectance product produced by ISOFIT, an
optimal-estimation atmospheric correction that solves for surface reflectance and atmospheric state
together, and users turn that reflectance into a quantitative result by applying an algorithm
recipe. ISOFIT retrieves aerosol optical depth alongside reflectance. In the Malibu fire scene,
the shipped aerosol optical depth rails near 0.001 across bright and dark pixels, against a VIIRS
measurement of 0.028 taken the same day. At 441 nanometers, the per-pixel uncertainty is about
0.002 in both railed and non-railed conditions, so it does not diagnose this retrieval failure. A
wrong aerosol optical depth moves different algorithms by amounts that differ by orders of magnitude,
and nothing in the delivered product says by how much.

## 2. The stress test

AtmoResponse accepts an algorithm, a Tanager scene, and an area of interest. An algorithm maps a set
of wavelengths and their reflectances to a score or a label.

**Potential Sensitivity** corrects representative pixels across a plausible aerosol-optical-depth
range and re-applies the algorithm at each step. **Realized Sensitivity** corrects an area of
interest twice, at Tanager's shipped aerosol optical depth and at an external value resolved from
AERONET, VIIRS, GOES, or MERRA-2. Because it compares two look-up-table corrections rather than the
delivered ISOFIT result or ground truth, the result is sensitivity rather than error. MERRA-2 is a
reanalysis that assimilates AERONET and MODIS, so it supplies a spatial extrapolation rather than an
independent measurement.

AtmoResponse selects a readout that matches the algorithm: reassigned pixels for a classifier,
an aerosol-driven variance fraction for a continuous score, or coverage, the fraction of the scene
where the output stays defined, for a score that can diverge. The variance fraction carries its own
check: when a handful of divergent pixels dominates the reference-corrected output, the shipped,
reference, and difference variances collapse onto one population and the fraction is forced toward
one half regardless of the true balance. AtmoResponse detects that case from the variance of the sum
against the sum of the variances and reports coverage in its place. A baseline difference can cancel
a smooth reflectance shift, whereas an unbounded ratio can amplify it. Miura et al. (2001) measured
atmospheric resistance in vegetation indices, and Blessing and Giering (2026) found
atmospheric-correction error covariance of both signs.

A precomputed table of 6S solutions (Vermote et al. 1997) spanning geometry, aerosol optical depth,
and column water vapour supplies corrections at the shipped and external values, so a full scene
resolves in minutes. The accompanying Method Notes quantify its limits. An aerosol optical
depth near 0.001 is implausibly low for a real atmosphere and sits at the look-up table's lowest
node. Two of the scenes below carry values in that range.

## 3. The algorithms

### 3.1 A red-edge crop ratio is the most exposed land algorithm

Rajanpur, Pakistan, acquired January 14, 2025, is the strongest land-surface case (Figure 1). Inoue
et al. (2016) estimate canopy chlorophyll with a narrow-band ratio, R815/R704, spanning the red
edge. Across 211,873 canopy pixels, replacing the shipped aerosol value with a MERRA-2 value, the
closest available reference, changes 9.86 percent of the algorithm's variance. The result is not a
generic property of vegetation. It follows from the algorithm's construction: a smooth aerosol
perturbation is poorly canceled by a ratio whose bands sit on opposite sides of a steep spectral
slope. The reconstruction
check supports carrying this sensitivity to the delivered product, with correlation 0.997 and
residual scatter 0.14 times the aerosol-driven change.

### 3.2 A burn-severity classifier flips inside the fire perimeter

Spectral Angle Mapper (Kruse et al. 1993) labels each pixel with the nearest member of a fixed
spectral library, measured by spectral angle. The library used here holds 28 curated burned-surface,
vegetation, and substrate spectra, and labels are grouped into burned and unburned before a change is
counted, so that a swap between two near-identical char spectra does not register as a decision.
Over the Palisades fire in Malibu, acquired January 23, 2025 (Figure 2), correcting at a VIIRS
measurement of 0.028 rather than at the shipped floor value reassigns 4.18 percent of pixels. Within the mapped
fire perimeter the rate is 5.9 percent, against 2.4 percent outside it. Reassignment
concentrates where surfaces are spectrally ambiguous, which is inside the burn. The reconstruction
check makes this a controlled atmosphere-response case rather than a prediction that delivered
ISOFIT would move by the same amount.

### 3.3 An inversion splits along its own construction

The NOAA Cyanobacteria Index (Wynne et al. 2008, with the derivative-invariance argument of Philpot
1991 and the detection threshold of Stumpf et al. 2012) measures the curvature of the reflectance
spectrum at 681 nanometers, its value there minus a baseline interpolated from 665 and 709
nanometers. On a Tanager scene of western Lake Ontario from July 4, 2025 (Figure 3), carrying a
bloom along the Rochester shoreline under a shipped aerosol optical depth near 0.14 where GOES at 1
kilometer and 2 minutes measures 0.28, its variance fraction corrected at the GOES value is 0.33
percent. Subtracting that baseline cancels a smooth spectral shift.

The QAA v6 semi-analytical inversion (Lee et al. 2002, with the QAA_v6 update of Lee 2020) is built
from ratios and differences at once. It estimates total absorption from a reflectance ratio, then splits that absorption into a
phytoplankton term and a colored-dissolved-and-detrital term using a spectral slope that a
blue-to-green reflectance ratio sets. On the Malibu fire scene (Figure 4), corrected at an AERONET value of
0.041 against the shipped floor near 0.001, the parts of the retrieval behave oppositely. The
dissolved-and-detrital absorption at 443 nanometers carries a variance fraction of 54 percent and
the phytoplankton absorption at the same wavelength carries 64 percent. The spectral slope, set by a
ratio, carries 2.7 percent. Almost no water pixels leave the inversion undefined. The reconstruction
check separates the slope and the phytoplankton term from look-up-table scatter and places the
dissolved-and-detrital term at the edge of what it supports.

Post-fire coastal water is a case where ash, colored dissolved matter, and chlorophyll all absorb
in the blue, and the retrieval meant to separate them lets the aerosol-assumption error account for more than
half the variance of its dissolved-matter term. The Nam et al. (2021) phycocyanin band
ratio, a raw quotient with no baseline, diverges outright over dark water. The variance check
identifies that and returns coverage, defined across 81 percent of the lake. The Method Notes
tabulate it. One index cancels the shift. One inversion, read three ways, both cancels and amplifies
it within a single run.

### 3.4 Additional results

The Method Notes tabulate the rest of the inventory, from a Dogliotti turbidity retrieval to the
Sims and Gamon water indices. The three case studies already show the three readouts, and the
further examples do not change that arc.

### 3.5 What the tool does not do

AtmoResponse measures aerosol-assumption sensitivity, not ISOFIT's joint retrieval. A systematic
look-up-table offset cancels when two look-up-table corrections are differenced. Pixel-to-pixel
residual scatter determines whether a result transfers to the delivered product. The check does not
support the AlOH depth or the SAM angle, so those results describe correction physics rather than a
quantitative prediction for delivered ISOFIT. The Method Notes report each reconstruction check.

## 4. The acquisitions

AtmoResponse runs anywhere, because AERONET, VIIRS, GOES, and MERRA-2 supply an aerosol reference over
almost any target. Verifying its answer against a ground measurement is a separate matter, and the
open archive offers few places to do it.

The core check is an independent aerosol optical depth at the time of overpass, which AERONET
supplies. A coincident radiosonde adds an independent measure of the column water vapour the
look-up table also corrects for, and over water an AERONET-OC measurement of water-leaving radiance
checks the reflectance itself rather than the aerosol alone. Of the 153 open scenes, 17 sit within
a tight AERONET and radiosonde match, clustered at four places. No scene has a usable AERONET-OC
match, and the nearest Tanager water scene to any reliably reporting SeaPRISM platform lies 238
kilometers away.

Those 17 scenes are also compositionally narrow. Classified against ESA WorldCover 2021, they
average 37 percent built surface against 8 percent across the rest of the archive, and 0.3 percent
exposed rock or soil against 10 percent. The instrumented scenes are cities and the water beside
them. Exposed geology and semi-arid rangeland, the surfaces imaging spectroscopy is flown for, are
close to absent from the set a user can validate against: of 22 scenes with substantial bare ground,
one has an AERONET station that reported at the overpass. The case studies above divide on the same
line, with the Paraná turbidity scene verifiable and the Cuprite, Rajanpur, and Lake Ontario scenes
not at all.

A defensible allocation of the released scenes pairs under-represented surfaces with the reference
tier that verifies them. The candidate pool is substantial, with 393 AERONET land sites reporting in
ten or more months of a typical year and 200 of those within 100 kilometers of an operating
radiosonde. Ten targets span the gap, each schedulable year-round under Tanager's sun-elevation
limits, each near an AERONET site, and most within reach of a radiosonde. Six carry the land
surfaces the validatable archive lacks: Tamanrasset in the Algerian Sahara, Dalanzadgad in the
Gobi, Banizoumbou in the Sahel, Sevilleta in New Mexico, Upington in South Africa, and the ARM
Southern Great Plains site in Oklahoma. Izaña on Tenerife places a mineral surface and open
Atlantic water in one swath. Three sit on AERONET-OC platforms for the water case: Chiba in Tokyo
Bay, LISCO in Long Island Sound, and AAOT in the northern Adriatic. The accompanying Acquisition Targets document supplies each site's purpose, the scoring
method, the top 40 of 435 scored candidates, and Tanager's measured operating envelope. Where the sensor is asked to
look is what this competition's prize decides.

## 5. The tool

AtmoResponse is open source and pip installable. It takes any user-supplied algorithm and returns
its Potential and Realized Sensitivity from public Tanager data and a resolved external aerosol
reference. The public materials are at github.com/ppeshette/AtmoResponse: this summary and its
figures, the Method Notes, the Acquisition Targets, the package with its tests, and an annotated
notebook that reproduces the four case-study figures and the values they carry.

---

## Acknowledgements

This work was carried out with the assistance of AI coding tools, principally Anthropic's Claude
Code and OpenAI's Codex, which supported implementation, code review, and analysis throughout. All
code, comments, and prose in this repository were written, edited, or reviewed by the author.

---

## References

- Blessing, S. and Giering, R. (2026). Error covariance structure caused by uncertainties in
  atmospheric correction for optical sensors. *Surveys in Geophysics*, advance online publication.
  doi:10.1007/s10712-025-09921-8
- Dogliotti, A. I., Ruddick, K. G., Nechad, B., Doxaran, D., and Knaeps, E. (2015). A single
  algorithm to retrieve turbidity from remotely-sensed data in all coastal and estuarine waters.
  *Remote Sensing of Environment*, 156, 157-168. doi:10.1016/j.rse.2014.09.020
- Inoue, Y., Guérif, M., Baret, F., Skidmore, A., Gitelson, A., Schlerf, M., Darvishzadeh, R., and
  Olioso, A. (2016). Simple and robust methods for remote sensing of canopy chlorophyll content: a
  comparative analysis of hyperspectral data for different types of vegetation. *Plant, Cell and
  Environment*, 39(12), 2609-2623. doi:10.1111/pce.12815
- Kruse, F. A., Lefkoff, A. B., Boardman, J. W., Heidebrecht, K. B., Shapiro, A. T., Barloon, P. J.,
  and Goetz, A. F. H. (1993). The spectral image processing system (SIPS): interactive
  visualization and analysis of imaging spectrometer data. *Remote Sensing of Environment*, 44,
  145-163. doi:10.1016/0034-4257(93)90013-N
- Lee, Z., Carder, K. L., and Arnone, R. A. (2002). Deriving inherent optical properties from water
  color: a multiband quasi-analytical algorithm for optically deep waters. *Applied Optics*, 41(27),
  5755-5772. doi:10.1364/AO.41.005755
- Lee, Z.-P. (2020). Steps and calculations of the Quasi-Analytical Algorithm (QAA_v6). International
  Ocean Colour Coordinating Group. https://www.ioccg.org/groups/software.html
- Miura, T., Huete, A. R., Yoshioka, H., and Holben, B. N. (2001). An error and sensitivity analysis
  of atmospheric resistant vegetation indices derived from dark target-based atmospheric correction.
  *Remote Sensing of Environment*, 78(3), 284-298. doi:10.1016/S0034-4257(01)00223-1
- Nam, G., Shin, H., Ha, R., Song, H., Yoo, J., Lee, H., Park, S., Kang, T., and Kim, K. (2021).
  Quantification of phycocyanin in inland waters through remote measurement of ratios and shifts in
  reflection spectral peaks. *Remote Sensing*, 13(16), 3335. doi:10.3390/rs13163335
- Philpot, W. D. (1991). The derivative ratio algorithm: avoiding atmospheric effects in remote
  sensing. *IEEE Transactions on Geoscience and Remote Sensing*, 29(3), 350-357.
  doi:10.1109/36.79425
- Sims, D. A. and Gamon, J. A. (2003). Estimation of vegetation water content and photosynthetic
  tissue area from spectral reflectance: a comparison of indices based on liquid water and
  chlorophyll absorption features. *Remote Sensing of Environment*, 84(4), 526-537.
  doi:10.1016/S0034-4257(02)00151-7
- Stumpf, R. P., Wynne, T. T., Baker, D. B., and Fahnenstiel, G. L. (2012). Interannual variability
  of cyanobacterial blooms in Lake Erie. *PLoS ONE*, 7(8), e42444.
  doi:10.1371/journal.pone.0042444
- Vermote, E. F., Tanré, D., Deuzé, J. L., Herman, M., and Morcrette, J.-J. (1997). Second
  simulation of the satellite signal in the solar spectrum, 6S: an overview. *IEEE Transactions on
  Geoscience and Remote Sensing*, 35(3), 675-686. doi:10.1109/36.581987
- Wynne, T. T., Stumpf, R. P., Tomlinson, M. C., Warner, R. A., Tester, P. A., Dyble, J., and
  Fahnenstiel, G. L. (2008). Relating spectral shape to cyanobacterial blooms in the Laurentian
  Great Lakes. *International Journal of Remote Sensing*, 29(12), 3665-3672.
  doi:10.1080/01431160802007640

