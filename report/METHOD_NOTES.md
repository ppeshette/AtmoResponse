# Method notes

Supplementary to the project summary. The result inventory records additional examples, and two
limits qualify every Realized Sensitivity figure the summary reports.

## Additional algorithm results

Three project-summary case studies demonstrate AtmoResponse's classifier, continuous-score, and
coverage readouts. The result inventory below retains further examples and can grow without changing that
case-study arc.

| Algorithm | Structure | Scene | Variance fraction |
|---|---|---|---|
| Continuum-removed AlOH band depth | local-continuum absorption depth | Cuprite, February 22, 2025 | 0.0026 percent |
| Dogliotti et al. (2015) turbidity | switching single-band reflectance | Paraná delta, June 26, 2025 | 0.61 percent |
| Sims & Gamon (2003) WI_1240 | two-band water-absorption ratio | Rajanpur, January 14, 2025 | 1.06 percent |
| Sims & Gamon (2003) WI_1530 | two-band water-absorption ratio | Rajanpur, January 14, 2025 | 2.09 percent |
| QAA v6 CDOM slope (Lee et al. 2002) | absorption slope set by a blue-green ratio | Malibu, January 23, 2025 | 2.7 percent |
| QAA v6 dissolved-and-detrital absorption at 443 nm | absorption magnitude through the u-to-a chain | Malibu, January 23, 2025 | 54 percent |
| QAA v6 phytoplankton absorption at 443 nm | difference of two exposed absorption terms | Malibu, January 23, 2025 | 64 percent |
| Nam et al. (2021) phycocyanin BRPD | raw two-band ratio, no baseline | Lake Ontario, July 4, 2025 | coverage 81 percent (guard-failed) |

The three QAA rows are one inversion run three ways on the same scene: the slope, which a ratio
sets, is close to unaffected, while the absorption magnitudes it separates are among the most
exposed results in the inventory. The phycocyanin ratio diverges over dark water, where the variance
fraction collapses toward one half and AtmoResponse returns coverage instead.

## What a lookup-table correction reproduces

Every figure in the project summary differences two lookup-table corrections rather than two ISOFIT runs. The
table reproduces ISOFIT closely but not exactly, so the question is whether the aerosol-driven
change an algorithm shows is larger than the table's own departure from the delivered product.

A systematic offset between the two corrections cancels in a difference, because both aerosol
evaluations carry it. What limits a sensitivity measurement is the part that varies from pixel to
pixel: the residual scatter about a linear fit of the lookup-table result against the delivered one.
Applying each algorithm to Tanager's delivered surface reflectance at the same pixels measures it.

| Algorithm | correlation with delivered ISOFIT | residual scatter as a fraction of the aerosol-driven change |
|---|---|---|
| QAA v6 CDOM slope | 0.987 | 0.14 |
| Inoue RSI | 0.997 | 0.14 |
| Nam phycocyanin BRPD | 0.728 | 0.14 |
| QAA v6 phytoplankton absorption at 443 nm | 0.883 | 0.27 |
| Sims & Gamon WI_1530 | 0.999 | 0.37 |
| Dogliotti turbidity | 0.996 | 0.39 |
| QAA v6 dissolved-and-detrital absorption at 443 nm | 0.702 | 0.57 |
| Sims & Gamon WI_1240 | 0.997 | 0.59 |
| Wynne Cyanobacteria Index | 0.990 | 0.66 |
| Cuprite AlOH feature depth | 0.999 | 14.9 |
| Malibu SAM angle | 0.708 | 15.1 |

For the rows above the Cuprite AlOH depth the aerosol-driven change is at least as large as the
reconstruction scatter. Where the fraction is below about 0.4 the reported sensitivity is separable
from the reconstruction gap. Where it is between 0.4 and about one, including the QAA
dissolved-and-detrital term at 0.57, the WI_1240 ratio, and the Cyanobacteria Index, the aerosol
change and the scatter are the same order and the sensitivity is an upper bound. The AlOH depth and
the Malibu SAM angle do not clear the gate. Their aerosol effects are about 15 times smaller than the
pixel-level departure from ISOFIT, so those figures describe the sensitivity of the correction
physics rather than verified predictions of how the delivered product would respond. Direction is
unaffected because differencing two lookup-table corrections isolates the aerosol term, but the
numbers should not be transferred to ISOFIT quantitatively.

## Two limits on a Realized Sensitivity map

**Aerosols are retrieved per segment, not per pixel.** ISOFIT solves the aerosol field over spatial
segments rather than at every pixel (the approach documented in the EMIT L2A ATBD, which uses the
same retrieval), and on the Tanager delta scenes that field
varies over roughly a kilometre. Neighbouring pixels in a Realized Sensitivity map therefore do not
carry independent aerosol information, and the maps should be read for their spatial pattern rather
than pixel by pixel.

**The table has a column water vapour floor.** Its lowest column water vapour node is 1.0 gram per
square centimetre, so a drier scene is corrected at that boundary value rather than at its own. The
affected pixel fraction is reported alongside each result.

## Wavelength dependence over water

ACIX-Aqua (Pahlevan et al. 2021) and ACIX-III Aqua (Giardino et al. 2025) both locate the largest
atmospheric-correction disagreement at the blue end, the latter at 443 nanometres across 239 PRISMA
scenes. Tanager exhibits the same wavelength dependence, which is why the water diagnostics in the
project summary rely on the green rather than the blue. Over water the shortwave infrared carries no
water-leaving signal, so the aerosol constraint that a dark-target land correction draws from those
bands is unavailable there.

## References

Works named in the algorithm inventory (Lee 2002, Dogliotti 2015, Sims and Gamon 2003, Nam 2021,
and the Cyanobacteria Index sources) are cited in full in the project summary. The additional works
named here:

- Green, R. O. et al. (2022). EMIT Level 2A Algorithm Theoretical Basis Document: Surface
  Reflectance and Mask. Jet Propulsion Laboratory, California Institute of Technology. Available
  from the NASA Land Processes DAAC.
- Giardino, C., Pahlevan, N., Fabbretto, A., Panizza, L., Pellegrino, A., et al. (2025). ACIX-III
  Aqua: evaluation of atmospheric correction for hyperspectral PRISMA imagery over inland and
  coastal waters. *International Journal of Remote Sensing*, 46(23), 9066-9090.
  doi:10.1080/01431161.2025.2574517
- Pahlevan, N., Mangin, A., Balasubramanian, S. V., Smith, B., Alikas, K., et al. (2021). ACIX-Aqua:
  a global assessment of atmospheric correction methods for Landsat-8 and Sentinel-2 over lakes,
  rivers, and coastal waters. *Remote Sensing of Environment*, 258, 112366.
  doi:10.1016/j.rse.2021.112366
