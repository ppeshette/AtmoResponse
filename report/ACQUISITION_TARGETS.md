# Supplementary acquisition targets

This is the ranked candidate set behind the short list in the project summary. Each entry is a place
where a Tanager scene could be checked against an independent atmospheric measurement, which the
current archive rarely allows.

## How the list was built

Three filters, applied in order.

**Reporting reliability.** Every AERONET land site was scored for seasonal reliability by the method
already used for the marine platforms: for each calendar month, how many of the last five and last
ten years carried at least one cloud-screened observation. Sites reporting in fewer than eight
months of a typical year, or with fewer than 400 observation-days in ten years, were dropped. That
leaves 435 sites. A stricter cut, sites reporting in ten or more months, leaves 393, and 200 of
those sit within 100 kilometers of an operating radiosonde.

**Surface.** ESA WorldCover 2021 was sampled in a 40 km box around each site to describe what a
scene there would contain. Credit went to exposed rock and soil, open shrub and grassland, and
cropland, the surfaces almost absent from the scenes a user can currently validate against, and to
proximity to an AERONET-OC platform, which is the reference used here to close the loop on a water
product rather than on the aerosol alone.

**Tanager's operating envelope.** Measured from the 153 open scenes: a 406 km sun-synchronous orbit
crossing at about 11:00 local solar time, pointing up to 30 degrees off-nadir, and never tasked
below about 15 degrees sun elevation. A site was rated schedulable year-round if it clears roughly
20 degrees sun at the overpass even near the winter solstice, which holds to about 52 degrees
latitude, schedulable in the summer half of the year to about 62 degrees, and marginal beyond that.
Nearby radiosonde launches were rated on whether a 00 or 12 UTC ascent falls within three hours of
the overpass, since a profile taken twelve hours away constrains little.

Revisit is a tasking decision, not an orbital limit. The archive shows Tanager imaging Buenos Aires
13 times across 215 days and Rochester 16 times across four months once a campaign is committed, so
the recommendation is for a small number of campaigns rather than scattered single scenes.

## Short list

Ten targets that together span the gap, each schedulable year-round, at an AERONET site, and near a
radiosonde. The list is a focused acquisition recommendation. The final thirty-scene allocation rests with the
Open Committee, so this is a recommendation to that committee rather than a fixed request.

| target | site | what it adds |
|---|---|---|
| **Saharan desert** | Tamanrasset, Algeria | Exposed rock at 98 percent of the frame, a radiosonde 9 km away, and a 30-year aerosol record under heavy dust. The reference case for mineral and geologic work, which currently has no validatable scene anywhere. |
| **Cold desert, Asian dust** | Dalanzadgad, Mongolia | Bare ground at 86 percent with a radiosonde at the site itself. A cold, dry continental dust regime, which neither Tamanrasset nor the North American sites supply. |
| **Sahel grassland** | Banizoumbou, Niger | Grass at 79 percent over bare soil, a well-timed radiosonde 66 km away, and a seasonal alternation between Saharan dust and biomass-burning aerosol at one location. |
| **North American arid grassland** | Sevilleta, New Mexico | Chihuahuan Desert shrub and grassland with bare soil, a long AERONET and NEON record, and the Albuquerque radiosonde 80 km away. Semi-arid rangeland, absent from the validatable archive. |
| **Southern Hemisphere semi-arid** | Upington, South Africa | Kalahari shrub and grassland, a radiosonde 11 km away, and a location that fills the Southern Hemisphere land gap the archive leaves almost empty. |
| **Continental cropland** | ARM Southern Great Plains, Oklahoma | The most instrumented continental atmospheric site in the world, with radiosondes four times a day, lidar, and flux towers, over winter wheat and pasture. Agricultural cropland beside a reference that constrains the full column. |
| **Subtropical montane and coastal** | Izaña, Tenerife | A 2,400 m volcanic observatory with a radiosonde 12 km away and open Atlantic water within the same swath, so one campaign serves both a mineral surface and a coastal water target. |
| **Water-leaving radiance closure** | Chiba, Japan | A land AERONET site 7 km from the Kemigawa Offshore SeaPRISM platform and 48 km from a well-timed radiosonde. The rare place all three references sit together, which closes the loop on a water product. |
| **Optically complex coastal water** | LISCO, Long Island Sound | A SeaPRISM platform reporting in all 12 months with a radiosonde 42 km away, over turbid, CDOM-rich coastal water. The nearest Tanager water scene today is 426 km away. |
| **Clear-water reference basin** | AAOT, northern Adriatic | The canonical ocean-color validation platform, reporting in all 12 months, radiosonde 85 km. Contrasts with LISCO as the clearer-water end of the coastal range, and the nearest Tanager water scene is 648 km away. |

## Water targets

For water the constraint is sharper, because AERONET-OC is the reference used here to close the loop
on a water-leaving reflectance rather than on the aerosol alone. Six SeaPRISM platforms report
reliably and sit near a radiosonde, and not one of them has a Tanager water scene within 238
kilometers.

| platform | reliable months | radiosonde | nearest Tanager water scene |
|---|---|---|---|
| Banana River, Indian River Lagoon | 9 of 12 | 13.8 km | 238 km |
| Kemigawa Offshore, Tokyo Bay | 10 of 12 | 50.6 km | 289 km |
| LISCO, Long Island Sound | 12 of 12 | 41.5 km | 426 km |
| Chesapeake Bay | 12 of 12 | 97.0 km | 469 km |
| AAOT, northern Adriatic | 12 of 12 | 84.7 km | 648 km |
| Casablanca Platform, western Mediterranean | 12 of 12 | 97.8 km | 687 km |

## Full ranked list

The top 40 of the 435 scored candidates. The score rewards reporting reliability, record length, a
well-timed radiosonde, proximity to an AERONET-OC platform, and a surface the validatable archive
lacks.

| # | site | lat, lon | dominant surface | reliable months | obs-days (10 yr) | nearest radiosonde | nearest AERONET-OC | sun regime |
|---|---|---|---|---|---|---|---|---|
| 1 | NEON PUUM, Hawaii | 19.55, -155.32 | tree 42%, grass 27% | 12 | 1105 | 33 km (timed) | - | year-round |
| 2 | Murcia, Spain | 38.00, -1.17 | grass 32%, tree 28% | 12 | 2952 | 0 km (timed) | - | year-round |
| 3 | Chiba University, Japan | 35.62, 140.10 | built 34%, tree 27% | 12 | 2215 | 48 km (timed) | Kemigawa Offshore 7 km | year-round |
| 4 | Teide, Tenerife | 28.27, -16.64 | grass 26%, shrub 22% | 12 | 2107 | 26 km (timed) | - | year-round |
| 5 | Cairo EMA, Egypt | 30.08, 31.29 | built 48%, bare 26% | 12 | 2737 | 25 km (timed) | - | year-round |
| 6 | Mauna Loa, Hawaii | 19.54, -155.58 | bare 69%, grass 26% | 12 | 3232 | 58 km (timed) | - | year-round |
| 7 | Tamanrasset, Algeria | 22.79, 5.53 | bare 98% | 12 | 2916 | 9 km (timed) | - | year-round |
| 8 | Izaña, Tenerife | 28.31, -16.50 | water 32%, grass 23% | 12 | 3194 | 12 km (timed) | - | year-round |
| 9 | Sevilleta, New Mexico | 34.35, -106.89 | bare 38%, grass 34% | 12 | 2545 | 80 km | - | year-round |
| 10 | White Sands, New Mexico | 32.63, -106.34 | shrub 42%, bare 42% | 12 | 2731 | 91 km | - | year-round |
| 11 | Upington, South Africa | -28.38, 21.16 | shrub 45%, grass 43% | 12 | 2319 | 11 km (timed) | - | year-round |
| 12 | Birdsville, Australia | -25.90, 139.35 | bare 54%, grass 43% | 12 | 2587 | 581 km (timed) | - | year-round |
| 13 | Agia Marina, Cyprus | 35.04, 33.06 | tree 35%, grass 28% | 12 | 2593 | 33 km (timed) | - | year-round |
| 14 | Lake Lefroy, Australia | -31.25, 121.70 | tree 35%, bare 26% | 11 | 1702 | 58 km | - | year-round |
| 15 | La Paz, Bolivia | -16.54, -68.07 | grass 59%, bare 20% | 12 | 2855 | 372 km | - | year-round |
| 16 | Fukuoka, Japan | 33.52, 130.47 | tree 52%, built 24% | 12 | 2206 | 11 km (timed) | Ariake Tower 49 km | year-round |
| 17 | Nicosia, Cyprus | 35.14, 33.38 | cropland 38%, grass 32% | 12 | 2041 | 1 km (timed) | - | year-round |
| 18 | Madrid, Spain | 40.45, -3.72 | grass 35%, built 27% | 12 | 2784 | 14 km (timed) | - | year-round |
| 19 | Karachi, Pakistan | 24.95, 67.14 | built 29%, bare 26% | 12 | 1886 | 5 km | - | year-round |
| 20 | Banizoumbou, Niger | 13.55, 2.67 | grass 79%, bare 15% | 11 | 2120 | 66 km (timed) | - | year-round |
| 21 | Malindi, Kenya | -3.00, 40.19 | water 58%, grass 15% | 12 | 2205 | 426 km | San Marco Platform 6 km | year-round |
| 22 | El Arenosillo, Spain | 37.10, -6.73 | water 46%, grass 19% | 12 | 2728 | 25 km (timed) | - | year-round |
| 23 | Railroad Valley, Nevada | 38.50, -115.69 | bare 79%, shrub 8% | 12 | 3102 | 263 km | - | year-round |
| 24 | AAOT, northern Adriatic | 45.31, 12.51 | water 92% | 12 | 2577 | 85 km (timed) | AAOT 0 km | year-round |
| 25 | NEON MOAB, Utah | 38.25, -109.39 | shrub 41%, grass 25% | 12 | 2508 | 122 km | - | year-round |
| 26 | Medenine, Tunisia | 33.50, 10.64 | bare 44%, water 21% | 12 | 2433 | 234 km (timed) | - | year-round |
| 27 | Maricopa, Arizona | 33.07, -111.97 | bare 39%, shrub 26% | 12 | 1323 | 42 km | - | year-round |
| 28 | Dalanzadgad, Mongolia | 43.58, 104.42 | bare 86%, grass 12% | 12 | 2476 | 0 km | - | year-round |
| 29 | Rome La Sapienza, Italy | 41.90, 12.52 | tree 35%, cropland 23% | 12 | 2109 | 28 km (timed) | - | year-round |
| 30 | Rome Tor Vergata, Italy | 41.84, 12.65 | tree 45%, built 20% | 12 | 1989 | 26 km (timed) | - | year-round |
| 31 | Albuquerque, New Mexico | 35.05, -106.54 | grass 30%, tree 30% | 12 | 2091 | 8 km | - | year-round |
| 32 | A Coruña, Spain | 43.36, -8.42 | water 45%, tree 30% | 12 | 2049 | 0 km (timed) | - | year-round |
| 33 | Yuma, Arizona | 32.64, -114.58 | bare 56%, cropland 26% | 12 | 2307 | 27 km | - | year-round |
| 34 | LISCO, Long Island Sound | 40.95, -73.34 | water 50%, tree 35% | 12 | 1701 | 42 km | LISCO 0 km | year-round |
| 35 | Palma de Mallorca, Spain | 39.55, 2.63 | water 40%, tree 27% | 12 | 2669 | 9 km (timed) | - | year-round |
| 36 | Ispra, Italy | 45.80, 8.63 | tree 63%, grass 12% | 12 | 2568 | 30 km (timed) | - | year-round |
| 37 | NASA Kennedy Space Center | 28.46, -80.66 | water 52%, tree 26% | 12 | 1716 | 11 km | Banana River 11 km | year-round |
| 38 | Hankuk UFS, South Korea | 37.34, 127.27 | tree 62%, built 21% | 12 | 2463 | 34 km (timed) | - | year-round |
| 39 | Thessaloniki, Greece | 40.63, 22.96 | cropland 30%, tree 20% | 12 | 2372 | 11 km (timed) | - | year-round |
| 40 | San Marco Platform, Kenya | -2.94, 40.21 | water 62%, grass 13% | 12 | 1502 | 425 km | San Marco Platform 0 km | year-round |

## Caveats

The surface fractions come from a 2021 annual land-cover map, so they describe persistent cover
(built, bare, cropland, water) well but cannot speak to seasonal snow. The 40 km sampling box is
larger than a single Tanager swath, so a scene placed at a site can favor or avoid any surface the
box reports. AERONET inversion products, which constrain the aerosol model itself, and MPLNET lidar
co-location, which gives the aerosol vertical profile, were not scored and would refine the ranking.
The commercial and impact priorities that the Open Committee weighs are not represented here at all.
