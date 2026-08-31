import numpy as np
import pytest

from atmoresponse.recipes import as_algorithm, canopy_chlorophyll_rsi, cyanobacteria_index

WL = np.array([620.0, 665.0, 681.0, 704.0, 709.0, 815.0])
SPECTRA = [
    dict(zip(WL, [0.02, 0.026, 0.023, 0.030, 0.030, 0.120])),
    dict(zip(WL, [0.03, 0.020, 0.021, 0.028, 0.028, 0.150])),
]


def test_as_algorithm_matches_the_raw_recipe():
    algo = as_algorithm(canopy_chlorophyll_rsi)
    for spectrum in SPECTRA:
        wavelengths = np.fromiter(sorted(spectrum), dtype=float)
        reflectance = np.array([spectrum[w] for w in wavelengths])
        assert algo(spectrum) == pytest.approx(
            float(canopy_chlorophyll_rsi(reflectance, wavelengths)))


def test_as_algorithm_evaluate_many_agrees_with_scalar_path():
    algo = as_algorithm(canopy_chlorophyll_rsi)
    batched = algo.evaluate_many(SPECTRA)
    assert batched == pytest.approx([algo(s) for s in SPECTRA])


def test_as_algorithm_result_index_selects_a_tuple_element():
    algo = as_algorithm(cyanobacteria_index, result_index=0)
    ci_only = algo(SPECTRA[0])
    full = cyanobacteria_index(
        np.array([SPECTRA[0][w] for w in sorted(SPECTRA[0])]),
        np.fromiter(sorted(SPECTRA[0]), dtype=float))
    assert ci_only == pytest.approx(float(full[0]))
