"""Tests for manuscript-specific metric helpers."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.manuscript.manuscript_metrics import (
    emission_region,
    wavelength_nm_to_ev,
)


def test_wavelength_to_ev_conversion() -> None:
    converted = wavelength_nm_to_ev([620.0, 500.0])
    assert np.allclose(converted, [2.0, 2.48])


@pytest.mark.parametrize(
    ("wavelength", "expected"),
    [
        (399.9, "UV"),
        (400, "blue"),
        (499, "blue"),
        (500, "green"),
        (559, "green"),
        (560, "yellow/orange"),
        (619, "yellow/orange"),
        (620, "red/NIR"),
    ],
)
def test_region_binning(wavelength: float, expected: str) -> None:
    assert emission_region(wavelength) == expected
