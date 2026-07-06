"""Tests for thermal image palette rendering."""

import numpy as np
import pytest
from infiray_t2pro.palettes import Palette, apply_palette


class TestPaletteEnum:
    def test_inferno_exists(self):
        assert Palette.INFERNO is not None

    def test_jet_exists(self):
        assert Palette.JET is not None

    def test_grayscale_value(self):
        assert Palette.GRAYSCALE == -1


class TestApplyPalette:
    def test_returns_bgr_image(self):
        """Output should be a 3-channel BGR image."""
        thermal = np.random.randint(0, 65535, (192, 256), dtype=np.uint16)
        result = apply_palette(thermal, Palette.INFERNO)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_output_is_uint8(self):
        thermal = np.random.randint(0, 65535, (192, 256), dtype=np.uint16)
        result = apply_palette(thermal, Palette.INFERNO)
        assert result.dtype == np.uint8

    def test_output_is_scaled_up(self):
        """Output should be 4x the input resolution for visibility."""
        thermal = np.random.randint(0, 65535, (192, 256), dtype=np.uint16)
        result = apply_palette(thermal, Palette.INFERNO)
        assert result.shape[0] == 192 * 4
        assert result.shape[1] == 256 * 4

    def test_uniform_input_produces_uniform_output(self):
        """A uniform thermal field should produce a uniform color image."""
        thermal = np.full((192, 256), 5000, dtype=np.uint16)
        result = apply_palette(thermal, Palette.INFERNO)
        # All pixels should be the same color
        unique_colors = np.unique(result.reshape(-1, 3), axis=0)
        assert len(unique_colors) == 1

    def test_grayscale_produces_grayscale(self):
        """Grayscale palette should produce equal R, G, B channels."""
        thermal = np.random.randint(0, 65535, (192, 256), dtype=np.uint16)
        result = apply_palette(thermal, Palette.GRAYSCALE)
        b, g, r = result[:, :, 0], result[:, :, 1], result[:, :, 2]
        assert np.array_equal(b, g)
        assert np.array_equal(g, r)

    def test_different_inputs_produce_different_outputs(self):
        """Two different thermal patterns should produce different images."""
        thermal_a = np.zeros((192, 256), dtype=np.uint16)
        thermal_a[:96] = 1000      # top half cold
        thermal_a[96:] = 60000     # bottom half hot
        thermal_b = np.zeros((192, 256), dtype=np.uint16)
        thermal_b[:96] = 60000     # top half hot
        thermal_b[96:] = 1000      # bottom half cold
        result_a = apply_palette(thermal_a, Palette.INFERNO)
        result_b = apply_palette(thermal_b, Palette.INFERNO)
        assert not np.array_equal(result_a, result_b)