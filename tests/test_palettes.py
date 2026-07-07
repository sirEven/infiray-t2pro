"""Tests for thermal image palette rendering."""

import numpy as np
import pytest
from infiray_t2pro.palettes import Palette, apply_palette


class TestPaletteEnum:
    def test_inferno_exists(self):
        assert Palette.INFERNO is not None

    def test_jet_exists(self):
        assert Palette.JET is not None

    def test_white_hot_value(self):
        assert Palette.WHITE_HOT == -1

    def test_black_hot_value(self):
        assert Palette.BLACK_HOT == -2


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
        """Output should be scaled up for visibility (default 5x)."""
        thermal = np.random.randint(0, 65535, (192, 256), dtype=np.uint16)
        result = apply_palette(thermal, Palette.INFERNO)
        assert result.shape[0] == 192 * 5
        assert result.shape[1] == 256 * 5

    def test_custom_scale(self):
        """Scale parameter controls output resolution."""
        thermal = np.random.randint(0, 65535, (192, 256), dtype=np.uint16)
        result = apply_palette(thermal, Palette.INFERNO, scale=3)
        assert result.shape[0] == 192 * 3
        assert result.shape[1] == 256 * 3

    def test_uniform_input_produces_uniform_output(self):
        """A uniform thermal field should produce a uniform color image."""
        thermal = np.full((192, 256), 5000, dtype=np.uint16)
        result = apply_palette(thermal, Palette.INFERNO)
        # All pixels should be the same color
        unique_colors = np.unique(result.reshape(-1, 3), axis=0)
        assert len(unique_colors) == 1

    def test_white_hot_produces_grayscale(self):
        """White Hot palette should produce equal R, G, B channels."""
        thermal = np.random.randint(0, 65535, (192, 256), dtype=np.uint16)
        result = apply_palette(thermal, Palette.WHITE_HOT)
        b, g, r = result[:, :, 0], result[:, :, 1], result[:, :, 2]
        assert np.array_equal(b, g)
        assert np.array_equal(g, r)

    def test_black_hot_produces_inverted_grayscale(self):
        """Black Hot palette should produce inverted grayscale (hot=dark)."""
        thermal = np.zeros((192, 256), dtype=np.uint16)
        thermal[:96] = 1000      # top half cold
        thermal[96:] = 60000     # bottom half hot
        result = apply_palette(thermal, Palette.BLACK_HOT)
        # In Black Hot, hot pixels should be darker (lower values)
        top_avg = result[:96].mean()
        bottom_avg = result[96:].mean()
        assert bottom_avg < top_avg

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