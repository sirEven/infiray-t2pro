"""Tests for improved NUC correction and AGC.

The thermal image has two types of noise:
- Fixed Pattern Noise (FPN): per-pixel offset differences (vertical stripes)
- Gain variation: per-pixel gain differences (some pixels brighter/darker)

Two-point NUC fixes both: (raw - dark) * gain_map
Where gain_map normalizes each pixel to the mean response.
"""

import numpy as np
import pytest
from infiray_t2pro.camera import T2Pro, VideoBackend
from infiray_t2pro.processing import two_point_nuc, agc_linear, agc_percentile, correct_column_fpn


class FakeVideoBackend(VideoBackend):
    """Minimal fake backend for processing tests."""
    def __init__(self, thermal_values=None):
        self._thermal = thermal_values if thermal_values is not None else np.full(
            (192, 256), 5000, dtype=np.uint16
        )
    def open(self): pass
    def read_raw(self):
        raw = np.zeros((196, 256, 2), dtype=np.uint8)
        raw[:192, :, 0] = (self._thermal & 0xFF).astype(np.uint8)
        raw[:192, :, 1] = (self._thermal >> 8).astype(np.uint8)
        return raw
    def set_zoom(self, value): pass
    def close(self): pass


class TestTwoPointNUC:
    def test_flat_field_after_correction(self):
        """A uniform scene should produce a flat output after two-point NUC.

        We use KNOWN gain maps (not random) so that dark/bright/raw share
        the same per-pixel gain — this is what two-point NUC corrects.
        """
        np.random.seed(42)
        # Per-pixel gain variation (0.9 to 1.1)
        gain_map = np.random.uniform(0.9, 1.1, (192, 256)).astype(np.float32)
        # Per-pixel offset (FPN)
        offsets = np.random.uniform(1000, 1100, (192, 256)).astype(np.float32)

        # Dark frame: just offsets (lens covered)
        dark = offsets.copy()
        # Bright frame: offsets + uniform scene (5000) * gain
        bright = offsets + 5000 * gain_map
        # Raw frame: offsets + uniform scene (3000) * gain
        raw = offsets + 3000 * gain_map

        corrected = two_point_nuc(raw, dark, bright)

        # After two-point NUC, the gain and offset variation should be removed
        # Result should be ~3000 * mean_response / mean(5000 * gain_map) * gain_map / gain_map
        # = 3000 * mean_response (roughly)
        # The corrected std should be much smaller than raw std
        assert corrected.std() < raw.std() * 0.15  # At least 85% noise reduction

    def test_dark_subtraction_only(self):
        """With uniform gain (bright-dark is constant), two-point NUC reduces to one-point."""
        dark = np.full((192, 256), 1000, dtype=np.float32)
        bright = np.full((192, 256), 6000, dtype=np.float32)  # uniform gain: all pixels +5000
        raw = np.full((192, 256), 4000, dtype=np.float32)  # offset + 3000

        corrected = two_point_nuc(raw, dark, bright)
        # Should be 3000 * mean_response / (6000-1000) = 3000
        assert np.allclose(corrected, 3000, atol=1)

    def test_gain_correction_fixes_stripes(self):
        """Vertical stripes (column FPN) should be reduced by two-point NUC."""
        dark = np.zeros((192, 256), dtype=np.float32)
        # Simulate column-wise gain variation: odd columns are 10% brighter
        gain_map = np.ones((192, 256), dtype=np.float32)
        gain_map[:, ::2] = 1.0
        gain_map[:, 1::2] = 1.1

        bright = (5000 * gain_map)
        raw = (3000 * gain_map)

        corrected = two_point_nuc(raw, dark, bright)
        # After correction, odd and even columns should be nearly identical
        even_cols = corrected[:, 0::2].mean()
        odd_cols = corrected[:, 1::2].mean()
        assert abs(even_cols - odd_cols) < 10  # Nearly identical


class TestAGCLinear:
    def test_scales_to_full_range(self):
        """AGC should scale input to fill 0-255."""
        thermal = np.random.randint(3000, 8000, (192, 256), dtype=np.uint16).astype(np.float32)
        result = agc_linear(thermal)
        assert result.min() == 0
        assert result.max() == 255
        assert result.dtype == np.uint8

    def test_uniform_input(self):
        """Uniform input should not crash."""
        thermal = np.full((192, 256), 5000, dtype=np.float32)
        result = agc_linear(thermal)
        assert result.dtype == np.uint8
        assert np.all(result == result[0, 0])

    def test_preserves_relative_order(self):
        """Pixels that were hotter should stay hotter after AGC."""
        thermal = np.zeros((192, 256), dtype=np.float32)
        thermal[0, 0] = 3000
        thermal[0, 1] = 7000
        result = agc_linear(thermal)
        assert result[0, 1] > result[0, 0]


class TestAGCPercentile:
    def test_clips_outliers(self):
        """Percentile AGC should ignore outlier pixels and use the full range for the bulk."""
        # Create data with 5% cold outliers and 5% hot outliers
        thermal = np.full((192, 256), 5000, dtype=np.float32)
        # 5% cold pixels
        thermal[:10, :] = 1000
        # 5% hot pixels
        thermal[-10:, :] = 9000
        # The 5th and 95th percentiles should clip these

        result = agc_percentile(thermal, low_percentile=5, high_percentile=95)
        assert result.dtype == np.uint8
        # The bulk of the image (5000) should map to mid-range
        assert 50 < result[96, 128] < 205

    def test_same_as_linear_at_0_100_percentile(self):
        """Percentile AGC with 0/100 percentiles should match linear AGC."""
        thermal = np.random.randint(3000, 8000, (192, 256), dtype=np.uint16).astype(np.float32)
        result_linear = agc_linear(thermal)
        result_pct = agc_percentile(thermal, low_percentile=0, high_percentile=100)
        assert np.allclose(result_linear, result_pct, atol=1)

    def test_preserves_relative_order_with_spread(self):
        """Hotter pixels stay hotter when there's enough dynamic range."""
        thermal = np.random.randint(3000, 8000, (192, 256), dtype=np.uint16).astype(np.float32)
        # Make one pixel notably hotter and one notably cooler
        thermal[0, 0] = 2000  # Cold spot
        thermal[0, 1] = 9000  # Hot spot
        result = agc_percentile(thermal)
        assert result[0, 1] > result[0, 0]


class TestColumnFPN:
    def test_removes_vertical_stripes(self):
        """Column FPN correction should flatten column-wise offsets."""
        np.random.seed(42)
        # Base uniform scene + per-column offset (vertical stripes)
        col_offsets = np.random.uniform(-200, 200, 256).astype(np.float32)
        base = np.full((192, 256), 5000, dtype=np.float32)
        striped = base + col_offsets[np.newaxis, :]

        corrected = correct_column_fpn(striped)
        # After correction, all columns should have the same mean
        col_means_after = corrected.mean(axis=0)
        assert col_means_after.std() < 1.0  # Nearly flat

    def test_preserves_horizontal_features(self):
        """Column FPN correction should not remove horizontal temperature gradients."""
        thermal = np.zeros((192, 256), dtype=np.float32)
        # Hot horizontal band in the middle
        thermal[96, :] = 8000
        thermal[:, :] += 5000

        corrected = correct_column_fpn(thermal)
        # The hot band should still be visible
        assert corrected[96, 128] > corrected[0, 128]

    def test_uniform_input_stays_uniform(self):
        """A perfectly uniform input should not be changed."""
        thermal = np.full((192, 256), 5000, dtype=np.float32)
        corrected = correct_column_fpn(thermal)
        assert np.allclose(corrected, 5000, atol=0.1)