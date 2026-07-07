"""Tests for AGC (Automatic Gain Control) processing."""

import numpy as np
import pytest
from infiray_t2pro.processing import AgcAutoRange


class TestAgcAutoRange:
    def test_first_frame_initializes_range(self):
        """First frame should initialize the lo/hi range from percentiles."""
        agc = AgcAutoRange(low_percentile=1.0, high_percentile=99.0)
        frame = np.random.randint(1000, 8000, (192, 256), dtype=np.uint16)
        result = agc.apply(frame)
        assert result.shape == (192, 256)
        assert result.dtype == np.uint8
        assert agc._lo is not None
        assert agc._hi is not None

    def test_output_is_uint8(self):
        """Output should be uint8 in 0-255 range."""
        agc = AgcAutoRange()
        frame = np.random.randint(3000, 7000, (192, 256), dtype=np.uint16)
        result = agc.apply(frame)
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_smooth_adaptation(self):
        """Range should adapt slowly, not jump to match each frame."""
        agc = AgcAutoRange(adapt_speed=0.1)

        # Frame 1: values 3000-7000
        frame1 = np.full((192, 256), 5000, dtype=np.uint16)
        frame1[:96] = 3000
        frame1[96:] = 7000
        agc.apply(frame1)
        lo1 = agc._lo
        hi1 = agc._hi

        # Frame 2: values 5000-9000 (shifted up by 2000)
        frame2 = np.full((192, 256), 7000, dtype=np.uint16)
        frame2[:96] = 5000
        frame2[96:] = 9000
        agc.apply(frame2)

        # Range should NOT jump to 5000-9000 immediately
        # It should only move 10% toward the new range
        assert agc._lo > lo1  # Moved toward 5000 but not all the way
        assert agc._hi > hi1  # Moved toward 9000 but not all the way
        assert agc._lo < 5000  # Should not have reached new lo yet
        assert agc._hi < 9000  # Should not have reached new hi yet

    def test_min_range_prevents_over_amplification(self):
        """When range is very narrow, enforce minimum range."""
        agc = AgcAutoRange(min_range=500.0, adapt_speed=1.0)

        # Uniform frame (zero range)
        frame = np.full((192, 256), 5000, dtype=np.uint16)
        result = agc.apply(frame)
        assert result.dtype == np.uint8

        # The enforced range should be at least min_range
        assert agc._hi - agc._lo >= 500.0

    def test_reset_clears_range(self):
        """Reset should clear the tracked range so next frame initializes fresh."""
        agc = AgcAutoRange()
        frame = np.random.randint(3000, 7000, (192, 256), dtype=np.uint16)
        agc.apply(frame)
        assert agc._lo is not None

        agc.reset()
        assert agc._lo is None
        assert agc._hi is None

    def test_uniform_frame_gives_center_gray(self):
        """A completely uniform frame should produce middle-gray output."""
        agc = AgcAutoRange(min_range=100.0, adapt_speed=1.0)
        frame = np.full((192, 256), 5000, dtype=np.uint16)
        result = agc.apply(frame)
        # With min_range, center value should map to ~128
        assert abs(result.mean() - 128) < 10

    def test_temporal_consistency(self):
        """Identical consecutive frames should produce nearly identical output."""
        agc = AgcAutoRange(adapt_speed=0.05)
        frame = np.random.randint(3000, 7000, (192, 256), dtype=np.uint16)

        result1 = agc.apply(frame)
        result2 = agc.apply(frame)

        # Second frame output should be very close to first
        diff = np.abs(result1.astype(np.int16) - result2.astype(np.int16))
        assert diff.mean() < 5  # Less than 5 gray levels difference on average

    def test_handles_float32_input(self):
        """Should accept float32 input (e.g., NUC-corrected frames)."""
        agc = AgcAutoRange()
        frame = np.random.randint(3000, 7000, (192, 256)).astype(np.float32)
        result = agc.apply(frame)
        assert result.shape == (192, 256)
        assert result.dtype == np.uint8