"""Tests for denoise_thermal bilateral filter."""

import numpy as np
import pytest
from infiray_t2pro.processing import denoise_thermal


class TestDenoiseThermal:
    def test_output_shape_matches_input(self):
        """Denoise should preserve shape."""
        thermal = np.random.randint(3000, 8000, (192, 256), dtype=np.uint16)
        result = denoise_thermal(thermal)
        assert result.shape == (192, 256)

    def test_output_is_float32(self):
        """Bilateral filter returns float32."""
        thermal = np.random.randint(3000, 8000, (192, 256), dtype=np.uint16)
        result = denoise_thermal(thermal)
        assert result.dtype == np.float32

    def test_reduces_noise(self):
        """Denoised image should have lower variance than noisy input."""
        # Create a smooth gradient with added noise
        x = np.linspace(3000, 7000, 256, dtype=np.float32)
        gradient = np.tile(x, (192, 1))
        noise = np.random.normal(0, 100, (192, 256)).astype(np.float32)
        noisy = gradient + noise

        denoised = denoise_thermal(noisy)
        # Variance should be lower after denoising
        assert np.var(denoised - gradient) < np.var(noisy - gradient)

    def test_preserves_edges(self):
        """Bilateral filter should preserve sharp temperature boundaries."""
        # Left half cold, right half hot — sharp edge
        frame = np.zeros((192, 256), dtype=np.float32)
        frame[:, 128:] = 6000.0

        denoised = denoise_thermal(frame, spatial_sigma=1.5, range_sigma=15.0)

        # The edge at column 128 should still be steep
        # Left side should be close to 0, right side close to 6000
        left_mean = denoised[:, 64].mean()
        right_mean = denoised[:, 192].mean()
        assert abs(left_mean) < 50  # Near zero
        assert abs(right_mean - 6000) < 50  # Near 6000

    def test_uniform_input_passes_through(self):
        """A perfectly uniform frame should remain uniform after denoise."""
        frame = np.full((192, 256), 5000.0, dtype=np.float32)
        result = denoise_thermal(frame)
        # All values should be ~5000 (within float precision)
        assert np.allclose(result, 5000.0, atol=1.0)

    def test_accepts_uint16_input(self):
        """Should handle uint16 input (raw sensor data)."""
        thermal = np.random.randint(3000, 8000, (192, 256), dtype=np.uint16)
        result = denoise_thermal(thermal)
        assert result.shape == (192, 256)
        assert result.dtype == np.float32

    def test_custom_sigmas(self):
        """Custom sigma parameters should affect output differently."""
        thermal = np.random.randint(3000, 8000, (192, 256), dtype=np.uint16).astype(np.float32)

        # Light smoothing
        light = denoise_thermal(thermal, spatial_sigma=0.5, range_sigma=5.0)
        # Heavy smoothing
        heavy = denoise_thermal(thermal, spatial_sigma=3.0, range_sigma=50.0)

        # Heavy smoothing should produce lower variance
        assert np.var(heavy) <= np.var(light)