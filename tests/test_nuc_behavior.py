"""Tests for NUC auto-loading and first-frame skip behavior.

Credits: First-frame skip pattern inspired by julled/py_InfiRay_T2_Pro.
"""

import numpy as np
import pytest
import os
from infiray_t2pro.camera import T2Pro, VideoBackend


class FakeVideoBackend(VideoBackend):
    """Minimal fake backend for NUC behavior tests."""

    def __init__(self, thermal_values=None):
        self._thermal = thermal_values if thermal_values is not None else np.full(
            (192, 256), 5000, dtype=np.uint16
        )

    def open(self):
        pass

    def read_raw(self):
        raw = np.zeros((196, 256, 2), dtype=np.uint8)
        raw[:192, :, 0] = (self._thermal & 0xFF).astype(np.uint8)
        raw[:192, :, 1] = (self._thermal >> 8).astype(np.uint8)
        return raw

    def set_zoom(self, value):
        pass

    def close(self):
        pass


class TestNUCAutoLoad:
    def test_auto_loads_calibration_from_path(self, tmp_path):
        """If a calibration file exists at the given path, load it on init."""
        calib = np.full((192, 256), 1000, dtype=np.uint16)
        calib_path = str(tmp_path / "t2pro_nuc_calib.npy")
        np.save(calib_path, calib)

        cam = T2Pro(nuc_calib_path=calib_path)
        assert cam.nuc_calib is not None
        assert np.array_equal(cam.nuc_calib, calib)

    def test_no_calibration_when_file_missing(self, tmp_path):
        """If no calibration file exists, nuc_calib should be None."""
        cam = T2Pro(nuc_calib_path=str(tmp_path / "nonexistent.npy"))
        assert cam.nuc_calib is None

    def test_default_path_does_not_crash(self):
        """Default path should not crash if file doesn't exist."""
        cam = T2Pro(nuc_calib_path="nonexistent_calib_file.npy")
        assert cam.nuc_calib is None


class TestFirstFrameSkip:
    def test_capture_skips_nuc_on_first_frame(self):
        """First capture after init should NOT apply NUC (first frame can be corrupted)."""
        thermal = np.full((192, 256), 5000, dtype=np.uint16)
        backend = FakeVideoBackend(thermal_values=thermal)

        cam = T2Pro(backend=backend)
        cam.nuc_calib = np.full((192, 256), 1000, dtype=np.uint16)

        # First capture should NOT apply NUC (raw value 5000)
        frame1 = cam.capture()
        assert np.allclose(frame1, 5000)

        # Second capture SHOULD apply NUC (5000 - 1000 = 4000)
        frame2 = cam.capture()
        assert np.allclose(frame2, 4000)

    def test_capture_raw_always_returns_raw(self):
        """capture_raw should never apply NUC, regardless of frame number."""
        thermal = np.full((192, 256), 5000, dtype=np.uint16)
        backend = FakeVideoBackend(thermal_values=thermal)

        cam = T2Pro(backend=backend)
        cam.nuc_calib = np.full((192, 256), 1000, dtype=np.uint16)

        frame1 = cam.capture_raw(n_frames=1, trash=0)
        frame2 = cam.capture_raw(n_frames=1, trash=0)
        assert np.allclose(frame1, 5000)
        assert np.allclose(frame2, 5000)