"""Tests for the T2Pro camera class with mock hardware layer.

The T2Pro class uses an injectable hardware backend (VideoBackend) so that
all logic can be tested without a physical camera connected.
"""

import numpy as np
import pytest
import os
from infiray_t2pro.camera import T2Pro, VideoBackend
from infiray_t2pro.commands import Command
from infiray_t2pro.palettes import Palette


class FakeVideoBackend(VideoBackend):
    """A fake backend that returns synthetic frames for testing."""

    def __init__(self, thermal_values=None, camera_id=0):
        self.camera_id = camera_id
        self._thermal = thermal_values if thermal_values is not None else np.random.randint(
            3000, 9000, (192, 256), dtype=np.uint16
        )
        self.zoom_values_sent = []
        self.open_count = 0

    def open(self):
        self.open_count += 1

    def read_raw(self):
        """Return a raw 196×256×2 frame derived from the thermal values."""
        raw = np.zeros((196, 256, 2), dtype=np.uint8)
        lsb = (self._thermal & 0xFF).astype(np.uint8)
        msb = (self._thermal >> 8).astype(np.uint8)
        raw[:192, :, 0] = lsb
        raw[:192, :, 1] = msb
        raw[192:, :, :] = 0
        return raw

    def set_zoom(self, value):
        self.zoom_values_sent.append(int(value))

    def close(self):
        pass


class TestT2ProInit:
    def test_creates_with_default_camera_id(self):
        cam = T2Pro()
        assert cam.camera_id == 0

    def test_creates_with_custom_camera_id(self):
        cam = T2Pro(camera_id=2)
        assert cam.camera_id == 2

    def test_starts_without_nuc_calibration(self):
        cam = T2Pro()
        assert cam.nuc_calib is None

    def test_default_palette_is_inferno(self):
        cam = T2Pro()
        assert cam.palette == Palette.INFERNO


class TestT2ProCapture:
    def test_capture_returns_192x256_array(self):
        backend = FakeVideoBackend()
        cam = T2Pro(backend=backend)
        frame = cam.capture()
        assert frame.shape == (192, 256)

    def test_capture_applies_nuc_when_calibrated(self):
        """When NUC calibration is loaded, capture (after first frame) should subtract it."""
        thermal = np.full((192, 256), 5000, dtype=np.uint16)
        backend = FakeVideoBackend(thermal_values=thermal)

        cam = T2Pro(backend=backend)
        cam.nuc_calib = np.full((192, 256), 1000, dtype=np.uint16)

        # First capture skips NUC (first frame can be corrupted)
        frame1 = cam.capture()
        assert np.allclose(frame1, 5000)

        # Second capture applies NUC: 5000 - 1000 = 4000
        frame2 = cam.capture()
        assert np.allclose(frame2, 4000)

    def test_capture_without_nuc_returns_raw_values(self):
        thermal = np.full((192, 256), 5000, dtype=np.uint16)
        backend = FakeVideoBackend(thermal_values=thermal)
        cam = T2Pro(backend=backend)
        frame = cam.capture()
        assert np.allclose(frame, 5000)

    def test_capture_averages_multiple_frames(self):
        """Capture with n_frames > 1 should average frames."""
        thermal = np.full((192, 256), 5000, dtype=np.uint16)
        backend = FakeVideoBackend(thermal_values=thermal)
        cam = T2Pro(backend=backend)
        frame = cam.capture(n_frames=3)
        assert np.allclose(frame, 5000)


class TestT2ProCommands:
    def test_trigger_shutter_sends_correct_command(self):
        backend = FakeVideoBackend()
        cam = T2Pro(backend=backend)
        cam.trigger_shutter()
        assert Command.SHUTTER_CLOSE in [v for v in backend.zoom_values_sent]

    def test_send_command_sends_value(self):
        backend = FakeVideoBackend()
        cam = T2Pro(backend=backend)
        cam.send_command(Command.PALETTE_IRON)
        assert int(Command.PALETTE_IRON) in backend.zoom_values_sent

    def test_reset_sends_default_command(self):
        backend = FakeVideoBackend()
        cam = T2Pro(backend=backend)
        cam.reset()
        assert 1 in backend.zoom_values_sent  # Command.DEFAULT == 1


class TestT2ProNUC:
    def test_calibrate_nuc_manual_stores_calibration(self):
        thermal = np.full((192, 256), 5000, dtype=np.uint16)
        backend = FakeVideoBackend(thermal_values=thermal)
        cam = T2Pro(backend=backend)
        cam.calibrate_nuc_manual(n_frames=1)
        assert cam.nuc_calib is not None
        assert cam.nuc_calib.shape == (192, 256)

    def test_load_nuc_calibration_loads_array(self, tmp_path):
        calib = np.full((192, 256), 1000, dtype=np.uint16)
        path = tmp_path / "calib.npy"
        np.save(path, calib)

        cam = T2Pro()
        cam.load_nuc_calibration(str(path))
        assert cam.nuc_calib is not None
        assert np.array_equal(cam.nuc_calib, calib)


class TestT2ProRender:
    def test_render_returns_bgr_image(self):
        backend = FakeVideoBackend()
        cam = T2Pro(backend=backend)
        frame = cam.capture()
        image = cam.render(frame)
        assert image.ndim == 3
        assert image.shape[2] == 3

    def test_render_with_custom_palette(self):
        backend = FakeVideoBackend()
        cam = T2Pro(backend=backend)
        frame = cam.capture()
        image = cam.render(frame, palette=Palette.JET)
        assert image.ndim == 3