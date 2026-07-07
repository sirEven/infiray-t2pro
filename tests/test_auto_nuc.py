"""Tests for auto-NUC (automatic shutter calibration during streaming)."""

import time
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from infiray_t2pro.camera import T2Pro, VideoBackend, StreamClosedError
from infiray_t2pro.commands import Command


class FakeVideoBackend(VideoBackend):
    """A fake backend that returns synthetic frames for testing."""

    def __init__(self, thermal_values=None, camera_id=0):
        self.camera_id = camera_id
        if thermal_values is not None:
            self._thermal = thermal_values
        else:
            rng = np.random.RandomState(42)
            self._thermal = (rng.randint(3000, 9000, (192, 256))).astype(np.uint16)
        self.zoom_values_sent = []
        self.open_count = 0
        self.close_count = 0
        self._is_open = False

    def open(self):
        self.open_count += 1
        self._is_open = True

    def read_raw(self):
        if not self._is_open:
            raise RuntimeError("Stream not open")
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
        self.close_count += 1
        self._is_open = False


def _make_camera(**kwargs):
    backend = FakeVideoBackend()
    cam = T2Pro(backend=backend, **kwargs)
    cam._backend = backend
    return cam, backend


class TestAutoNuc:
    def test_auto_nuc_disabled_by_default(self):
        """Without auto_nuc, no shutter triggers during streaming."""
        cam, backend = _make_camera()
        cam.start_stream(warmup=1)
        for _ in range(50):
            cam.read_frame()
        cam.stop_stream()
        # No SHUTTER_CLOSE commands should have been sent
        assert Command.SHUTTER_CLOSE not in backend.zoom_values_sent

    def test_auto_nuc_triggers_after_interval(self):
        """With auto_nuc=0.1, shutter should trigger after 0.1s of streaming."""
        cam, backend = _make_camera()
        # Mock trigger_shutter so we don't actually sleep 0.5s
        nuc_calls = []
        cam.trigger_shutter = lambda: nuc_calls.append(time.monotonic())

        cam.start_stream(warmup=1, auto_nuc=0.1)
        # First frame — no NUC yet
        cam.read_frame()
        assert len(nuc_calls) == 0

        # Wait for interval to elapse
        time.sleep(0.15)
        cam.read_frame()

        # Now NUC should have been triggered
        assert len(nuc_calls) >= 1
        cam.stop_stream()

    def test_auto_nuc_does_not_trigger_before_interval(self):
        """Shutter should NOT trigger if interval hasn't elapsed yet."""
        cam, backend = _make_camera()
        nuc_calls = []
        cam.trigger_shutter = lambda: nuc_calls.append(1)

        cam.start_stream(warmup=1, auto_nuc=60.0)  # 60 seconds
        for _ in range(10):
            cam.read_frame()
        assert len(nuc_calls) == 0
        cam.stop_stream()

    def test_auto_nuc_triggers_multiple_times(self):
        """Shutter should trigger each time the interval elapses."""
        cam, backend = _make_camera()
        nuc_calls = []
        cam.trigger_shutter = lambda: nuc_calls.append(time.monotonic())

        cam.start_stream(warmup=1, auto_nuc=0.05)

        cam.read_frame()
        assert len(nuc_calls) == 0

        time.sleep(0.08)
        cam.read_frame()
        assert len(nuc_calls) >= 1

        time.sleep(0.08)
        cam.read_frame()
        assert len(nuc_calls) >= 2

        cam.stop_stream()

    def test_auto_nuc_resets_timer_after_trigger(self):
        """After a NUC trigger, the timer resets — no double-trigger."""
        cam, backend = _make_camera()
        nuc_calls = []
        cam.trigger_shutter = lambda: nuc_calls.append(1)

        cam.start_stream(warmup=1, auto_nuc=0.5)

        # Wait past interval
        time.sleep(0.55)
        cam.read_frame()  # Triggers NUC
        assert len(nuc_calls) == 1

        # Read another frame immediately — should NOT trigger again
        cam.read_frame()
        assert len(nuc_calls) == 1  # No additional NUC

        cam.stop_stream()

    def test_auto_nuc_none_means_disabled(self):
        """Explicitly passing auto_nuc=None disables auto-NUC."""
        cam, backend = _make_camera()
        cam.start_stream(warmup=1, auto_nuc=None)
        for _ in range(50):
            cam.read_frame()
        assert Command.SHUTTER_CLOSE not in backend.zoom_values_sent
        cam.stop_stream()

    def test_auto_nuc_interval_zero_triggers_every_frame(self):
        """auto_nuc=0 means NUC triggers on every frame (after first)."""
        cam, backend = _make_camera()
        nuc_calls = []
        cam.trigger_shutter = lambda: nuc_calls.append(1)

        cam.start_stream(warmup=1, auto_nuc=0)
        # First frame — timer was just set, 0s elapsed is >= 0 interval
        cam.read_frame()
        assert len(nuc_calls) >= 1  # Triggers immediately
        cam.stop_stream()