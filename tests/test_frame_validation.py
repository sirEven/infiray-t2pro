"""Tests for frame validation during streaming.

The driver should detect and reject corrupt frames rather than silently
passing them through. Invalid frames should raise FrameReadError with
a clear message so the caller (future wrapper) can decide what to do.
"""

import numpy as np
import pytest
from infiray_t2pro.camera import (
    T2Pro, VideoBackend, StreamClosedError, FrameReadError
)


def _make_thermal_raw(thermal_2d: np.ndarray) -> np.ndarray:
    """Encode a 192x256 thermal array into raw YUYV-like bytes."""
    raw = np.zeros((196, 256, 2), dtype=np.uint8)
    raw[:192, :, 0] = (thermal_2d & 0xFF).astype(np.uint8)
    raw[:192, :, 1] = (thermal_2d >> 8).astype(np.uint8)
    return raw


def _realistic_thermal(base=5000, noise_std=50, seed=42):
    """Generate a realistic thermal frame with natural variance."""
    rng = np.random.RandomState(seed)
    return (base + rng.randn(192, 256) * noise_std).astype(np.float32).astype(np.uint16)


class RealisticBackend(VideoBackend):
    """Backend that returns realistic thermal frames with variance."""

    def __init__(self, seed=42):
        self._is_open = False
        self._seed = seed
        self._frame_count = 0

    def open(self):
        self._is_open = True
        self._frame_count = 0

    def read_raw(self):
        if not self._is_open:
            raise RuntimeError("Stream not open")
        self._frame_count += 1
        thermal = _realistic_thermal(base=5000 + self._frame_count * 10, seed=self._seed + self._frame_count)
        return _make_thermal_raw(thermal)

    def set_zoom(self, value):
        pass

    def close(self):
        self._is_open = False


class AllZerosBackend(VideoBackend):
    """Backend that returns all-zero frames (dead sensor)."""

    def __init__(self):
        self._is_open = False

    def open(self):
        self._is_open = True

    def read_raw(self):
        if not self._is_open:
            raise RuntimeError("Stream not open")
        return np.zeros((196, 256, 2), dtype=np.uint8)

    def set_zoom(self, value):
        pass

    def close(self):
        self._is_open = False


class UniformValueBackend(VideoBackend):
    """Backend that returns frames where every pixel is identical (stuck sensor)."""

    def __init__(self, value=8000):
        self._is_open = False
        self._value = value

    def open(self):
        self._is_open = True

    def read_raw(self):
        if not self._is_open:
            raise RuntimeError("Stream not open")
        thermal = np.full((192, 256), self._value, dtype=np.uint16)
        return _make_thermal_raw(thermal)

    def set_zoom(self, value):
        pass

    def close(self):
        self._is_open = False


class PartialCorruptBackend(VideoBackend):
    """Backend that returns good frames then a corrupt one."""

    def __init__(self, good_count=3):
        self._is_open = False
        self._good_count = good_count
        self._read_count = 0

    def open(self):
        self._is_open = True
        self._read_count = 0

    def read_raw(self):
        if not self._is_open:
            raise RuntimeError("Stream not open")
        self._read_count += 1
        if self._read_count <= self._good_count:
            thermal = _realistic_thermal(base=5000 + self._read_count * 100, seed=self._read_count)
            return _make_thermal_raw(thermal)
        else:
            # All zeros — dead frame
            return np.zeros((196, 256, 2), dtype=np.uint8)

    def set_zoom(self, value):
        pass

    def close(self):
        self._is_open = False


class TestAllZerosFrame:
    """An all-zero frame (every pixel = 0) is a dead sensor or corrupt read."""

    def test_all_zeros_raises_frame_read_error(self):
        backend = AllZerosBackend()
        cam = T2Pro(backend=backend)
        cam.start_stream(warmup=0)
        with pytest.raises(FrameReadError, match="[Zz]ero"):
            cam.read_frame()

    def test_stream_off_after_zero_frame(self):
        backend = AllZerosBackend()
        cam = T2Pro(backend=backend)
        cam.start_stream(warmup=0)
        try:
            cam.read_frame()
        except FrameReadError:
            pass
        assert cam.is_streaming is False

    def test_zero_frame_error_message_indicates_zeros(self):
        """For validation-detected errors, the message should describe the problem."""
        backend = AllZerosBackend()
        cam = T2Pro(backend=backend)
        cam.start_stream(warmup=0)
        with pytest.raises(FrameReadError, match="[Zz]ero"):
            cam.read_frame()


class TestUniformValueFrame:
    """A frame where every pixel has the same value (stuck sensor)."""

    def test_uniform_value_raises_frame_read_error(self):
        backend = UniformValueBackend(value=8000)
        cam = T2Pro(backend=backend)
        cam.start_stream(warmup=0)
        with pytest.raises(FrameReadError, match="[Uu]niform"):
            cam.read_frame()

    def test_stream_off_after_uniform_frame(self):
        backend = UniformValueBackend(value=8000)
        cam = T2Pro(backend=backend)
        cam.start_stream(warmup=0)
        try:
            cam.read_frame()
        except FrameReadError:
            pass
        assert cam.is_streaming is False


class TestPartialCorrupt:
    """Good frames followed by a corrupt one."""

    def test_good_frames_pass_then_corrupt_raises(self):
        backend = PartialCorruptBackend(good_count=3)
        cam = T2Pro(backend=backend)
        cam.start_stream(warmup=0)
        # First 3 frames should be fine
        for i in range(3):
            frame = cam.read_frame()
            assert frame.shape == (192, 256)
            assert frame.std() > 0
        # 4th frame is all zeros — should raise
        with pytest.raises(FrameReadError, match="[Zz]ero"):
            cam.read_frame()

    def test_can_reconnect_after_corrupt_frame(self):
        backend = PartialCorruptBackend(good_count=1)
        cam = T2Pro(backend=backend)
        cam.start_stream(warmup=0)
        frame = cam.read_frame()  # good
        assert frame.std() > 0
        with pytest.raises(FrameReadError):
            cam.read_frame()  # corrupt
        # Reconnect
        cam.stop_stream()
        cam.start_stream(warmup=0)
        frame2 = cam.read_frame()  # good again
        assert frame2.std() > 0


class TestValidFramesPass:
    """Normal frames with variance should pass validation."""

    def test_normal_frame_passes(self):
        """A frame with realistic thermal values and variance should not raise."""
        backend = RealisticBackend()
        cam = T2Pro(backend=backend)
        cam.start_stream(warmup=0)
        frame = cam.read_frame()
        assert frame.shape == (192, 256)
        assert frame.std() > 0

    def test_raw_frame_validated(self):
        """Validation also runs when apply_nuc=False."""
        backend = RealisticBackend()
        cam = T2Pro(backend=backend)
        cam.start_stream(warmup=0)
        frame = cam.read_frame(apply_nuc=False)
        assert frame.shape == (192, 256)
        assert frame.std() > 0


class TestValidationWithNUC:
    """Validation should work correctly when NUC correction is applied."""

    def test_valid_frame_with_nuc_passes(self):
        dark = np.full((192, 256), 500, dtype=np.float32)
        backend = RealisticBackend()
        cam = T2Pro(backend=backend)
        cam.nuc_calib = dark
        cam.start_stream(warmup=0)
        # Second frame gets NUC applied (first frame skips)
        cam.read_frame()  # warmup skip
        frame = cam.read_frame()
        assert frame.shape == (192, 256)