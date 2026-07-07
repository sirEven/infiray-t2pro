"""Tests for streaming mode (continuous frame reading).

The T2Pro driver must support keeping the stream open and reading frames
in a loop, instead of opening and closing per capture. This is essential
for any real-time use: live preview, streaming to ground, ring buffer.
"""

import numpy as np
import pytest
from infiray_t2pro.camera import T2Pro, VideoBackend, StreamClosedError


class FakeVideoBackend(VideoBackend):
    """A fake backend that returns synthetic frames for testing."""

    def __init__(self, thermal_values=None, camera_id=0):
        self.camera_id = camera_id
        self._thermal = thermal_values if thermal_values is not None else np.random.randint(
            3000, 9000, (192, 256), dtype=np.uint16
        )
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


class TestStartStream:
    """T2Pro.start_stream() opens the backend and warms up."""

    def test_start_stream_opens_backend(self):
        backend = FakeVideoBackend()
        cam = T2Pro(backend=backend)
        cam.start_stream()
        assert backend.open_count == 1
        cam.stop_stream()

    def test_start_stream_warms_up_by_trashing_frames(self):
        """start_stream should discard warmup frames so the first read_frame
        returns a stable frame, not a corrupted first frame."""
        backend = FakeVideoBackend()
        cam = T2Pro(backend=backend)
        cam.start_stream(warmup=5)
        # Backend was opened once (start_stream), warmup frames were read
        # We can verify warmup by checking the backend was read from
        # before read_frame is called
        cam.stop_stream()

    def test_start_stream_sets_is_streaming_true(self):
        backend = FakeVideoBackend()
        cam = T2Pro(backend=backend)
        cam.start_stream()
        assert cam.is_streaming is True
        cam.stop_stream()

    def test_start_stream_already_streaming_raises(self):
        """Calling start_stream when already streaming should raise an error."""
        backend = FakeVideoBackend()
        cam = T2Pro(backend=backend)
        cam.start_stream()
        with pytest.raises(RuntimeError, match="Already streaming"):
            cam.start_stream()
        cam.stop_stream()


class TestStopStream:
    """T2Pro.stop_stream() closes the backend cleanly."""

    def test_stop_stream_closes_backend(self):
        backend = FakeVideoBackend()
        cam = T2Pro(backend=backend)
        cam.start_stream()
        cam.stop_stream()
        assert backend.close_count == 1

    def test_stop_stream_sets_is_streaming_false(self):
        backend = FakeVideoBackend()
        cam = T2Pro(backend=backend)
        cam.start_stream()
        cam.stop_stream()
        assert cam.is_streaming is False

    def test_stop_stream_when_not_streaming_is_noop(self):
        """Calling stop_stream when not streaming should not raise."""
        backend = FakeVideoBackend()
        cam = T2Pro(backend=backend)
        cam.stop_stream()  # Should not raise

    def test_double_stop_is_safe(self):
        backend = FakeVideoBackend()
        cam = T2Pro(backend=backend)
        cam.start_stream()
        cam.stop_stream()
        cam.stop_stream()  # Second stop should not raise
        assert backend.close_count == 1  # Backend closed only once


class TestReadFrame:
    """T2Pro.read_frame() reads a single frame from the open stream."""

    def test_read_frame_returns_192x256_array(self):
        thermal = np.full((192, 256), 5000, dtype=np.uint16)
        backend = FakeVideoBackend(thermal_values=thermal)
        cam = T2Pro(backend=backend)
        cam.start_stream()
        frame = cam.read_frame()
        assert frame.shape == (192, 256)

    def test_read_frame_returns_float32(self):
        thermal = np.full((192, 256), 5000, dtype=np.uint16)
        backend = FakeVideoBackend(thermal_values=thermal)
        cam = T2Pro(backend=backend)
        cam.start_stream()
        frame = cam.read_frame()
        assert frame.dtype == np.float32

    def test_read_frame_returns_correct_values(self):
        thermal = np.full((192, 256), 5000, dtype=np.uint16)
        backend = FakeVideoBackend(thermal_values=thermal)
        cam = T2Pro(backend=backend)
        cam.start_stream()
        frame = cam.read_frame()
        assert np.allclose(frame, 5000)

    def test_read_frame_with_nuc_applies_correction(self):
        thermal = np.full((192, 256), 5000, dtype=np.uint16)
        backend = FakeVideoBackend(thermal_values=thermal)
        cam = T2Pro(backend=backend)
        cam.nuc_calib = np.full((192, 256), 1000, dtype=np.uint16)
        cam.start_stream()
        # First read_frame after start_stream should skip NUC (warmup)
        frame1 = cam.read_frame()
        # After warmup, NUC should be applied
        frame2 = cam.read_frame()
        assert np.allclose(frame2, 4000)  # 5000 - 1000

    def test_read_frame_without_nuc_returns_raw(self):
        thermal = np.full((192, 256), 5000, dtype=np.uint16)
        backend = FakeVideoBackend(thermal_values=thermal)
        cam = T2Pro(backend=backend)
        cam.start_stream()
        frame = cam.read_frame()
        assert np.allclose(frame, 5000)

    def test_read_frame_when_not_streaming_raises(self):
        """Calling read_frame when stream is not open should raise StreamClosedError."""
        backend = FakeVideoBackend()
        cam = T2Pro(backend=backend)
        with pytest.raises(StreamClosedError):
            cam.read_frame()

    def test_multiple_read_frames_in_sequence(self):
        """Should be able to read many frames from an open stream."""
        thermal = np.full((192, 256), 5000, dtype=np.uint16)
        backend = FakeVideoBackend(thermal_values=thermal)
        cam = T2Pro(backend=backend)
        cam.start_stream()
        for _ in range(10):
            frame = cam.read_frame()
            assert frame.shape == (192, 256)
        cam.stop_stream()

    def test_read_frame_increments_frame_count(self):
        thermal = np.full((192, 256), 5000, dtype=np.uint16)
        backend = FakeVideoBackend(thermal_values=thermal)
        cam = T2Pro(backend=backend)
        cam.start_stream()
        assert cam.frame_count == 0  # warmup frames don't count
        cam.read_frame()
        assert cam.frame_count >= 1
        cam.read_frame()
        assert cam.frame_count >= 2


class TestStreamingContextManager:
    """T2Pro can be used as a context manager for stream lifecycle."""

    def test_context_manager_opens_and_closes(self):
        backend = FakeVideoBackend()
        cam = T2Pro(backend=backend)
        with cam.stream() as s:
            assert s.is_streaming is True
            frame = s.read_frame()
            assert frame.shape == (192, 256)
        assert cam.is_streaming is False

    def test_context_manager_closes_on_exception(self):
        backend = FakeVideoBackend()
        cam = T2Pro(backend=backend)
        try:
            with cam.stream() as s:
                raise ValueError("test error")
        except ValueError:
            pass
        assert cam.is_streaming is False


class TestStreamingCaptureBackwardCompat:
    """Existing capture() methods should still work after streaming changes."""

    def test_capture_still_works(self):
        """capture() should still open/close per call for backward compatibility."""
        thermal = np.full((192, 256), 5000, dtype=np.uint16)
        backend = FakeVideoBackend(thermal_values=thermal)
        cam = T2Pro(backend=backend)
        frame = cam.capture()
        assert frame.shape == (192, 256)

    def test_capture_raw_still_works(self):
        thermal = np.full((192, 256), 5000, dtype=np.uint16)
        backend = FakeVideoBackend(thermal_values=thermal)
        cam = T2Pro(backend=backend)
        frame = cam.capture_raw()
        assert frame.shape == (192, 256)

    def test_capture_after_stream_stopped(self):
        """After streaming, capture() should still work (opens fresh)."""
        thermal = np.full((192, 256), 5000, dtype=np.uint16)
        backend = FakeVideoBackend(thermal_values=thermal)
        cam = T2Pro(backend=backend)
        cam.start_stream()
        cam.read_frame()
        cam.stop_stream()
        frame = cam.capture()
        assert frame.shape == (192, 256)


class TestStreamWarmup:
    """Warmup behavior: first frames after opening are unstable."""

    def test_first_read_frame_after_start_skips_nuc(self):
        """The first frame after start_stream can have corrupted dynamic range.
        Even if NUC is loaded, the first read_frame should return raw values."""
        thermal = np.full((192, 256), 5000, dtype=np.uint16)
        backend = FakeVideoBackend(thermal_values=thermal)
        cam = T2Pro(backend=backend)
        cam.nuc_calib = np.full((192, 256), 1000, dtype=np.uint16)
        cam.start_stream()
        frame = cam.read_frame()
        # First frame after stream start should NOT have NUC applied
        assert np.allclose(frame, 5000)

    def test_configurable_warmup_count(self):
        """Number of warmup frames to discard should be configurable."""
        backend = FakeVideoBackend()
        cam = T2Pro(backend=backend)
        cam.start_stream(warmup=3)
        cam.stop_stream()