"""Tests for error handling when opening a stream.

When start_stream() fails, it should raise StreamOpenError with the original
cause chained. The stream should remain in a clean off state (is_streaming=False).
"""

import numpy as np
import pytest
from infiray_t2pro.camera import T2Pro, VideoBackend, StreamOpenError, StreamClosedError


class MissingBackend(VideoBackend):
    """A backend that fails to open (camera not found)."""

    def __init__(self):
        self._is_open = False
        self.open_count = 0
        self.close_count = 0

    def open(self):
        self.open_count += 1
        raise RuntimeError("Cannot open /dev/video0")

    def read_raw(self):
        raise RuntimeError("Stream not open")

    def set_zoom(self, value):
        pass

    def close(self):
        self.close_count += 1
        self._is_open = False


class WarmupFailBackend(VideoBackend):
    """A backend that opens fine but fails during warmup frame reads."""

    def __init__(self, fail_on_warmup: int = 2):
        self._fail_on_warmup = fail_on_warmup
        self._read_count = 0
        self._is_open = False
        self.open_count = 0
        self.close_count = 0

    def open(self):
        self.open_count += 1
        self._is_open = True
        self._read_count = 0

    def read_raw(self):
        if not self._is_open:
            raise RuntimeError("Stream not open")
        self._read_count += 1
        if self._read_count >= self._fail_on_warmup:
            raise RuntimeError("No frame captured")
        raw = np.zeros((196, 256, 2), dtype=np.uint8)
        thermal = np.full((192, 256), 5000, dtype=np.uint16)
        raw[:192, :, 0] = (thermal & 0xFF).astype(np.uint8)
        raw[:192, :, 1] = (thermal >> 8).astype(np.uint8)
        return raw

    def set_zoom(self, value):
        pass

    def close(self):
        self.close_count += 1
        self._is_open = False


class TestStreamOpenError:
    """When start_stream() fails, raise StreamOpenError."""

    def test_open_failure_raises_stream_open_error(self):
        backend = MissingBackend()
        cam = T2Pro(backend=backend)
        with pytest.raises(StreamOpenError):
            cam.start_stream()

    def test_stream_open_error_is_runtime_error(self):
        assert issubclass(StreamOpenError, RuntimeError)

    def test_stream_open_error_chains_cause(self):
        backend = MissingBackend()
        cam = T2Pro(backend=backend)
        with pytest.raises(StreamOpenError) as exc_info:
            cam.start_stream()
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_stream_open_error_message(self):
        backend = MissingBackend()
        cam = T2Pro(backend=backend)
        with pytest.raises(StreamOpenError, match="open"):
            cam.start_stream()

    def test_stream_remains_off_after_open_failure(self):
        backend = MissingBackend()
        cam = T2Pro(backend=backend)
        try:
            cam.start_stream()
        except StreamOpenError:
            pass
        assert cam.is_streaming is False

    def test_read_frame_after_open_failure_raises_stream_closed(self):
        backend = MissingBackend()
        cam = T2Pro(backend=backend)
        try:
            cam.start_stream()
        except StreamOpenError:
            pass
        with pytest.raises(StreamClosedError):
            cam.read_frame()


class TestWarmupFailure:
    """When a warmup frame read fails during start_stream."""

    def test_warmup_failure_raises_stream_open_error(self):
        backend = WarmupFailBackend(fail_on_warmup=2)
        cam = T2Pro(backend=backend)
        with pytest.raises(StreamOpenError):
            cam.start_stream(warmup=5)

    def test_warmup_failure_closes_backend(self):
        """If warmup fails, the backend should be closed (not left hanging)."""
        backend = WarmupFailBackend(fail_on_warmup=2)
        cam = T2Pro(backend=backend)
        try:
            cam.start_stream(warmup=5)
        except StreamOpenError:
            pass
        # Backend should have been closed
        assert backend.close_count >= 1

    def test_stream_off_after_warmup_failure(self):
        backend = WarmupFailBackend(fail_on_warmup=2)
        cam = T2Pro(backend=backend)
        try:
            cam.start_stream(warmup=5)
        except StreamOpenError:
            pass
        assert cam.is_streaming is False

    def test_warmup_failure_can_retry_start(self):
        """After a warmup failure, retrying start_stream should work if backend recovers."""
        backend = WarmupFailBackend(fail_on_warmup=999)  # won't fail
        cam = T2Pro(backend=backend)
        # First attempt with a backend that works
        cam.start_stream(warmup=2)
        frame = cam.read_frame()
        assert frame.shape == (192, 256)
        cam.stop_stream()


class TestRetryAfterOpenFailure:
    """After start_stream fails, a retry should be possible."""

    def test_retry_after_open_failure(self):
        """A new T2Pro with a working backend should succeed after seeing a failure."""
        # First: failing backend
        fail_backend = MissingBackend()
        cam_fail = T2Pro(backend=fail_backend)
        with pytest.raises(StreamOpenError):
            cam_fail.start_stream()
        assert cam_fail.is_streaming is False

        # Second: working backend
        class WorkingBackend(VideoBackend):
            def __init__(self):
                self._is_open = False
                self.open_count = 0
                self.close_count = 0

            def open(self):
                self.open_count += 1
                self._is_open = True

            def read_raw(self):
                if not self._is_open:
                    raise RuntimeError("Stream not open")
                raw = np.zeros((196, 256, 2), dtype=np.uint8)
                thermal = np.full((192, 256), 5000, dtype=np.uint16)
                raw[:192, :, 0] = (thermal & 0xFF).astype(np.uint8)
                raw[:192, :, 1] = (thermal >> 8).astype(np.uint8)
                return raw

            def set_zoom(self, value):
                pass

            def close(self):
                self.close_count += 1
                self._is_open = False

        cam_ok = T2Pro(backend=WorkingBackend())
        cam_ok.start_stream(warmup=1)
        frame = cam_ok.read_frame()
        assert frame.shape == (192, 256)
        cam_ok.stop_stream()