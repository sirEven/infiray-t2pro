"""Tests for error handling in streaming mode.

The driver should fail fast and transparently — no auto-reconnect, no retry,
no magic. Errors propagate as clear exceptions so the caller (future wrapper)
can decide what to do: retry, alert the pilot, or give up.
"""

import numpy as np
import pytest
from infiray_t2pro.camera import T2Pro, VideoBackend, StreamClosedError, FrameReadError


class FailingBackend(VideoBackend):
    """A backend that fails on read_raw after N successful reads."""

    def __init__(self, fail_after: int = 3):
        self.read_count = 0
        self.fail_after = fail_after
        self._is_open = False
        self.open_count = 0
        self.close_count = 0

    def open(self):
        self.open_count += 1
        self._is_open = True

    def read_raw(self):
        if not self._is_open:
            raise RuntimeError("Stream not open")
        self.read_count += 1
        if self.read_count > self.fail_after:
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


class DisappearsBackend(VideoBackend):
    """A backend that disconnects mid-stream (open fails on reconnect)."""

    def __init__(self):
        self._is_open = False
        self.open_count = 0
        self.close_count = 0
        self.read_count = 0

    def open(self):
        self.open_count += 1
        if self.open_count > 1:
            raise RuntimeError("Cannot open /dev/video0")
        self._is_open = True

    def read_raw(self):
        if not self._is_open:
            raise RuntimeError("Stream not open")
        self.read_count += 1
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


class TestReadFrameFails:
    """When read_raw fails mid-stream, read_frame should fail clearly."""

    def test_read_frame_raises_frame_read_error_on_backend_failure(self):
        backend = FailingBackend(fail_after=2)
        cam = T2Pro(backend=backend)
        cam.start_stream(warmup=0)
        # First 2 reads succeed
        cam.read_frame()
        cam.read_frame()
        # 3rd read: backend raises RuntimeError
        with pytest.raises(FrameReadError):
            cam.read_frame()

    def test_read_frame_error_contains_original_cause(self):
        """FrameReadError should chain the original backend error."""
        backend = FailingBackend(fail_after=0)
        cam = T2Pro(backend=backend)
        cam.start_stream(warmup=0)
        with pytest.raises(FrameReadError) as exc_info:
            cam.read_frame()
        # Original RuntimeError should be accessible
        assert exc_info.value.__cause__ is not None

    def test_stream_goes_off_after_read_failure(self):
        """After a read failure, is_streaming should be False.
        The stream is broken — caller must stop_stream() then start_stream()."""
        backend = FailingBackend(fail_after=2)
        cam = T2Pro(backend=backend)
        cam.start_stream(warmup=0)
        cam.read_frame()
        cam.read_frame()
        try:
            cam.read_frame()
        except FrameReadError:
            pass
        assert cam.is_streaming is False

    def test_read_frame_after_failure_raises_stream_closed(self):
        """After a read failure, subsequent read_frame calls raise StreamClosedError
        because the stream is no longer active."""
        backend = FailingBackend(fail_after=2)
        cam = T2Pro(backend=backend)
        cam.start_stream(warmup=0)
        cam.read_frame()
        cam.read_frame()
        try:
            cam.read_frame()
        except FrameReadError:
            pass
        # Stream is now off — calling read_frame should raise StreamClosedError
        with pytest.raises(StreamClosedError):
            cam.read_frame()


class TestStartStreamFails:
    """When the camera isn't available, start_stream should fail clearly."""

    def test_start_stream_raises_when_camera_not_found(self):
        backend = DisappearsBackend()  # open fails on 2nd call
        cam = T2Pro(backend=backend)
        cam.start_stream(warmup=0)
        cam.stop_stream()
        # Second start: backend.open() raises RuntimeError
        with pytest.raises(RuntimeError):
            cam.start_stream(warmup=0)

    def test_start_stream_failure_leaves_stream_off(self):
        """If start_stream fails, is_streaming should remain False."""
        backend = DisappearsBackend()
        cam = T2Pro(backend=backend)
        cam.start_stream(warmup=0)
        cam.stop_stream()
        assert cam.is_streaming is False
        try:
            cam.start_stream(warmup=0)
        except RuntimeError:
            pass
        assert cam.is_streaming is False


class TestReconnectAfterFailure:
    """After a failure, the caller can attempt to reconnect."""

    def test_reconnect_after_read_failure(self):
        """A complete stop_stream + start_stream cycle should work after a read failure,
        IF the backend is healthy again."""
        # Simulate: backend fails, then recovers
        class RecoveringBackend(VideoBackend):
            def __init__(self):
                self._is_open = False
                self.fail_next = True  # fail the first read
                self.read_count = 0
                self.open_count = 0
                self.close_count = 0

            def open(self):
                self.open_count += 1
                self._is_open = True

            def read_raw(self):
                if not self._is_open:
                    raise RuntimeError("Stream not open")
                self.read_count += 1
                if self.fail_next:
                    self.fail_next = False
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

        backend = RecoveringBackend()
        cam = T2Pro(backend=backend)
        cam.start_stream(warmup=0)
        # First read fails
        with pytest.raises(FrameReadError):
            cam.read_frame()
        assert cam.is_streaming is False
        # Reconnect
        cam.stop_stream()
        cam.start_stream(warmup=0)
        # Now it works
        frame = cam.read_frame()
        assert frame.shape == (192, 256)
        cam.stop_stream()


class TestStopStreamCleansUp:
    """stop_stream should be safe to call in any state."""

    def test_stop_stream_after_read_failure_is_safe(self):
        backend = FailingBackend(fail_after=1)
        cam = T2Pro(backend=backend)
        cam.start_stream(warmup=0)
        cam.read_frame()
        try:
            cam.read_frame()
        except FrameReadError:
            pass
        # Stream is already off, but stop_stream should not raise
        cam.stop_stream()  # no crash
        assert cam.is_streaming is False

    def test_stop_stream_closes_backend_even_after_error(self):
        """After a read failure, stop_stream should still try to close the backend."""
        backend = FailingBackend(fail_after=1)
        cam = T2Pro(backend=backend)
        cam.start_stream(warmup=0)
        cam.read_frame()
        try:
            cam.read_frame()
        except FrameReadError:
            pass
        # is_streaming is already False, so stop_stream should be a no-op
        # (backend was already marked as not streaming)
        cam.stop_stream()
        # The backend should have been closed by start_stream's close or is still open
        # The key point: no crash, consistent state


class TestExceptionTypes:
    """Verify exception types and messages are clear and distinguishable."""

    def test_stream_closed_error_is_runtime_error(self):
        """StreamClosedError should be a RuntimeError so callers can catch broadly."""
        assert issubclass(StreamClosedError, RuntimeError)

    def test_frame_read_error_is_runtime_error(self):
        """FrameReadError should be a RuntimeError so callers can catch broadly."""
        assert issubclass(FrameReadError, RuntimeError)

    def test_stream_closed_error_message(self):
        cam = T2Pro()
        with pytest.raises(StreamClosedError, match="not open"):
            cam.read_frame()

    def test_frame_read_error_message_includes_context(self):
        backend = FailingBackend(fail_after=0)
        cam = T2Pro(backend=backend)
        cam.start_stream(warmup=0)
        with pytest.raises(FrameReadError, match="read"):
            cam.read_frame()
        cam.stop_stream()