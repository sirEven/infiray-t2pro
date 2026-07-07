"""Tests for T2ProSession — high-level pipeline context manager."""

import numpy as np
import pytest
import tempfile
import os

from infiray_t2pro.camera import T2Pro, VideoBackend
from infiray_t2pro.session import T2ProSession


class FakeVideoBackend(VideoBackend):
    """A fake backend that returns synthetic frames for testing."""

    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self._frame_count = 0
        self._is_open = False
        self.zoom_values_sent = []

    def open(self):
        self._is_open = True

    def read_raw(self):
        if not self._is_open:
            raise RuntimeError("Stream not open")
        self._frame_count += 1
        raw = np.zeros((196, 256, 2), dtype=np.uint8)
        thermal = np.random.RandomState(42 + self._frame_count).randint(
            3000, 9000, (192, 256)
        ).astype(np.uint16)
        raw[:192, :, 0] = (thermal & 0xFF).astype(np.uint8)
        raw[:192, :, 1] = (thermal >> 8).astype(np.uint8)
        # Shutter temp at flat[547]
        flat = raw[192:, :, :].view(np.uint16).reshape(4, 256)
        flat[2, 35] = 3000
        flat[2, 89] = 2900
        return raw

    def set_zoom(self, value):
        self.zoom_values_sent.append(int(value))

    def close(self):
        self._is_open = False


def _make_camera():
    backend = FakeVideoBackend()
    cam = T2Pro(backend=backend)
    cam._backend = backend
    return cam


class TestT2ProSessionInit:
    def test_session_creates_with_defaults(self):
        cam = _make_camera()
        session = T2ProSession(cam)
        assert not session.is_streaming
        assert session.frame_count == 0
        assert session.is_recording is False
        assert session.fps == 0.0

    def test_session_not_streaming_before_enter(self):
        cam = _make_camera()
        session = T2ProSession(cam)
        assert not session.is_streaming


class TestT2ProSessionContextManager:
    def test_context_manager_starts_and_stops_stream(self):
        cam = _make_camera()
        session = T2ProSession(cam)

        with session:
            assert session.is_streaming
        assert not session.is_streaming

    def test_context_manager_stops_recording_on_exit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cam = _make_camera()
            session = T2ProSession(cam, output_dir=tmpdir)

            with session:
                session.start_recording()
                assert session.is_recording
            # Recording should be stopped after exit
            assert not session.is_recording


class TestT2ProSessionReading:
    def test_read_frame_returns_array(self):
        cam = _make_camera()
        session = T2ProSession(cam)

        with session:
            frame = session.read_frame()
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (192, 256)

    def test_read_frame_increments_count(self):
        cam = _make_camera()
        session = T2ProSession(cam)

        with session:
            session.read_frame()
            session.read_frame()
            assert session.frame_count == 2

    def test_read_frame_raw_returns_raw(self):
        cam = _make_camera()
        session = T2ProSession(cam)

        with session:
            raw = session.read_frame_raw()
            assert isinstance(raw, np.ndarray)
            assert raw.shape == (196, 256, 2)

    def test_read_temperature_returns_result(self):
        cam = _make_camera()
        session = T2ProSession(cam)

        with session:
            result = session.read_temperature()
            assert result is not None
            assert hasattr(result, 'center_temp')
            assert hasattr(result, 'max_temp')
            assert hasattr(result, 'fpa_temp')


class TestT2ProSessionProcessing:
    def test_process_returns_display_image(self):
        cam = _make_camera()
        session = T2ProSession(cam)

        with session:
            frame = session.read_frame()
            display = session.process(frame)
            assert isinstance(display, np.ndarray)
            assert display.shape[0] > 0
            assert display.shape[1] > 0
            # 5x upscale default
            assert display.shape[0] == 192 * 5
            assert display.shape[1] == 256 * 5

    def test_process_with_custom_scale(self):
        cam = _make_camera()
        session = T2ProSession(cam, scale=3)

        with session:
            frame = session.read_frame()
            display = session.process(frame)
            assert display.shape[0] == 192 * 3
            assert display.shape[1] == 256 * 3

    def test_process_without_denoise(self):
        cam = _make_camera()
        session = T2ProSession(cam)

        with session:
            frame = session.read_frame()
            display = session.process(frame, use_denoise=False)
            assert isinstance(display, np.ndarray)


class TestT2ProSessionRecording:
    def test_start_recording_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cam = _make_camera()
            session = T2ProSession(cam, output_dir=tmpdir)

            with session:
                path = session.start_recording()
                assert os.path.isdir(path)
                assert os.path.isdir(os.path.join(path, "frames"))
                metadata = session.stop_recording()
                assert metadata["frame_count"] == 0

    def test_recording_raises_if_already_recording(self):
        cam = _make_camera()
        session = T2ProSession(cam)

        with session:
            session.start_recording()
            with pytest.raises(RuntimeError, match="Already recording"):
                session.start_recording()
            session.stop_recording()

    def test_stop_recording_raises_if_not_recording(self):
        cam = _make_camera()
        session = T2ProSession(cam)

        with session:
            with pytest.raises(RuntimeError, match="Not recording"):
                session.stop_recording()

    def test_recording_properties(self):
        cam = _make_camera()
        session = T2ProSession(cam)

        with session:
            assert not session.is_recording
            session.start_recording()
            assert session.is_recording
            session.stop_recording()
            assert not session.is_recording


class TestT2ProSessionControls:
    def test_trigger_nuc(self):
        cam = _make_camera()
        session = T2ProSession(cam)

        with session:
            session.trigger_nuc()
            # Should have sent a shutter command
            assert len(cam._backend.zoom_values_sent) > 0

    def test_set_denoise(self):
        cam = _make_camera()
        session = T2ProSession(cam)
        session.set_denoise(False)
        assert session._use_denoise is False
        session.set_denoise(True)
        assert session._use_denoise is True

    def test_set_palette(self):
        cam = _make_camera()
        session = T2ProSession(cam)
        from infiray_t2pro.palettes import Palette
        session.set_palette(Palette.WHITE_HOT)
        assert session.palette == Palette.WHITE_HOT

    def test_set_scale(self):
        cam = _make_camera()
        session = T2ProSession(cam)
        session.set_scale(3)
        assert session.scale == 3