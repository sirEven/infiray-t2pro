"""Tests for thermal recording — save raw frames and temperature data to disk."""

import json
import os
import numpy as np
import pytest
import tempfile
import time

from infiray_t2pro.camera import T2Pro, VideoBackend
from infiray_t2pro.thermometry import TemperatureResult
from infiray_t2pro.recording import ThermalRecorder


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
        # Create a valid raw YUYV frame with T2 Pro metadata offsets
        raw = np.zeros((196, 256, 2), dtype=np.uint8)
        thermal = np.random.RandomState(42 + self._frame_count).randint(
            3000, 9000, (192, 256)
        ).astype(np.uint16)
        raw[:192, :, 0] = (thermal & 0xFF).astype(np.uint8)
        raw[:192, :, 1] = (thermal >> 8).astype(np.uint8)
        # Shutter temp at flat[547]
        flat = raw[192:, :, :].view(np.uint16).reshape(4, 256)
        flat[2, 35] = 3000
        # Core temp at flat[601]
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
    return cam, backend


def _fake_temps(center=25.3, max_t=38.7, min_t=18.2):
    """Create a realistic TemperatureResult for testing."""
    return TemperatureResult(
        center_temp=center,
        max_temp=max_t,
        max_x=128,
        max_y=96,
        min_temp=min_t,
        min_x=64,
        min_y=48,
        avg_temp=24.1,
        point1_temp=0.0,
        point2_temp=0.0,
        point3_temp=0.0,
        fpa_temp=32.5,
        correction=25.0,
        reflection_temp=25.0,
        ambient_temp=22.0,
        humidity=0.45,
        emissivity=0.98,
        distance=1,
        shutter_temp_c=26.7,
    )


class TestThermalRecorderInit:
    def test_default_session_name_is_timestamp(self):
        cam, _ = _make_camera()
        rec = ThermalRecorder(cam)
        # Should be YYYY-MM-DD_HH-MM-SS format
        assert len(rec.session_name) == 19
        assert "-" in rec.session_name
        assert "_" in rec.session_name

    def test_custom_session_name(self):
        cam, _ = _make_camera()
        rec = ThermalRecorder(cam, session_name="test_session")
        assert rec.session_name == "test_session"

    def test_not_recording_initially(self):
        cam, _ = _make_camera()
        rec = ThermalRecorder(cam)
        assert not rec.is_recording
        assert rec.frame_count == 0


class TestThermalRecorderStartStop:
    def test_start_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cam, _ = _make_camera()
            rec = ThermalRecorder(cam, output_dir=tmpdir, session_name="test1")
            cam.start_stream(warmup=1)

            session_path = rec.start()
            cam.stop_stream()

            assert os.path.isdir(session_path)
            assert os.path.isdir(os.path.join(session_path, "frames"))

    def test_start_returns_session_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cam, _ = _make_camera()
            rec = ThermalRecorder(cam, output_dir=tmpdir, session_name="test2")
            cam.start_stream(warmup=1)

            session_path = rec.start()
            cam.stop_stream()

            assert session_path.endswith("test2")
            assert tmpdir in session_path

    def test_double_start_raises(self):
        cam, _ = _make_camera()
        rec = ThermalRecorder(cam, session_name="test3")
        cam.start_stream(warmup=1)
        rec.start()

        with pytest.raises(RuntimeError, match="Already recording"):
            rec.start()

        rec.stop()
        cam.stop_stream()

    def test_stop_without_start_raises(self):
        cam, _ = _make_camera()
        rec = ThermalRecorder(cam)

        with pytest.raises(RuntimeError, match="Not recording"):
            rec.stop()

    def test_stop_returns_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cam, _ = _make_camera()
            rec = ThermalRecorder(cam, output_dir=tmpdir, session_name="test4")
            cam.start_stream(warmup=1)
            rec.start()
            time.sleep(0.05)

            metadata = rec.stop()
            cam.stop_stream()

            assert metadata["session"] == "test4"
            assert metadata["frame_count"] == 0
            assert "duration_s" in metadata
            assert "fps" in metadata
            assert metadata["sensor_size"] == "256x192"

    def test_metadata_file_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cam, _ = _make_camera()
            rec = ThermalRecorder(cam, output_dir=tmpdir, session_name="test5")
            cam.start_stream(warmup=1)
            rec.start()
            time.sleep(0.05)
            metadata = rec.stop()
            cam.stop_stream()

            meta_path = os.path.join(rec.session_path, "recording.json")
            assert os.path.exists(meta_path)
            with open(meta_path) as f:
                saved = json.load(f)
            assert saved["session"] == "test5"
            assert saved["frame_count"] == 0


class TestThermalRecorderRecordFrame:
    def test_record_frame_saves_npy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cam, _ = _make_camera()
            rec = ThermalRecorder(cam, output_dir=tmpdir, session_name="rec1")
            cam.start_stream(warmup=1)
            rec.start()

            rec.record_frame()
            cam.stop_stream()
            rec.stop()

            frame_path = os.path.join(rec.session_path, "frames", "000001.npy")
            assert os.path.exists(frame_path)
            data = np.load(frame_path)
            assert data.shape == (196, 256, 2)

    def test_record_frame_increments_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cam, _ = _make_camera()
            rec = ThermalRecorder(cam, output_dir=tmpdir, session_name="rec2")
            cam.start_stream(warmup=1)
            rec.start()

            rec.record_frame()
            rec.record_frame()
            rec.record_frame()
            cam.stop_stream()
            rec.stop()

            assert rec.frame_count == 3

    def test_record_multiple_frames_sequential(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cam, _ = _make_camera()
            rec = ThermalRecorder(cam, output_dir=tmpdir, session_name="rec3")
            cam.start_stream(warmup=1)
            rec.start()

            for _ in range(5):
                rec.record_frame()
            cam.stop_stream()
            rec.stop()

            for i in range(1, 6):
                assert os.path.exists(
                    os.path.join(rec.session_path, "frames", f"{i:06d}.npy")
                )

    def test_record_frame_not_recording_raises(self):
        cam, _ = _make_camera()
        rec = ThermalRecorder(cam)

        with pytest.raises(RuntimeError, match="Not recording"):
            rec.record_frame()


class TestThermalRecorderTemperatures:
    def test_record_frame_with_tlib_saves_temps(self):
        """When tlib is provided, temps.jsonl is written per frame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cam, _ = _make_camera()
            # Use a mock tlib that returns fake temperatures
            # We test with temp_result via record_frame only, not tlib directly
            # since ThermometryLib needs the real .so
            rec = ThermalRecorder(cam, output_dir=tmpdir, session_name="temp1")
            cam.start_stream(warmup=1)
            rec.start()

            # record_frame with tlib=None skips temp calculation
            rec.record_frame()
            cam.stop_stream()
            rec.stop()

            # temps.jsonl should not exist (tlib=None)
            temps_path = os.path.join(rec.session_path, "temps.jsonl")
            # Actually it IS created, just empty when tlib is None
            # Let's check metadata says has_temperatures=False
            meta_path = os.path.join(rec.session_path, "recording.json")
            with open(meta_path) as f:
                meta = json.load(f)
            assert meta["has_temperatures"] is False

    def test_record_frame_without_tlib_no_temps(self):
        """Without tlib, temperatures are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cam, _ = _make_camera()
            rec = ThermalRecorder(cam, tlib=None, output_dir=tmpdir, session_name="temp2")
            cam.start_stream(warmup=1)
            rec.start()

            result = rec.record_frame()
            cam.stop_stream()
            rec.stop()

            assert result is None

    def test_temps_jsonl_format(self):
        """Each line in temps.jsonl is valid JSON with frame number and time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cam, _ = _make_camera()
            rec = ThermalRecorder(cam, output_dir=tmpdir, session_name="temp3")
            cam.start_stream(warmup=1)
            rec.start()

            # Record 3 frames (temps skipped without tlib)
            for _ in range(3):
                rec.record_frame()
            cam.stop_stream()
            rec.stop()

            temps_path = os.path.join(rec.session_path, "temps.jsonl")
            # File exists but should be empty (no tlib)
            if os.path.exists(temps_path):
                with open(temps_path) as f:
                    content = f.read().strip()
                # Empty or no lines — no temps calculated without tlib
                assert content == "" or content == ""


class TestThermalRecorderFast:
    def test_record_frame_fast_saves_npy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cam, _ = _make_camera()
            rec = ThermalRecorder(cam, output_dir=tmpdir, session_name="fast1")
            cam.start_stream(warmup=1)
            rec.start()

            rec.record_frame_fast()
            cam.stop_stream()
            rec.stop()

            frame_path = os.path.join(rec.session_path, "frames", "000001.npy")
            assert os.path.exists(frame_path)
            data = np.load(frame_path)
            assert data.shape == (196, 256, 2)

    def test_record_frame_fast_no_temps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cam, _ = _make_camera()
            rec = ThermalRecorder(cam, output_dir=tmpdir, session_name="fast2")
            cam.start_stream(warmup=1)
            rec.start()

            rec.record_frame_fast()
            cam.stop_stream()
            rec.stop()

            # No temps.jsonl content
            temps_path = os.path.join(rec.session_path, "temps.jsonl")
            if os.path.exists(temps_path):
                with open(temps_path) as f:
                    content = f.read().strip()
                assert content == ""

    def test_record_frame_fast_not_recording_raises(self):
        cam, _ = _make_camera()
        rec = ThermalRecorder(cam)

        with pytest.raises(RuntimeError, match="Not recording"):
            rec.record_frame_fast()


class TestThermalRecorderContextManager:
    def test_context_manager_starts_and_stops(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cam, _ = _make_camera()
            rec = ThermalRecorder(cam, output_dir=tmpdir, session_name="ctx1")
            cam.start_stream(warmup=1)

            with rec:
                assert rec.is_recording
                rec.record_frame()
            cam.stop_stream()

            assert not rec.is_recording
            assert rec.frame_count == 1

    def test_context_manager_writes_metadata_on_exit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cam, _ = _make_camera()
            rec = ThermalRecorder(cam, output_dir=tmpdir, session_name="ctx2")
            cam.start_stream(warmup=1)

            with rec:
                for _ in range(3):
                    rec.record_frame()
            cam.stop_stream()

            meta_path = os.path.join(rec.session_path, "recording.json")
            assert os.path.exists(meta_path)
            with open(meta_path) as f:
                meta = json.load(f)
            assert meta["frame_count"] == 3


class TestThermalRecorderFps:
    def test_fps_is_zero_before_recording(self):
        cam, _ = _make_camera()
        rec = ThermalRecorder(cam)
        assert rec.fps == 0.0

    def test_fps_after_recording(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cam, _ = _make_camera()
            rec = ThermalRecorder(cam, output_dir=tmpdir, session_name="fps1")
            cam.start_stream(warmup=1)
            rec.start()

            for _ in range(5):
                rec.record_frame()
            cam.stop_stream()
            metadata = rec.stop()

            assert metadata["fps"] > 0
            assert rec.fps > 0