"""Tests for snapshot mode — save complete thermal snapshot to disk."""

import json
import os
import numpy as np
import pytest
import tempfile

from infiray_t2pro.camera import T2Pro, VideoBackend
from infiray_t2pro.thermometry import TemperatureResult
from infiray_t2pro.palettes import Palette
from infiray_t2pro.snapshot import take_snapshot, SnapshotResult


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
        return raw

    def set_zoom(self, value):
        self.zoom_values_sent.append(int(value))

    def close(self):
        self.close_count += 1
        self._is_open = False


def _make_camera():
    backend = FakeVideoBackend()
    cam = T2Pro(backend=backend)
    cam._backend = backend
    return cam, backend


def _fake_temps():
    """Create a realistic TemperatureResult for testing."""
    return TemperatureResult(
        center_temp=25.3,
        max_temp=38.7,
        max_x=128,
        max_y=96,
        min_temp=18.2,
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


class TestSnapshotResult:
    def test_str_representation(self):
        result = SnapshotResult(
            directory="/tmp/snapshots/2026-07-07_15-30-00",
            png_path="/tmp/snapshots/2026-07-07_15-30-00/thermal_2026-07-07_15-30-00.png",
            npy_path="/tmp/snapshots/2026-07-07_15-30-00/raw_2026-07-07_15-30-00.npy",
            json_path="/tmp/snapshots/2026-07-07_15-30-00/metadata_2026-07-07_15-30-00.json",
            center_temp=25.3,
            max_temp=38.7,
            min_temp=18.2,
            avg_temp=24.1,
            fpa_temp=32.5,
            timestamp="2026-07-07_15-30-00",
        )
        s = str(result)
        assert "25.3" in s
        assert "38.7" in s
        assert "18.2" in s
        assert "32.5" in s


class TestTakeSnapshot:
    def test_creates_snapshot_directory(self):
        """take_snapshot creates a timestamped directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cam, _ = _make_camera()
            cam.start_stream(warmup=1)

            result = take_snapshot(cam, temp_result=_fake_temps(), output_dir=tmpdir)
            cam.stop_stream()

            assert os.path.isdir(result.directory)
            assert result.directory.startswith(tmpdir)

    def test_saves_png_file(self):
        """take_snapshot saves a PNG with temperature overlay."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cam, _ = _make_camera()
            cam.start_stream(warmup=1)

            result = take_snapshot(cam, temp_result=_fake_temps(), output_dir=tmpdir)
            cam.stop_stream()

            assert os.path.exists(result.png_path)
            assert result.png_path.endswith(".png")
            assert os.path.getsize(result.png_path) > 0

    def test_saves_npy_file(self):
        """take_snapshot saves frame data as .npy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cam, _ = _make_camera()
            cam.start_stream(warmup=1)

            result = take_snapshot(cam, temp_result=_fake_temps(), output_dir=tmpdir)
            cam.stop_stream()

            assert os.path.exists(result.npy_path)
            assert result.npy_path.endswith(".npy")
            data = np.load(result.npy_path)
            assert data.size > 0  # Has actual content

    def test_saves_json_metadata(self):
        """take_snapshot saves complete temperature metadata as JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cam, _ = _make_camera()
            cam.start_stream(warmup=1)

            result = take_snapshot(cam, temp_result=_fake_temps(), output_dir=tmpdir)
            cam.stop_stream()

            assert os.path.exists(result.json_path)
            with open(result.json_path) as f:
                meta = json.load(f)

            # Required sections
            assert "timestamp" in meta
            assert "palette" in meta
            assert "temperatures" in meta
            assert "parameters" in meta
            assert "files" in meta

            # Temperature data
            temps = meta["temperatures"]
            assert temps["center_c"] == 25.3
            assert temps["max_c"] == 38.7
            assert temps["min_c"] == 18.2
            assert temps["avg_c"] == 24.1
            assert temps["fpa_c"] == 32.5

            # Parameters
            params = meta["parameters"]
            assert params["emissivity"] == 0.98
            assert params["distance_m"] == 1

            # Files
            assert "png" in meta["files"]
            assert "npy" in meta["files"]
            assert "json" in meta["files"]

    def test_snapshot_result_has_temperatures(self):
        """SnapshotResult contains temperature readings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cam, _ = _make_camera()
            cam.start_stream(warmup=1)

            result = take_snapshot(cam, temp_result=_fake_temps(), output_dir=tmpdir)
            cam.stop_stream()

            assert result.center_temp == 25.3
            assert result.max_temp == 38.7
            assert result.min_temp == 18.2
            assert result.avg_temp == 24.1
            assert result.fpa_temp == 32.5

    def test_custom_palette(self):
        """take_snapshot works with different palettes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cam, _ = _make_camera()
            cam.start_stream(warmup=1)

            result = take_snapshot(
                cam, temp_result=_fake_temps(),
                output_dir=tmpdir, palette=Palette.WHITE_HOT,
            )
            cam.stop_stream()

            assert os.path.exists(result.png_path)
            with open(result.json_path) as f:
                meta = json.load(f)
            assert meta["palette"] == "White Hot"

    def test_directory_is_timestamped(self):
        """Snapshot directory name contains the timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cam, _ = _make_camera()
            cam.start_stream(warmup=1)

            result = take_snapshot(cam, temp_result=_fake_temps(), output_dir=tmpdir)
            cam.stop_stream()

            dir_name = os.path.basename(result.directory)
            assert result.timestamp in dir_name

    def test_requires_tlib_or_temp_result(self):
        """take_snapshot raises ValueError if neither tlib nor temp_result provided."""
        cam, _ = _make_camera()
        cam.start_stream(warmup=1)
        with pytest.raises(ValueError, match="Provide either tlib or temp_result"):
            take_snapshot(cam, output_dir="/tmp/test_snap")
        cam.stop_stream()