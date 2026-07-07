"""Tests for T2Pro.calculate_temperature() integration."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from infiray_t2pro.decode import IMAGE_WIDTH, IMAGE_HEIGHT, TOTAL_ROWS, METADATA_ROWS


def _make_raw_frame(thermal_2d=None, metadata_rows=None, seed=42):
    """Build a (196, 256, 2) uint8 raw frame."""
    rng = np.random.RandomState(seed)
    if thermal_2d is None:
        thermal_2d = rng.randint(3000, 7000, size=(IMAGE_HEIGHT, IMAGE_WIDTH)).astype(np.float32)

    raw = np.zeros((TOTAL_ROWS, IMAGE_WIDTH, 2), dtype=np.uint8)
    thermal_uint16 = np.clip(thermal_2d, 0, 16383).astype(np.uint16)
    raw[:IMAGE_HEIGHT, :, 0] = (thermal_uint16 & 0xFF).astype(np.uint8)
    raw[:IMAGE_HEIGHT, :, 1] = (thermal_uint16 >> 8).astype(np.uint8)

    if metadata_rows is not None:
        raw[IMAGE_HEIGHT:, :, :] = metadata_rows

    return raw


def _make_metadata_with_params(
    correction=0.0,
    reflection_temp=25.0,
    ambient_temp=25.0,
    humidity=0.45,
    emissivity=0.95,
    distance=3,
    shut_temper=2982,
    core_temper=2982,
):
    """Build (4, 256, 2) metadata rows with embedded parameters."""
    meta = np.zeros((METADATA_ROWS, IMAGE_WIDTH, 2), dtype=np.uint8)
    flat_uint16 = meta.view(np.uint16).reshape(METADATA_ROWS * IMAGE_WIDTH)

    flat_uint16[256 + 1] = shut_temper
    flat_uint16[256 + 2] = core_temper

    offset = 256 + 127
    for val in [correction, reflection_temp, ambient_temp, humidity, emissivity]:
        raw_bytes = np.array([val], dtype=np.float32).view(np.uint16)
        flat_uint16[offset] = raw_bytes[0]
        flat_uint16[offset + 1] = raw_bytes[1]
        offset += 2
    flat_uint16[offset] = distance

    return meta


class TestCalculateTemperatureOnT2Pro:
    """Test T2Pro.calculate_temperature() method."""

    def test_calculate_temperature_returns_result(self):
        """calculate_temperature returns a TemperatureResult."""
        from infiray_t2pro.camera import T2Pro, VideoBackend
        from infiray_t2pro.thermometry import TemperatureResult, ThermometryLib

        # Fake backend that returns realistic frames
        meta = _make_metadata_with_params(emissivity=0.95, distance=3)
        raw = _make_raw_frame(metadata_rows=meta)

        fake = MagicMock(spec=VideoBackend)
        fake.read_raw.return_value = raw

        # Mock ThermometryLib
        with patch('infiray_t2pro.camera.ThermometryLib') as MockTLib:
            mock_lib_instance = MagicMock()
            MockTLib.return_value = mock_lib_instance

            # Mock thermometry_t4_line to fill temp_table and return FPA temp
            def fill_table(**kwargs):
                table = kwargs['temp_table']
                for i in range(16384):
                    table[i] = (i - 5000) * 0.01 + 25.0
                return 25.5

            mock_lib_instance.thermometry_t4_line.side_effect = fill_table

            # Mock thermometry_search to return stats
            mock_lib_instance.thermometry_search.return_value = np.array(
                [25.5, 128, 96, 30.0, 10, 20, 18.0, 22.0, 24.0, 23.0],
                dtype=np.float32,
            )

            cam = T2Pro(backend=fake)
            cam.start_stream(warmup=0)

            result = cam.calculate_temperature()

            assert isinstance(result, TemperatureResult)
            assert result.center_temp == 25.5
            assert result.max_temp == 30.0
            assert result.min_temp == 18.0
            assert result.avg_temp == 23.0
            assert pytest.approx(result.emissivity, abs=0.01) == 0.95
            assert result.distance == 3

    def test_calculate_temperature_requires_streaming(self):
        """calculate_temperature raises if not streaming."""
        from infiray_t2pro.camera import T2Pro, VideoBackend, StreamClosedError

        fake = MagicMock(spec=VideoBackend)
        cam = T2Pro(backend=fake)

        with pytest.raises(StreamClosedError, match="not open"):
            cam.calculate_temperature()

    def test_calculate_temperature_with_explicit_lib(self):
        """calculate_temperature accepts an explicit ThermometryLib instance."""
        from infiray_t2pro.camera import T2Pro, VideoBackend
        from infiray_t2pro.thermometry import TemperatureResult

        meta = _make_metadata_with_params()
        raw = _make_raw_frame(metadata_rows=meta)

        fake = MagicMock(spec=VideoBackend)
        fake.read_raw.return_value = raw

        mock_tlib = MagicMock()
        mock_tlib.thermometry_t4_line.return_value = 25.0
        mock_tlib.thermometry_search.return_value = np.zeros(10, dtype=np.float32)

        cam = T2Pro(backend=fake)
        cam.start_stream(warmup=0)

        result = cam.calculate_temperature(tlib=mock_tlib)
        assert isinstance(result, TemperatureResult)
        mock_tlib.thermometry_t4_line.assert_called_once()
        mock_tlib.thermometry_search.assert_called_once()

    def test_calculate_temperature_with_range_mode(self):
        """calculate_temperature passes range_mode through."""
        from infiray_t2pro.camera import T2Pro, VideoBackend
        from infiray_t2pro.thermometry import RANGE_MODE_400

        meta = _make_metadata_with_params()
        raw = _make_raw_frame(metadata_rows=meta)

        fake = MagicMock(spec=VideoBackend)
        fake.read_raw.return_value = raw

        mock_tlib = MagicMock()
        mock_tlib.thermometry_t4_line.return_value = 25.0
        mock_tlib.thermometry_search.return_value = np.zeros(10, dtype=np.float32)

        cam = T2Pro(backend=fake)
        cam.start_stream(warmup=0)

        result = cam.calculate_temperature(tlib=mock_tlib, range_mode=RANGE_MODE_400)

        # Check range_mode was passed to thermometry_t4_line
        call_kwargs = mock_tlib.thermometry_t4_line.call_args
        assert call_kwargs.kwargs.get('range_mode', call_kwargs[1].get('range_mode')) == RANGE_MODE_400


class TestReadFrameRaw:
    """Test read_frame_raw() returns raw frame data for temperature calc."""

    def test_read_frame_raw_returns_raw_frame(self):
        """read_frame_raw returns the raw (196, 256, 2) uint8 frame."""
        from infiray_t2pro.camera import T2Pro, VideoBackend

        raw = _make_raw_frame()
        fake = MagicMock(spec=VideoBackend)
        fake.read_raw.return_value = raw

        cam = T2Pro(backend=fake)
        cam.start_stream(warmup=0)

        result = cam.read_frame_raw()
        assert result.shape == (TOTAL_ROWS, IMAGE_WIDTH, 2)
        assert result.dtype == np.uint8

    def test_read_frame_raw_requires_streaming(self):
        """read_frame_raw raises if not streaming."""
        from infiray_t2pro.camera import T2Pro, VideoBackend, StreamClosedError

        fake = MagicMock(spec=VideoBackend)
        cam = T2Pro(backend=fake)

        with pytest.raises(StreamClosedError):
            cam.read_frame_raw()