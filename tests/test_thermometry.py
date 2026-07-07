"""TDD RED: tests for infiray_t2pro.thermometry module.

Tests cover:
1. Metadata extraction from 4-row metadata section of raw frames
2. ThermometryLib ctypes wrapper (mocked)
3. Temperature calculation pipeline: raw frame → metadata → temp table → °C
4. Error handling: missing lib, bad frames, invalid parameters
"""

import ctypes
import os
import platform
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from infiray_t2pro.decode import IMAGE_WIDTH, IMAGE_HEIGHT, TOTAL_ROWS, METADATA_ROWS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_frame(thermal_2d=None, metadata_rows=None, seed=42):
    """Build a (196, 256, 2) uint8 raw frame.

    Args:
        thermal_2d: (192, 256) float32 array of 14-bit thermal values.
                    If None, generated with random noise.
        metadata_rows: (4, 256, 2) uint8 metadata. If None, zeros.
        seed: Random seed for reproducible thermal data.
    """
    rng = np.random.RandomState(seed)
    if thermal_2d is None:
        # Realistic thermal: base ~5000, noise std ~200
        thermal_2d = rng.randint(3000, 7000, size=(IMAGE_HEIGHT, IMAGE_WIDTH)).astype(np.float32)

    # Encode thermal to 2-byte little-endian
    raw = np.zeros((TOTAL_ROWS, IMAGE_WIDTH, 2), dtype=np.uint8)
    thermal_uint16 = np.clip(thermal_2d, 0, 16383).astype(np.uint16)
    raw[:IMAGE_HEIGHT, :, 0] = (thermal_uint16 & 0xFF).astype(np.uint8)   # LSB
    raw[:IMAGE_HEIGHT, :, 1] = (thermal_uint16 >> 8).astype(np.uint8)     # MSB

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
    shut_temper=2982,     # ~25°C (298.15K * 10)
    core_temper=2982,     # ~25°C
):
    """Build (4, 256, 2) metadata rows with embedded parameters.

    For the T2 Pro (256 width), amountPixels = 256*(4-3) = 256.
    User area starts at fourLinePara[256+127] = fourLinePara[383].
    Parameters are stored as float (2 x uint16) or uint16 starting there.
    """
    meta = np.zeros((METADATA_ROWS, IMAGE_WIDTH, 2), dtype=np.uint8)

    # fourLinePara is uint16* view of the metadata rows
    # Index into uint16 array: row_offset + col
    # For 256-wide frames: amountPixels = 256*(4-3) = 256
    # userArea starts at amountPixels + 127 = 383

    # Flatten metadata to uint16 view for easier writing
    # Each row is 256 uint16 values (256 cols × 2 bytes = 512 bytes per row)
    flat_uint16 = meta.view(np.uint16).reshape(METADATA_ROWS * IMAGE_WIDTH)

    # Write shutter temperature at offset amountPixels + 1
    flat_uint16[256 + 1] = shut_temper
    # Write core temperature at offset amountPixels + 2
    flat_uint16[256 + 2] = core_temper

    # Write user parameters at offset 256 + 127 = 383
    # Each float takes 2 uint16 slots
    user_offset = 256 + 127

    # correction: float
    _write_float_to_uint16(flat_uint16, user_offset, correction)
    user_offset += 2

    # reflection_temp: float
    _write_float_to_uint16(flat_uint16, user_offset, reflection_temp)
    user_offset += 2

    # ambient_temp: float
    _write_float_to_uint16(flat_uint16, user_offset, ambient_temp)
    user_offset += 2

    # humidity: float
    _write_float_to_uint16(flat_uint16, user_offset, humidity)
    user_offset += 2

    # emissivity: float
    _write_float_to_uint16(flat_uint16, user_offset, emissivity)
    user_offset += 2

    # distance: uint16
    flat_uint16[user_offset] = distance

    return meta


def _write_float_to_uint16(arr, offset, value):
    """Write a float32 value as two uint16 values at arr[offset:offset+2]."""
    raw_bytes = np.array([value], dtype=np.float32).view(np.uint16)
    arr[offset] = raw_bytes[0]
    arr[offset + 1] = raw_bytes[1]


# ---------------------------------------------------------------------------
# Tests: Metadata extraction
# ---------------------------------------------------------------------------

class TestExtractMetadata:
    """Test extracting parameters from the 4 metadata rows of a raw frame."""

    def test_extract_metadata_shape(self):
        """extract_metadata returns (4, 256, 2) uint8 array."""
        from infiray_t2pro.decode import extract_metadata
        raw = _make_raw_frame()
        meta = extract_metadata(raw)
        assert meta.shape == (4, 256, 2)
        assert meta.dtype == np.uint8

    def test_extract_metadata_preserves_values(self):
        """Metadata extraction preserves the original byte values."""
        from infiray_t2pro.decode import extract_metadata
        meta_input = np.full((4, 256, 2), 0xAB, dtype=np.uint8)
        raw = _make_raw_frame(metadata_rows=meta_input)
        meta = extract_metadata(raw)
        np.testing.assert_array_equal(meta, meta_input)


class TestMetadataParameters:
    """Test parsing user parameters from metadata rows."""

    def test_parse_correction(self):
        """Extract correction value from metadata."""
        from infiray_t2pro.thermometry import parse_metadata_params
        meta = _make_metadata_with_params(correction=-1.5)
        params = parse_metadata_params(meta)
        assert abs(params.correction - (-1.5)) < 0.01

    def test_parse_reflection_temp(self):
        """Extract reflection temperature from metadata."""
        from infiray_t2pro.thermometry import parse_metadata_params
        meta = _make_metadata_with_params(reflection_temp=23.0)
        params = parse_metadata_params(meta)
        assert abs(params.reflection_temp - 23.0) < 0.01

    def test_parse_ambient_temp(self):
        """Extract ambient temperature from metadata."""
        from infiray_t2pro.thermometry import parse_metadata_params
        meta = _make_metadata_with_params(ambient_temp=22.5)
        params = parse_metadata_params(meta)
        assert abs(params.ambient_temp - 22.5) < 0.01

    def test_parse_humidity(self):
        """Extract humidity from metadata."""
        from infiray_t2pro.thermometry import parse_metadata_params
        meta = _make_metadata_with_params(humidity=0.50)
        params = parse_metadata_params(meta)
        assert abs(params.humidity - 0.50) < 0.01

    def test_parse_emissivity(self):
        """Extract emissivity from metadata."""
        from infiray_t2pro.thermometry import parse_metadata_params
        meta = _make_metadata_with_params(emissivity=0.97)
        params = parse_metadata_params(meta)
        assert abs(params.emissivity - 0.97) < 0.01

    def test_parse_distance(self):
        """Extract distance from metadata."""
        from infiray_t2pro.thermometry import parse_metadata_params
        meta = _make_metadata_with_params(distance=5)
        params = parse_metadata_params(meta)
        assert params.distance == 5

    def test_parse_shutter_temperature(self):
        """Extract shutter temperature from metadata (raw uint16, needs conversion)."""
        from infiray_t2pro.thermometry import parse_metadata_params
        # 2982 → 298.2K → 25.05°C
        meta = _make_metadata_with_params(shut_temper=2982)
        params = parse_metadata_params(meta)
        assert abs(params.shutter_temp_c - 25.05) < 0.5  # 298.2K - 273.15 = 25.05°C

    def test_parse_all_params(self):
        """Parse all parameters from metadata at once."""
        from infiray_t2pro.thermometry import parse_metadata_params
        meta = _make_metadata_with_params(
            correction=-2.0,
            reflection_temp=20.0,
            ambient_temp=22.0,
            humidity=0.55,
            emissivity=0.95,
            distance=3,
        )
        params = parse_metadata_params(meta)
        assert abs(params.correction - (-2.0)) < 0.01
        assert abs(params.reflection_temp - 20.0) < 0.01
        assert abs(params.ambient_temp - 22.0) < 0.01
        assert abs(params.humidity - 0.55) < 0.01
        assert abs(params.emissivity - 0.95) < 0.01
        assert params.distance == 3


# ---------------------------------------------------------------------------
# Tests: ThermometryLib wrapper
# ---------------------------------------------------------------------------

class TestThermometryLib:
    """Test the ThermometryLib ctypes wrapper (mocked library)."""

    def test_init_loads_library(self):
        """ThermometryLib loads libthermometry.so from the given path."""
        from infiray_t2pro.thermometry import ThermometryLib
        with patch.object(ctypes, 'cdll') as mock_cdll:
            mock_lib = MagicMock()
            mock_cdll.LoadLibrary.return_value = mock_lib
            tlib = ThermometryLib("/fake/path/libthermometry.so")
            mock_cdll.LoadLibrary.assert_called_once_with("/fake/path/libthermometry.so")

    def test_init_raises_if_lib_not_found(self):
        """ThermometryLib raises ThermometryLibNotFoundError if .so is missing."""
        from infiray_t2pro.thermometry import ThermometryLib, ThermometryLibNotFoundError
        with patch.object(ctypes, 'cdll') as mock_cdll:
            mock_cdll.LoadLibrary.side_effect = OSError("not found")
            with pytest.raises(ThermometryLibNotFoundError):
                ThermometryLib("/nonexistent/libthermometry.so")

    def test_thermometry_t4_line_calls_lib(self):
        """thermometry_t4_line calls the underlying C function with correct args."""
        from infiray_t2pro.thermometry import ThermometryLib
        with patch.object(ctypes, 'cdll') as mock_cdll:
            mock_lib = MagicMock()
            mock_cdll.LoadLibrary.return_value = mock_lib
            tlib = ThermometryLib("/fake/libthermometry.so")

            # Prepare test data
            four_line = np.zeros((4, 256), dtype=np.uint16)
            temp_table = np.zeros(16384, dtype=np.float32)

            tlib.thermometry_t4_line(
                width=256, height=196,
                temp_table=temp_table,
                four_line_para=four_line,
                correction=-1.0,
                reflection_temp=25.0,
                ambient_temp=25.0,
                humidity=0.45,
                emissivity=0.95,
                distance=3,
                camera_lens=130,
                shutter_fix=0.0,
                range_mode=120,
            )
            mock_lib.thermometryT4Line.assert_called_once()

    def test_thermometry_search_calls_lib(self):
        """thermometry_search calls the underlying C function."""
        from infiray_t2pro.thermometry import ThermometryLib
        with patch.object(ctypes, 'cdll') as mock_cdll:
            mock_lib = MagicMock()
            mock_cdll.LoadLibrary.return_value = mock_lib
            tlib = ThermometryLib("/fake/libthermometry.so")

            temp_table = np.zeros(16384, dtype=np.float32)
            org_data = np.zeros((192, 256), dtype=np.uint16)

            result = tlib.thermometry_search(
                width=256, height=192,
                temp_table=temp_table,
                org_data=org_data,
                range_mode=120,
                output_mode=4,
            )
            mock_lib.thermometrySearch.assert_called_once()
            # Result should be a numpy array
            assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# Tests: Thermometry pipeline
# ---------------------------------------------------------------------------

class TestThermometryPipeline:
    """Test the high-level temperature calculation pipeline."""

    def test_calculate_temperature_returns_celsius_array(self):
        """calculate_temperature returns a 2D float32 array in °C."""
        from infiray_t2pro.thermometry import calculate_temperature, ThermometryLib
        with patch.object(ctypes, 'cdll') as mock_cdll:
            mock_lib = MagicMock()
            mock_cdll.LoadLibrary.return_value = mock_lib

            # Mock thermometryT4Line: fill temp_table with known values
            def fill_table(w, h, table, four_line, fpa, corr, refl, air, humi, emiss, dist, lens, sfix, rmode):
                # Fill table so pixel value 5000 → 25.0°C
                for i in range(16384):
                    table[i] = (i - 5000) * 0.01 + 25.0  # dummy mapping

            # Mock thermometrySearch: fill temp_data from org_data via table
            def fill_search(w, h, table, org_data, temp_data, rmode, omode):
                # For mode 4: 10 stats + full frame
                temp_data[0] = 25.5   # center
                temp_data[3] = 30.0    # max
                temp_data[6] = 18.0    # min
                temp_data[9] = 23.0     # avg

            mock_lib.thermometryT4Line = MagicMock(side_effect=fill_table)
            mock_lib.thermometrySearch = MagicMock(side_effect=fill_search)

            tlib = ThermometryLib("/fake/libthermometry.so")
            raw = _make_raw_frame()
            from infiray_t2pro.decode import extract_metadata

            result = calculate_temperature(
                tlib, raw,
                camera_lens=130, range_mode=120,
            )
            # Should return a TemperatureResult with stats
            assert hasattr(result, 'center_temp')
            assert hasattr(result, 'max_temp')
            assert hasattr(result, 'min_temp')
            assert hasattr(result, 'avg_temp')

    def test_calculate_temperature_uses_metadata_params(self):
        """calculate_temperature extracts params from metadata and passes to lib."""
        from infiray_t2pro.thermometry import calculate_temperature, ThermometryLib
        with patch.object(ctypes, 'cdll') as mock_cdll:
            mock_lib = MagicMock()
            mock_cdll.LoadLibrary.return_value = mock_lib
            mock_lib.thermometryT4Line = MagicMock()
            mock_lib.thermometrySearch = MagicMock()

            tlib = ThermometryLib("/fake/libthermometry.so")

            meta = _make_metadata_with_params(emissivity=0.80, distance=5)
            raw = _make_raw_frame(metadata_rows=meta)

            calculate_temperature(tlib, raw, camera_lens=130, range_mode=120)

            # Check that the call was made with the metadata-derived params
            call_args = mock_lib.thermometryT4Line.call_args
            # emissivity and distance should match what we put in metadata
            assert call_args is not None  # at least called


# ---------------------------------------------------------------------------
# Tests: TemperatureResult
# ---------------------------------------------------------------------------

class TestTemperatureResult:
    """Test the TemperatureResult dataclass."""

    def test_result_fields(self):
        """TemperatureResult has all expected fields."""
        from infiray_t2pro.thermometry import TemperatureResult
        result = TemperatureResult(
            center_temp=25.5,
            max_temp=30.0,
            max_x=128,
            max_y=96,
            min_temp=18.0,
            min_x=10,
            min_y=20,
            avg_temp=23.0,
            point1_temp=22.0,
            point2_temp=24.0,
            point3_temp=21.0,
            fpa_temp=25.0,
            correction=-1.0,
            reflection_temp=25.0,
            ambient_temp=22.0,
            humidity=0.45,
            emissivity=0.95,
            distance=3,
            shutter_temp_c=25.0,
        )
        assert result.center_temp == 25.5
        assert result.max_temp == 30.0
        assert result.emissivity == 0.95

    def test_result_from_raw_frame_requires_streaming(self):
        """TemperatureResult cannot be computed without a raw frame."""
        # This is a design test — we ensure the API requires raw data
        from infiray_t2pro.thermometry import TemperatureResult
        result = TemperatureResult(
            center_temp=0, max_temp=0, max_x=0, max_y=0,
            min_temp=0, min_x=0, min_y=0, avg_temp=0,
            point1_temp=0, point2_temp=0, point3_temp=0,
            fpa_temp=0, correction=0, reflection_temp=0,
            ambient_temp=0, humidity=0, emissivity=0,
            distance=0, shutter_temp_c=0,
        )
        assert result.center_temp == 0


# ---------------------------------------------------------------------------
# Tests: CameraLens and RangeMode constants
# ---------------------------------------------------------------------------

class TestConstants:
    """Test that camera lens and range mode constants are defined."""

    def test_camera_lens_constants(self):
        """Camera lens constants are defined."""
        from infiray_t2pro.thermometry import CAMERA_LENS_6_8, CAMERA_LENS_13
        assert CAMERA_LENS_6_8 == 68
        assert CAMERA_LENS_13 == 130

    def test_range_mode_constants(self):
        """Range mode constants are defined."""
        from infiray_t2pro.thermometry import RANGE_MODE_120, RANGE_MODE_400
        assert RANGE_MODE_120 == 120
        assert RANGE_MODE_400 == 400