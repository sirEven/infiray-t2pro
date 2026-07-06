"""Tests for raw YUYV frame decoding.

The T2 Pro outputs 256×196 YUYV frames where:
- 192 rows are 16-bit thermal data (channel 1 = MSB, channel 0 = LSB)
- 4 rows are metadata
- Each pixel is 2 bytes, total frame = 100,352 bytes
"""

import numpy as np
import pytest
from infiray_t2pro.decode import decode_frame, extract_metadata, IMAGE_HEIGHT, TOTAL_ROWS, IMAGE_WIDTH


class TestConstants:
    def test_image_dimensions(self):
        assert IMAGE_WIDTH == 256
        assert IMAGE_HEIGHT == 192
        assert TOTAL_ROWS == 196


class TestDecodeFrame:
    def test_decodes_to_correct_shape(self):
        """Output should be 192×256 (image rows only, no metadata)."""
        raw = np.zeros((196, 256, 2), dtype=np.uint8)
        result = decode_frame(raw)
        assert result.shape == (192, 256)

    def test_decodes_to_float32(self):
        raw = np.zeros((196, 256, 2), dtype=np.uint8)
        result = decode_frame(raw)
        assert result.dtype == np.float32

    def test_zero_input_produces_zero_output(self):
        raw = np.zeros((196, 256, 2), dtype=np.uint8)
        result = decode_frame(raw)
        assert np.all(result == 0)

    def test_max_input_produces_max_output(self):
        """All 0xFF bytes should decode to 65535."""
        raw = np.full((196, 256, 2), 0xFF, dtype=np.uint8)
        result = decode_frame(raw)
        assert np.all(result == 65535)

    def test_specific_value_decodes_correctly(self):
        """Channel 1 = MSB, channel 0 = LSB: value = MSB*256 + LSB."""
        raw = np.zeros((196, 256, 2), dtype=np.uint8)
        # Set pixel [0, 0] to value 0x1234 = 4660
        # MSB = 0x12 = 18, LSB = 0x34 = 52
        raw[0, 0, 0] = 0x34  # LSB
        raw[0, 0, 1] = 0x12  # MSB
        result = decode_frame(raw)
        assert result[0, 0] == 0x1234

    def test_different_values_in_different_positions(self):
        """Each pixel should decode independently."""
        raw = np.zeros((196, 256, 2), dtype=np.uint8)
        raw[0, 0, 0] = 10; raw[0, 0, 1] = 20    # 20*256 + 10 = 5130
        raw[0, 1, 0] = 100; raw[0, 1, 1] = 200   # 200*256 + 100 = 51300
        raw[191, 255, 0] = 0; raw[191, 255, 1] = 255  # 255*256 = 65280
        result = decode_frame(raw)
        assert result[0, 0] == 5130
        assert result[0, 1] == 51300
        assert result[191, 255] == 65280

    def test_metadata_rows_excluded_from_output(self):
        """Rows 192-195 should not appear in decoded output."""
        raw = np.zeros((196, 256, 2), dtype=np.uint8)
        # Put a unique value in metadata row 192
        raw[192, 0, 0] = 0xFF
        raw[192, 0, 1] = 0xFF
        result = decode_frame(raw)
        # Row 192 should not be in output (output is only 192 rows)
        assert result.shape[0] == 192
        # The metadata value should NOT appear in the last image row
        assert result[191, 0] == 0  # row 191 (last image row) is still zero


class TestExtractMetadata:
    def test_returns_4_rows(self):
        raw = np.zeros((196, 256, 2), dtype=np.uint8)
        metadata = extract_metadata(raw)
        assert metadata.shape == (4, 256, 2)

    def test_metadata_is_last_4_rows(self):
        """Metadata should be rows 192-195 of the raw frame."""
        raw = np.zeros((196, 256, 2), dtype=np.uint8)
        raw[192, 0, 0] = 0xAA
        raw[193, 0, 0] = 0xBB
        raw[194, 0, 0] = 0xCC
        raw[195, 0, 0] = 0xDD
        metadata = extract_metadata(raw)
        assert metadata[0, 0, 0] == 0xAA
        assert metadata[1, 0, 0] == 0xBB
        assert metadata[2, 0, 0] == 0xCC
        assert metadata[3, 0, 0] == 0xDD

    def test_metadata_dtype_preserved(self):
        raw = np.zeros((196, 256, 2), dtype=np.uint8)
        metadata = extract_metadata(raw)
        assert metadata.dtype == np.uint8