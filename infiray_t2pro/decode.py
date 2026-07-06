"""Raw YUYV frame decoding for the InfiRay T2 Pro.

The T2 Pro outputs 256×196 YUYV frames via UVC bulk transfer.
Each pixel is 2 bytes: channel 0 = LSB (low byte), channel 1 = MSB (high byte).
Rows 0-191 are image data, rows 192-195 are metadata.
"""

import numpy as np

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 192
METADATA_ROWS = 4
TOTAL_ROWS = IMAGE_HEIGHT + METADATA_ROWS  # 196


def decode_frame(raw: np.ndarray) -> np.ndarray:
    """Decode a raw YUYV frame to 16-bit thermal values.

    Args:
        raw: Raw frame from OpenCV (196×256×2, uint8).

    Returns:
        2D float32 array (192×256) of 16-bit thermal values.
    """
    msb = raw[:IMAGE_HEIGHT, :, 1].astype(np.float32)
    lsb = raw[:IMAGE_HEIGHT, :, 0].astype(np.float32)
    return msb * 256.0 + lsb


def extract_metadata(raw: np.ndarray) -> np.ndarray:
    """Extract the 4 metadata rows from a raw frame.

    Args:
        raw: Raw frame from OpenCV (196×256×2, uint8).

    Returns:
        Metadata array (4×256×2, uint8) containing rows 192-195.
    """
    return raw[IMAGE_HEIGHT:, :, :].copy()