"""Vendor command definitions for the InfiRay T2 Pro.

The T2 Pro sends vendor commands via V4L2_CID_ZOOM_ABSOLUTE (control ID 512).
Commands are packed as 16-bit integers: (command_byte << 8) | data_byte.
"""

from enum import IntEnum


class Command(IntEnum):
    """Vendor commands sent via V4L2_CID_ZOOM_ABSOLUTE."""

    # Control commands (0x80XX)
    SHUTTER_CLOSE = 0x8000
    BACKGROUND_CALIBRATION = 0x8001
    RAW_OUTPUT = 0x8002
    NUC_16BIT = 0x8004
    YUYV_OUTPUT = 0x8005
    SAVE_CONFIG = 0x80FE
    SAVE_PARAMETERS = 0x80FF

    # Gain commands (0x802X)
    HIGH_GAIN = 0x8020
    LOW_GAIN = 0x8021

    # Palette commands (0x88XX)
    PALETTE_WHITE_HOT = 0x8800
    PALETTE_BLACK_HOT = 0x8801
    PALETTE_IRON = 0x8802
    PALETTE_LAVA = 0x8803
    PALETTE_RAINBOW = 0x8804
    PALETTE_IRON_GREY = 0x8805
    PALETTE_RED_HOT = 0x8806
    PALETTE_RAINBOW2 = 0x8807

    # Default
    DEFAULT = 1


def pack_command(command: int, data: int = 0) -> int:
    """Pack a command and data byte into a zoom_absolute value.

    Args:
        command: Command byte (upper 8 bits).
        data: Data byte (lower 8 bits), masked to 0xFF.

    Returns:
        16-bit integer for zoom_absolute.
    """
    return (command << 8) | (data & 0xFF)