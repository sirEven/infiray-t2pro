"""InfiRay T2 Pro — Python driver for the InfiRay T2 Pro USB thermal camera."""

from .commands import Command, pack_command
from .palettes import Palette, apply_palette
from .decode import decode_frame, extract_metadata, IMAGE_HEIGHT, IMAGE_WIDTH, TOTAL_ROWS
from .processing import two_point_nuc, agc_linear, agc_percentile, correct_column_fpn
from .camera import T2Pro, VideoBackend, V4L2Backend, StreamClosedError, StreamOpenError, FrameReadError

__version__ = "0.1.0"
__all__ = [
    "Command", "pack_command",
    "Palette", "apply_palette",
    "decode_frame", "extract_metadata", "IMAGE_HEIGHT", "IMAGE_WIDTH", "TOTAL_ROWS",
    "two_point_nuc", "agc_linear", "agc_percentile", "correct_column_fpn",
    "T2Pro", "VideoBackend", "V4L2Backend", "StreamClosedError", "StreamOpenError", "FrameReadError",
]