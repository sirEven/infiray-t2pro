"""Color palettes for thermal image visualization."""

import cv2
import numpy as np
from enum import IntEnum


class Palette(IntEnum):
    """Thermal color palettes matching the T2 Pro's built-in options."""
    INFERNO = cv2.COLORMAP_INFERNO
    JET = cv2.COLORMAP_JET
    TURBO = cv2.COLORMAP_TURBO
    HOT = cv2.COLORMAP_HOT
    GRAYSCALE = -1


def apply_palette(thermal: np.ndarray, palette: Palette = Palette.INFERNO) -> np.ndarray:
    """Apply a color palette to 16-bit thermal data.

    Args:
        thermal: 2D array of thermal values (typically 192×256).
        palette: Color palette to apply.

    Returns:
        BGR image (H×W×3, uint8), scaled up 4× for visibility.
    """
    normalized = cv2.normalize(
        thermal.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    if palette == Palette.GRAYSCALE:
        colored = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    else:
        colored = cv2.applyColorMap(normalized, int(palette))

    return cv2.resize(colored, (256 * 4, 192 * 4), interpolation=cv2.INTER_CUBIC)