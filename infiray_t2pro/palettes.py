"""Color palettes for thermal image visualization."""

import cv2
import numpy as np
from enum import IntEnum


class Palette(IntEnum):
    """Thermal color palettes.

    Standard thermal palettes first (white-hot is the most commonly used
    in professional thermal imaging), then OpenCV colormaps for variety.
    """
    # Professional thermal palettes
    WHITE_HOT = -1       # Grayscale: hot=white, cold=black
    BLACK_HOT = -2       # Grayscale: hot=black, cold=white
    # OpenCV colormaps
    INFERNO = cv2.COLORMAP_INFERNO
    JET = cv2.COLORMAP_JET
    TURBO = cv2.COLORMAP_TURBO
    HOT = cv2.COLORMAP_HOT
    VIRIDIS = cv2.COLORMAP_VIRIDIS
    PLASMA = cv2.COLORMAP_PLASMA
    MAGMA = cv2.COLORMAP_MAGMA
    CIVIDIS = cv2.COLORMAP_CIVIDIS
    TURBO_WINTER = cv2.COLORMAP_WINTER


# Human-readable names for display
PALETTE_NAMES = {
    Palette.WHITE_HOT: "White Hot",
    Palette.BLACK_HOT: "Black Hot",
    Palette.INFERNO: "Inferno",
    Palette.JET: "Jet",
    Palette.TURBO: "Turbo",
    Palette.HOT: "Hot",
    Palette.VIRIDIS: "Viridis",
    Palette.PLASMA: "Plasma",
    Palette.MAGMA: "Magma",
    Palette.CIVIDIS: "Cividis",
    Palette.TURBO_WINTER: "Winter",
}


def apply_palette(
    thermal: np.ndarray,
    palette: Palette = Palette.INFERNO,
    scale: int = 5,
    agc=None,
) -> np.ndarray:
    """Apply a color palette to 16-bit thermal data.

    Args:
        thermal: 2D array of thermal values (typically 192x256).
        palette: Color palette to apply.
        scale: Upscale factor (5 = 1280x960 default). Use 3 for smaller, 8 for larger.
        agc: AgcAutoRange instance for temporally-smooth contrast. If None,
             uses per-frame cv2.normalize (causes flicker and detail loss).

    Returns:
        BGR image (H*scale x W*scale x 3, uint8), upscaled for display.
    """
    if agc is not None:
        normalized = agc.apply(thermal)
    else:
        normalized = cv2.normalize(
            thermal.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

    if palette == Palette.WHITE_HOT:
        # Hot = white, cold = black (standard thermal)
        colored = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    elif palette == Palette.BLACK_HOT:
        # Hot = black, cold = white (inverted)
        inverted = 255 - normalized
        colored = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)
    else:
        colored = cv2.applyColorMap(normalized, int(palette))

    out_w = thermal.shape[1] * scale
    out_h = thermal.shape[0] * scale
    return cv2.resize(colored, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)