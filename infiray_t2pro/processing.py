"""Thermal image processing: NUC correction, AGC, and column FPN removal.

The T2 Pro's raw 16-bit output has two types of per-pixel noise:
- Fixed Pattern Noise (FPN): each pixel has a different offset (dark current)
- Gain variation: each pixel has a different responsivity (vertical stripes)

Two-point NUC corrects both:
    corrected = (raw - dark) / (bright - dark) * mean_response

This maps each pixel to a common gain reference, eliminating both offset
and gain non-uniformity.
"""

import numpy as np
from typing import Optional


def two_point_nuc(
    raw: np.ndarray,
    dark: np.ndarray,
    bright: np.ndarray,
    target_range: float = None,
) -> np.ndarray:
    """Apply two-point NUC correction using dark and bright reference frames.

    Args:
        raw: Raw thermal frame (192×256, float32).
        dark: Dark reference frame captured with lens covered (same shape).
        bright: Bright reference frame of a uniform hot scene (same shape).
        target_range: If set, scale the output so the mean response equals this value.
                      If None, the output preserves the original scale.

    Returns:
        Corrected thermal frame with per-pixel offset and gain normalized.
    """
    # Subtract dark frame (offset correction)
    offset_corrected = raw - dark
    bright_corrected = bright - dark

    # Compute mean response of the bright scene (reference gain)
    mean_response = np.mean(bright_corrected)

    # Per-pixel gain correction: normalize each pixel to the mean
    # corrected = (raw - dark) / (bright - dark) * mean_response
    # Avoid division by zero for dead pixels
    gain_map = np.where(
        bright_corrected > 0,
        mean_response / bright_corrected,
        0.0,
    )

    corrected = offset_corrected * gain_map

    if target_range is not None:
        # Scale to target range
        current_mean = np.mean(corrected)
        if current_mean > 0:
            corrected = corrected * (target_range / current_mean)

    return corrected


def agc_linear(
    thermal: np.ndarray,
    min_out: float = 0.0,
    max_out: float = 255.0,
) -> np.ndarray:
    """Simple linear Automatic Gain Control — scale to fill output range.

    Maps the full input range to [min_out, max_out].
    Outliers in the input will crush the dynamic range — use agc_percentile
    for better results on real thermal data.

    Args:
        thermal: Input thermal data (any shape, float32).
        min_out: Minimum output value.
        max_out: Maximum output value.

    Returns:
        uint8 array scaled to [min_out, max_out].
    """
    thermal = thermal.astype(np.float32)
    min_in = thermal.min()
    max_in = thermal.max()

    if max_in == min_in:
        return np.full(thermal.shape, int(max_out), dtype=np.uint8)

    scale = (max_out - min_out) / (max_in - min_in)
    offset = -scale * min_in + min_out

    agc_img = scale * thermal + offset
    agc_img = np.clip(agc_img, min_out, max_out)

    return agc_img.astype(np.uint8)


def agc_percentile(
    thermal: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    min_out: float = 0.0,
    max_out: float = 255.0,
) -> np.ndarray:
    """Percentile-based AGC — clips outlier pixels before scaling.

    This is better than agc_linear for thermal data because a few dead/hot
    pixels won't crush the dynamic range. The low/high percentiles define
    which pixels to clip before scaling.

    Based on advice from RMHansen on the thermal imagery Discord:
    "Scale the pixel data from the native 14-bit resolution to 8-bit,
    but don't let outlier pixels set the range."

    Args:
        thermal: Input thermal data (any shape, float32).
        low_percentile: Lower clip percentile (0-100). Pixels below this are clipped.
        high_percentile: Upper clip percentile (0-100). Pixels above this are clipped.
        min_out: Minimum output value.
        max_out: Maximum output value.

    Returns:
        uint8 array scaled to [min_out, max_out], with outliers clipped.
    """
    thermal = thermal.astype(np.float32)
    min_in = np.percentile(thermal, low_percentile)
    max_in = np.percentile(thermal, high_percentile)

    if max_in == min_in:
        return np.full(thermal.shape, int(max_out), dtype=np.uint8)

    # Clip outliers, then scale
    clipped = np.clip(thermal, min_in, max_in)
    scale = (max_out - min_out) / (max_in - min_in)
    offset = -scale * min_in + min_out

    agc_img = scale * clipped + offset
    agc_img = np.clip(agc_img, min_out, max_out)

    return agc_img.astype(np.uint8)


def correct_column_fpn(thermal: np.ndarray) -> np.ndarray:
    """Remove column-wise Fixed Pattern Noise (vertical stripes).

    Each column has a slight offset difference. This function computes
    the per-column mean deviation and subtracts it, flattening vertical stripes
    while preserving horizontal temperature gradients.

    Args:
        thermal: Input thermal data (192×256, float32).

    Returns:
        Column-FPN-corrected thermal data (same shape).
    """
    # Per-column mean
    col_means = np.mean(thermal, axis=0, keepdims=True)
    # Global mean (what all columns should converge to)
    global_mean = np.mean(thermal)
    # Per-column offset from global mean
    col_offsets = col_means - global_mean

    return thermal - col_offsets