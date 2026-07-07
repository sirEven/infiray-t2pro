"""Snapshot mode — save a complete thermal snapshot to disk.

A snapshot captures everything needed for an inspection deliverable:
  - PNG: rendered thermal image (with palette and processing applied)
  - .npy: raw 16-bit thermal data (for re-processing later)
  - .json: temperature metadata (center, max, min, avg, FPA, emissivity, etc.)

All files go into a single directory named by timestamp: snapshots/YYYY-MM-DD_HH-MM-SS/

Typical usage during streaming:
    cam.start_stream(warmup=10)
    # ... read frames ...
    snap = take_snapshot(cam, tlib)
    print(snap)  # Shows summary
"""

import json
import os
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from .camera import T2Pro
from .palettes import Palette, apply_palette, PALETTE_NAMES
from .processing import AgcAutoRange, correct_column_fpn, denoise_thermal
from .thermometry import ThermometryLib, TemperatureResult, calculate_temperature


@dataclass
class SnapshotResult:
    """Result of a snapshot operation."""
    directory: str           # Path to snapshot directory
    png_path: str            # Path to rendered PNG
    npy_path: str            # Path to raw numpy data
    json_path: str           # Path to metadata JSON
    center_temp: float       # Center temperature °C
    max_temp: float          # Maximum temperature °C
    min_temp: float          # Minimum temperature °C
    avg_temp: float          # Average temperature °C
    fpa_temp: float          # FPA temperature °C
    timestamp: str           # ISO 8601 timestamp

    def __str__(self) -> str:
        return (
            f"Snapshot: {self.timestamp}\n"
            f"  Center: {self.center_temp:.1f}°C  "
            f"Max: {self.max_temp:.1f}°C  "
            f"Min: {self.min_temp:.1f}°C  "
            f"Avg: {self.avg_temp:.1f}°C\n"
            f"  FPA: {self.fpa_temp:.1f}°C\n"
            f"  Files: {self.directory}"
        )


def take_snapshot(
    cam: T2Pro,
    tlib: Optional[ThermometryLib] = None,
    temp_result: Optional[TemperatureResult] = None,
    output_dir: str = "snapshots",
    palette: Palette = Palette.INFERNO,
    scale: int = 5,
    agc: Optional[AgcAutoRange] = None,
    dark: Optional[np.ndarray] = None,
    apply_denoise: bool = True,
) -> SnapshotResult:
    """Take a complete thermal snapshot and save to disk.

    Must be called while the stream is open (during start_stream/stop_stream).

    Either provide tlib (to calculate temperatures from a raw frame) or
    temp_result (pre-calculated temperatures, e.g. from a previous reading).

    Args:
        cam: T2Pro camera instance (must be streaming).
        tlib: ThermometryLib instance for temperature calculation.
              Mutually exclusive with temp_result.
        temp_result: Pre-calculated temperature result.
                     Mutually exclusive with tlib.
        output_dir: Base directory for snapshots.
        palette: Color palette for the PNG rendering.
        scale: Upscale factor for the PNG (5 = 1280x960).
        agc: AgcAutoRange instance for smooth contrast. None uses per-frame.
        dark: Dark reference for NUC correction. None skips correction.
        apply_denoise: Whether to apply bilateral filter denoising.

    Returns:
        SnapshotResult with file paths and temperature summary.

    Raises:
        StreamClosedError: If the camera stream is not open.
        FrameReadError: If the frame read fails.
        ValueError: If neither tlib nor temp_result is provided.
    """
    if tlib is None and temp_result is None:
        raise ValueError("Provide either tlib or temp_result")

    # Read processed frame for rendering
    frame = cam.read_frame()
    if dark is not None:
        frame = correct_column_fpn(frame)
    if apply_denoise:
        frame = denoise_thermal(frame)

    # Calculate or use provided temperatures
    if temp_result is not None:
        temps = temp_result
        raw_frame = None
    else:
        raw_frame = cam.read_frame_raw()
        temps = calculate_temperature(tlib, raw_frame)

    # Render PNG
    rendered = apply_palette(frame, palette, scale=scale, agc=agc)

    # Create timestamped directory
    ts = datetime.now()
    ts_str = ts.strftime("%Y-%m-%d_%H-%M-%S")
    snap_dir = os.path.join(output_dir, ts_str)
    os.makedirs(snap_dir, exist_ok=True)

    # Save PNG
    png_path = os.path.join(snap_dir, f"thermal_{ts_str}.png")
    # Overlay temperature info on PNG
    overlay_img = _add_temp_overlay(rendered, temps, palette)
    import cv2
    cv2.imwrite(png_path, overlay_img)

    # Save raw numpy data
    npy_path = os.path.join(snap_dir, f"raw_{ts_str}.npy")
    if raw_frame is not None:
        np.save(npy_path, raw_frame)
    else:
        # Save the processed frame as fallback if no raw available
        np.save(npy_path, frame.astype(np.uint16))

    # Save metadata JSON
    json_path = os.path.join(snap_dir, f"metadata_{ts_str}.json")
    metadata = {
        "timestamp": ts.isoformat(),
        "palette": PALETTE_NAMES.get(palette, str(palette)),
        "scale": scale,
        "image_size": f"{rendered.shape[1]}x{rendered.shape[0]}",
        "sensor_size": "256x192",
        "temperatures": {
            "center_c": round(temps.center_temp, 2),
            "max_c": round(temps.max_temp, 2),
            "max_x": temps.max_x,
            "max_y": temps.max_y,
            "min_c": round(temps.min_temp, 2),
            "min_x": temps.min_x,
            "min_y": temps.min_y,
            "avg_c": round(temps.avg_temp, 2),
            "fpa_c": round(temps.fpa_temp, 2),
        },
        "parameters": {
            "emissivity": temps.emissivity,
            "distance_m": temps.distance,
            "humidity": temps.humidity,
            "correction": temps.correction,
            "reflection_temp": round(temps.reflection_temp, 2),
            "ambient_temp": round(temps.ambient_temp, 2),
        },
        "files": {
            "png": os.path.basename(png_path),
            "npy": os.path.basename(npy_path),
            "json": os.path.basename(json_path),
        },
    }
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return SnapshotResult(
        directory=snap_dir,
        png_path=png_path,
        npy_path=npy_path,
        json_path=json_path,
        center_temp=round(temp_result.center_temp, 2),
        max_temp=round(temp_result.max_temp, 2),
        min_temp=round(temp_result.min_temp, 2),
        avg_temp=round(temp_result.avg_temp, 2),
        fpa_temp=round(temp_result.fpa_temp, 2),
        timestamp=ts_str,
    )


def _add_temp_overlay(
    img: np.ndarray,
    temp: TemperatureResult,
    palette: Palette,
) -> np.ndarray:
    """Add temperature overlay text to the rendered image."""
    import cv2

    # Temperature readout — top left
    lines = [
        f"Center: {temp.center_temp:.1f} C",
        f"Max: {temp.max_temp:.1f} C ({temp.max_x},{temp.max_y})",
        f"Min: {temp.min_temp:.1f} C ({temp.min_x},{temp.min_y})",
        f"Avg: {temp.avg_temp:.1f} C",
        f"FPA: {temp.fpa_temp:.1f} C",
        f"Emiss: {temp.emissivity:.2f}  Dist: {temp.distance}m",
    ]
    y = 30
    for line in lines:
        cv2.putText(img, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                    cv2.LINE_AA)
        y += 22

    # Big temperature — top right
    cv2.putText(img, f"{temp.center_temp:.1f} C",
                (img.shape[1] - 180, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                cv2.LINE_AA)

    # Timestamp — bottom right
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(img, ts_str,
                (img.shape[1] - 220, img.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
                cv2.LINE_AA)

    return img