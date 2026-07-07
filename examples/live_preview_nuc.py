#!/usr/bin/env python3
"""Live preview with one-point NUC + column FPN correction.

Uses the streaming API (start_stream / read_frame / stop_stream) instead of
manual VideoCapture. This is the same pattern the future wrapper will use.

Controls:
    q - quit
    s - save current frame
    c - cycle color palette
    n - trigger NUC calibration (cover lens first!)
"""

import numpy as np
import cv2
import sys
import os

from infiray_t2pro import T2Pro, Palette
from infiray_t2pro.processing import agc_percentile, correct_column_fpn
from infiray_t2pro.palettes import apply_palette

# Load dark reference
dark_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "thermal_captures", "dark_reference.npy")
if os.path.exists(dark_path):
    dark = np.load(dark_path)
    print(f"Loaded dark reference: mean={dark.mean():.0f} std={dark.std():.0f}")
else:
    print("No dark reference found! Run capture_dark.py first.")
    sys.exit(1)

cam = T2Pro()
cam.nuc_calib = dark  # Set NUC calibration so read_frame() applies it

print("Starting stream...")
cam.start_stream(warmup=10)
print(f"Stream open: is_streaming={cam.is_streaming}")

cv2.namedWindow("T2 Pro - Streaming NUC+FPN", cv2.WINDOW_NORMAL)
cv2.resizeWindow("T2 Pro - Streaming NUC+FPN", 1024, 768)

palettes = [Palette.INFERNO, Palette.JET, Palette.TURBO, Palette.HOT]
palette_idx = 0
frame_count = 0

try:
    while True:
        frame = cam.read_frame()
        # read_frame() with NUC calib loaded already subtracts dark reference
        # (after the first frame). Apply column FPN on top.
        corrected = correct_column_fpn(frame)

        # Apply palette to corrected thermal data for color rendering
        display = apply_palette(corrected, palettes[palette_idx])

        # Info overlay
        info = f"std={corrected.std():.0f} min={corrected.min():.0f} max={corrected.max():.0f} | #{frame_count}"
        cv2.putText(display, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "q=quit  s=save  c=palette  n=NUC calib", (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("T2 Pro - Streaming NUC+FPN", display)
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = np.datetime_as_string(np.datetime64('now'), unit='s').replace(':', '-')
            os.makedirs("thermal_captures", exist_ok=True)
            cv2.imwrite(f"thermal_captures/stream_{ts}.png", display)
            print(f"Saved frame #{frame_count}")
        elif key == ord('c'):
            palette_idx = (palette_idx + 1) % len(palettes)
            print(f"Palette: {palettes[palette_idx].name}")
        elif key == ord('n'):
            print("Triggering NUC calibration...")
            cam.trigger_shutter()
            print("Done.")
except KeyboardInterrupt:
    print("\nInterrupted.")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
finally:
    print("Stopping stream...")
    cam.stop_stream()
    print(f"Stream closed: is_streaming={cam.is_streaming}")
    cv2.destroyAllWindows()

print(f"Captured {frame_count} frames total.")