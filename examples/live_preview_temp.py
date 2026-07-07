#!/usr/bin/env python3
"""Live preview with temperature readout.

Shows the thermal camera feed with real-time temperature overlay
(center, max, min, avg) calculated from libthermometry.so.

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
import time

from infiray_t2pro import T2Pro, Palette, CAMERA_LENS_13, CAMERA_LENS_6_8
from infiray_t2pro.thermometry import ThermometryLib, calculate_temperature
from infiray_t2pro.processing import agc_percentile, correct_column_fpn
from infiray_t2pro.palettes import apply_palette

# Load dark reference if available
dark_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "thermal_captures", "dark_reference.npy")
dark = None
if os.path.exists(dark_path):
    dark = np.load(dark_path)
    print(f"Loaded dark reference: mean={dark.mean():.0f} std={dark.std():.0f}")
else:
    print("No dark reference found — running without NUC correction.")

cam = T2Pro()
if dark is not None:
    cam.nuc_calib = dark

print("Starting stream...")
cam.start_stream(warmup=10)
print(f"Stream open: is_streaming={cam.is_streaming}")

tlib = ThermometryLib()

cv2.namedWindow("T2 Pro - Thermal + Temperature", cv2.WINDOW_NORMAL)
cv2.resizeWindow("T2 Pro - Thermal + Temperature", 1024, 768)

palettes = [Palette.INFERNO, Palette.JET, Palette.TURBO, Palette.HOT]
palette_idx = 0
frame_count = 0
temp_interval = 5  # Update temperature every N frames
last_result = None
last_temp_time = 0

try:
    while True:
        frame = cam.read_frame()
        if dark is not None:
            corrected = correct_column_fpn(frame)
        else:
            corrected = frame

        display = apply_palette(corrected, palettes[palette_idx])

        # Update temperature reading every 0.5 seconds
        now = time.time()
        if now - last_temp_time > 0.5:
            try:
                raw = cam.read_frame_raw()
                result = calculate_temperature(tlib, raw)
                last_result = result
                last_temp_time = now
            except Exception as e:
                pass  # Keep last reading on error

        # Temperature overlay
        if last_result is not None:
            r = last_result
            lines = [
                f"Center: {r.center_temp:.1f} C",
                f"Max: {r.max_temp:.1f} C",
                f"Min: {r.min_temp:.1f} C",
                f"Avg: {r.avg_temp:.1f} C",
                f"FPA: {r.fpa_temp:.1f} C",
                f"Emiss: {r.emissivity:.2f} Dist: {r.distance}m",
            ]
            y = 30
            for line in lines:
                cv2.putText(display, line, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y += 25

            # Temperature in top-right
            cv2.putText(display, f"{r.center_temp:.1f} C", (display.shape[1] - 150, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.putText(display, "q=quit  s=save  c=palette  n=NUC",
                    (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("T2 Pro - Thermal + Temperature", display)
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = np.datetime_as_string(np.datetime64('now'), unit='s').replace(':', '-')
            os.makedirs("thermal_captures", exist_ok=True)
            cv2.imwrite(f"thermal_captures/temp_{ts}.png", display)
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