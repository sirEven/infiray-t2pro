#!/usr/bin/env python3
"""Live preview with temperature readout and smooth AGC.

Shows the thermal camera feed with real-time temperature overlay
(center, max, min, avg) calculated from libthermometry.so.

Uses AgcAutoRange for temporally-smooth contrast instead of per-frame
normalization. This preserves fine detail at distance and eliminates
the pixelation/flicker that per-frame normalize causes.

Controls:
    q - quit
    s - save current frame (PNG + raw numpy)
    c - cycle color palette
    +/- - zoom in/out (change upscale factor)
    n - trigger NUC calibration (cover lens first!)
    a - toggle AGC mode (smooth vs per-frame)
    d - toggle denoise (bilateral filter on/off)
    r - reset AGC range (use if image looks washed out)
"""

import numpy as np
import cv2
import sys
import os
import time

from infiray_t2pro import T2Pro, Palette
from infiray_t2pro.thermometry import ThermometryLib, calculate_temperature
from infiray_t2pro.palettes import PALETTE_NAMES, apply_palette
from infiray_t2pro.processing import AgcAutoRange, correct_column_fpn, denoise_thermal

# Load dark reference if available
dark_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "thermal_captures", "dark_reference.npy"
)
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

# Smooth AGC — adapts slowly, no per-frame flicker
agc = AgcAutoRange(
    low_percentile=0.5,
    high_percentile=99.5,
    adapt_speed=0.08,
    min_range=80.0,
)

# All available palettes — cycle with 'c'
palettes = list(Palette)
palette_idx = 0
scale = 5  # 5x upscale = 1280x960 display
use_smooth_agc = True
use_denoise = True

cv2.namedWindow("T2 Pro - Thermal + Temperature", cv2.WINDOW_NORMAL)
cv2.resizeWindow("T2 Pro - Thermal + Temperature", 1280, 960)

frame_count = 0
last_result = None
last_temp_time = 0

try:
    while True:
        frame = cam.read_frame()
        corrected = correct_column_fpn(frame) if dark is not None else frame
        denoised = denoise_thermal(corrected) if use_denoise else corrected

        display = apply_palette(
            denoised, palettes[palette_idx], scale=scale,
            agc=agc if use_smooth_agc else None,
        )

        # Temperature readout — update every 0.5s
        now = time.time()
        if now - last_temp_time > 0.5:
            try:
                raw = cam.read_frame_raw()
                result = calculate_temperature(tlib, raw)
                last_result = result
                last_temp_time = now
            except Exception:
                pass  # Keep last reading on error

        # Overlay
        if last_result is not None:
            r = last_result
            lines = [
                f"Center: {r.center_temp:.1f} C",
                f"Max: {r.max_temp:.1f} C",
                f"Min: {r.min_temp:.1f} C",
                f"Avg: {r.avg_temp:.1f} C",
                f"FPA: {r.fpa_temp:.1f} C",
                f"Emiss: {r.emissivity:.2f}  Dist: {r.distance}m",
            ]
            y = 30
            for line in lines:
                cv2.putText(display, line, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                            cv2.LINE_AA)
                y += 22

            # Big temperature top-right
            cv2.putText(display, f"{r.center_temp:.1f} C",
                        (display.shape[1] - 180, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)

        # Status bar
        agc_mode = "smooth" if use_smooth_agc else "per-frame"
        denoise_str = "on" if use_denoise else "off"
        pal_name = PALETTE_NAMES.get(palettes[palette_idx], str(palettes[palette_idx]))
        cv2.putText(display, f"{pal_name}  AGC:{agc_mode}  DNR:{denoise_str}  |  q=quit s=save c=palette n=NUC a=AGC d=DNR r=reset +/-=zoom",
                    (10, display.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
                    cv2.LINE_AA)

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
            pal_name = PALETTE_NAMES.get(palettes[palette_idx], str(palettes[palette_idx]))
            print(f"Palette: {pal_name}")
        elif key == ord('n'):
            print("Triggering NUC calibration...")
            cam.trigger_shutter()
            print("Done.")
        elif key == ord('a'):
            use_smooth_agc = not use_smooth_agc
            if not use_smooth_agc:
                agc.reset()
            print(f"AGC: {'smooth (temporal)' if use_smooth_agc else 'per-frame (legacy)'}")
        elif key == ord('d'):
            use_denoise = not use_denoise
            print(f"Denoise: {'on (bilateral filter)' if use_denoise else 'off (raw)'}")
        elif key in (ord('+'), ord('=')):
            scale = min(scale + 1, 8)
            print(f"Scale: {scale}x ({256*scale}x{192*scale})")
        elif key == ord('-'):
            scale = max(scale - 1, 2)
            print(f"Scale: {scale}x ({256*scale}x{192*scale})")
        elif key == ord('r'):
            agc.reset()
            print("AGC range reset.")

except KeyboardInterrupt:
    print("\nInterrupted.")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("Stopping stream...")
    cam.stop_stream()
    print(f"Stream closed: is_streaming={cam.is_streaming}")
    cv2.destroyAllWindows()

print(f"Captured {frame_count} frames total.")