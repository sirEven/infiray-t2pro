#!/usr/bin/env python3
"""Example: Two-point NUC calibration for high-quality thermal images.

The T2 Pro has per-pixel offset and gain variation (visible as vertical stripes).
Two-point NUC corrects both:

1. Cover the lens → capture a DARK reference (per-pixel offsets)
2. Point at a uniform warm surface → capture a BRIGHT reference (per-pixel gains)
3. Each subsequent frame is corrected: (raw - dark) * (mean_response / (bright - dark))

This eliminates both fixed-pattern noise (FPN) and gain non-uniformity.
"""

from infiray_t2pro import T2Pro
from infiray_t2pro.processing import two_point_nuc, agc_percentile, correct_column_fpn

cam = T2Pro()

# Step 1: Capture dark reference (cover the lens!)
input("Cover the lens completely, then press Enter...")
dark = cam.capture_raw(n_frames=10, trash=5)
print(f"Dark frame: mean={dark.mean():.0f} std={dark.std():.0f}")

# Step 2: Capture bright reference (point at a uniform warm surface)
input("Point at a uniform warm surface (e.g. wall, ceiling), then press Enter...")
bright = cam.capture_raw(n_frames=10, trash=5)
print(f"Bright frame: mean={bright.mean():.0f} std={bright.std():.0f}")

# Step 3: Capture a scene and apply two-point NUC
input("Point at your target scene, then press Enter...")
raw = cam.capture_raw(n_frames=5, trash=5)

# Apply corrections in order
corrected = two_point_nuc(raw, dark, bright)
print(f"After two-point NUC: mean={corrected.mean():.0f} std={corrected.std():.0f}")

# Optional: remove any remaining column stripes
corrected = correct_column_fpn(corrected)
print(f"After column FPN: mean={corrected.mean():.0f} std={corrected.std():.0f}")

# Apply percentile-based AGC for best contrast
rendered = agc_percentile(corrected, low_percentile=1, high_percentile=99)

# Save the result
import cv2
cv2.imwrite("thermal_captures/two_point_nuc.png", rendered)
print("Saved to thermal_captures/two_point_nuc.png")

# Save reference frames for later use
import numpy as np
np.save("thermal_captures/dark_reference.npy", dark)
np.save("thermal_captures/bright_reference.npy", bright)
print("Reference frames saved for future use.")