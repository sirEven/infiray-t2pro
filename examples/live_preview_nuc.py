#!/usr/bin/env python3
"""Live preview with one-point NUC + column FPN correction.

Use this to focus the lens: rotate the lens enclosure while watching
the preview until the image is sharp.
"""

import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infiray_t2pro.decode import decode_frame
from infiray_t2pro.processing import agc_percentile, correct_column_fpn

# Load dark reference
dark_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "thermal_captures", "dark_reference.npy")
if os.path.exists(dark_path):
    dark = np.load(dark_path)
    print(f"Loaded dark reference: mean={dark.mean():.0f} std={dark.std():.0f}")
else:
    print("No dark reference found! Run capture_dark.py first.")
    sys.exit(1)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
if not cap.isOpened():
    print("ERROR: Cannot open camera")
    sys.exit(1)

# Warm up
print("Warming up sensor...")
for _ in range(10):
    cap.read()

print("Live preview with NUC + FPN correction.")
print("Rotate the lens enclosure to focus.")
print("Press 'q' to quit, 's' to save frame.")

frame_count = 0
while True:
    ret, raw = cap.read()
    if not ret:
        break

    # Decode raw YUYV to 16-bit thermal
    thermal = decode_frame(raw)

    # Apply one-point NUC (dark subtraction)
    corrected = thermal - dark

    # Apply column FPN removal
    corrected = correct_column_fpn(corrected)

    # Percentile AGC for display
    display = agc_percentile(corrected, low_percentile=1, high_percentile=99)

    # Convert grayscale to BGR for display
    display_bgr = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

    # Add label
    cv2.putText(display_bgr, "NUC+FPN | Press 'q' quit, 's' save", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imshow("T2 Pro - Live (NUC+FPN)", display_bgr)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        os.makedirs("thermal_captures", exist_ok=True)
        cv2.imwrite(f"thermal_captures/focused_frame_{frame_count:04d}.png", display)
        print(f"Saved frame {frame_count}")
        frame_count += 1

cap.release()
cv2.destroyAllWindows()
print("Done.")