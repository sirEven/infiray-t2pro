#!/usr/bin/env python3
"""Example: NUC calibration and corrected capture."""

from infiray_t2pro import T2Pro

cam = T2Pro()

# Step 1: Cover the lens, then run calibration
print("Cover the lens with your hand, then press Enter...")
input()
cam.calibrate_nuc_manual()
cam.save_nuc_calibration("my_calib.npy")
print("Calibration saved!")

# Step 2: Uncover, point at something warm
print("\nUncover the lens and point at something warm, then press Enter...")
input()
frame = cam.capture()  # NUC correction applied automatically
cam.save(frame, "thermal_captures/nuc_corrected.png")
print(f"Corrected frame: std={frame.std():.0f}")