#!/usr/bin/env python3
"""Example: Capture a single thermal frame and save it."""

from infiray_t2pro import T2Pro, Palette

cam = T2Pro()
frame = cam.capture()
cam.save(frame, "thermal_captures/example_capture.png", palette=Palette.INFERNO)
print(f"Frame stats: min={frame.min():.0f} max={frame.max():.0f} mean={frame.mean():.0f}")