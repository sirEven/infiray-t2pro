#!/usr/bin/env python3
"""Example: Live thermal preview with shutter trigger."""

from infiray_t2pro import T2Pro

cam = T2Pro()

# Trigger shutter calibration first
print("Triggering shutter calibration...")
cam.trigger_shutter()

# Start live preview
print("Starting live preview (press 'q' to quit, 's' to save, 'c' to change colormap)")
cam.live_preview()