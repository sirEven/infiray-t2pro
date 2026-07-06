#!/usr/bin/env python3
"""Example: Capture frame with metadata extraction."""

from infiray_t2pro import T2Pro

cam = T2Pro()
image, metadata = cam.capture_with_metadata()

print(f"Image shape: {image.shape}")
print(f"Metadata shape: {metadata.shape}")
print(f"Metadata first 10 bytes: {metadata.flatten()[:20]}")

cam.save(image, "thermal_captures/with_metadata.png")