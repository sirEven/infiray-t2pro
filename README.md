# InfiRay T2 Pro — Python Driver

A clean, well-tested Python driver for the InfiRay T2 Pro USB thermal camera.
Designed for Linux. Built with TDD (49 tests, all passing).

## Install

```bash
git clone https://github.com/<your-org>/infiray-t2pro.git
cd infiray-t2pro
pip install -e .
```

## Quick Start

```python
from infiray_t2pro import T2Pro, Palette

# Capture a single frame
cam = T2Pro()
frame = cam.capture()
cam.save(frame, "thermal.png")

# Live preview (press 'q' to quit, 's' to save, 'c' to change colormap)
cam.live_preview()

# NUC calibration (cover the lens first!)
cam.calibrate_nuc_manual()
cam.save_nuc_calibration("my_calib.npy")

# Later: load calibration and capture corrected frames
cam.load_nuc_calibration("my_calib.npy")
corrected = cam.capture()  # NUC correction applied automatically
```

## Architecture

```
infiray_t2pro/
├── __init__.py      — Public API
├── commands.py      — Vendor command enum + packing logic
├── palettes.py      — Color palette enum + rendering
├── decode.py        — Raw YUYV frame decoding (16-bit thermal extraction)
└── camera.py        — T2Pro class + injectable VideoBackend
```

The `VideoBackend` abstract class enables testing without hardware.
`V4L2Backend` is the default real implementation. Inject a `FakeVideoBackend`
in tests or use a custom backend for different hardware.

## Examples

- `examples/capture_single.py` — Capture and save one frame
- `examples/live_preview.py` — Live thermal video feed
- `examples/nuc_calibration.py` — NUC calibration workflow
- `examples/extract_metadata.py` — Capture with metadata row extraction

## Tests

```bash
pytest tests/ -v
```

49 tests covering: command packing, palette rendering, frame decoding,
NUC calibration, and camera logic (with mock hardware).

## Requirements

- Linux with UVC support (uvcvideo kernel module)
- OpenCV (`pip install opencv-python`)
- v4l-utils (`v4l2-ctl` command)
- Optional: pyusb for UVC control transfers

## Camera Specs

| Spec | Value |
|---|---|
| Sensor | 256×192 thermal (16-bit) |
| Frame rate | 25 fps |
| USB | UVC, VID:PID 04b4:0100 (Cypress bridge) |
| Format | YUYV 4:2:2, 196×256×2 bytes (192 image + 4 metadata rows) |
| Temperature range | -20°C to 120°C (high gain) / -20°C to 400°C (low gain) |

## Status: Alpha

**Working:**
- ✅ Raw frame capture (256×192, 25fps, 16-bit)
- ✅ Live preview with multiple color palettes
- ✅ NUC calibration (manual dark frame subtraction)
- ✅ Vendor commands: shutter trigger, palette switching, gain selection
- ✅ Metadata row extraction (4 rows per frame)

**Not yet implemented:**
- ⚠️ Temperature reading (°C) — requires thermometry calibration
- ⚠️ Automatic NUC via mechanical shutter (shutter triggers but auto-reopens too fast to capture)
- ⚠️ Gain switching (0x8020/0x8021) can destabilize the camera — needs more work

**Tested on:** InfiRay T2 Pro (single unit), Arch Linux, x86_64

## License

MIT — see [LICENSE](LICENSE)