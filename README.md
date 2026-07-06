# InfiRay T2 Pro — Python Driver

A clean, well-tested Python driver for the InfiRay T2 Pro USB thermal camera.
Designed for Linux. Built with TDD (66 tests, all passing).

## Install

```bash
git clone https://github.com/sirEven/infiray-t2pro.git
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
├── processing.py    — Two-point NUC, AGC, column FPN correction
└── camera.py        — T2Pro class + injectable VideoBackend
```

The `VideoBackend` abstract class enables testing without hardware.
`V4L2Backend` is the default real implementation. Inject a `FakeVideoBackend`
in tests or use a custom backend for different hardware.

## Image Processing

Raw T2 Pro images have per-pixel offset and gain variation (visible as vertical
stripes). The `processing` module provides three correction functions:

```python
from infiray_t2pro.processing import two_point_nuc, agc_percentile, correct_column_fpn

# Two-point NUC: correct per-pixel offset AND gain
corrected = two_point_nuc(raw_frame, dark_ref, bright_ref)

# Column FPN: remove remaining vertical stripes
corrected = correct_column_fpn(corrected)

# Percentile-based AGC: best contrast, clips outlier pixels
rendered = agc_percentile(corrected, low_percentile=1, high_percentile=99)
```

**Two-point NUC** requires two reference frames:
1. **Dark reference**: captured with lens covered (per-pixel offsets)
2. **Bright reference**: captured pointing at a uniform warm surface (per-pixel gains)

See `examples/two_point_nuc.py` for the full workflow.

## Examples

- `examples/capture_single.py` — Capture and save one frame
- `examples/live_preview.py` — Live thermal video feed
- `examples/nuc_calibration.py` — NUC calibration workflow
- `examples/extract_metadata.py` — Capture with metadata row extraction
- `examples/two_point_nuc.py` — Two-point NUC for high-quality images (removes stripes + FPN)
- `examples/live_preview_nuc.py` — Live preview with NUC + FPN correction (use for manual focus)

## Tests

```bash
pytest tests/ -v
```

66 tests covering: command packing, palette rendering, frame decoding,
NUC calibration, first-frame skip, auto-load calibration, two-point NUC,
AGC (linear + percentile), column FPN correction, and camera logic
(with mock hardware).

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
| Focus | Manual (rotate lens enclosure by hand) |
| USB | UVC, VID:PID 04b4:0100 (Cypress bridge) |
| Format | YUYV 4:2:2, 196×256×2 bytes (192 image + 4 metadata rows) |
| Temperature range | -20°C to 120°C (high gain) / -20°C to 400°C (low gain) |

## Status: Alpha

**Working:**
- ✅ Raw frame capture (256×192, 25fps, 16-bit)
- ✅ Live preview with multiple color palettes
- ✅ NUC calibration (manual dark frame subtraction)
- ✅ Auto-load NUC calibration from file on init
- ✅ First-frame skip (first frame after stream open can have corrupted dynamic range)
- ✅ Two-point NUC correction (per-pixel offset + gain)
- ✅ Column FPN removal (vertical stripe correction)
- ✅ Percentile-based AGC (outlier-resistant contrast stretching)
- ✅ Live preview with NUC + FPN correction (use for manual focus tuning)
- ✅ Vendor commands: shutter trigger, palette switching, gain selection
- ✅ Metadata row extraction (4 rows per frame)

**Not yet implemented:**
- ⚠️ Temperature reading (°C) — requires thermometry calibration
- ⚠️ Automatic NUC via mechanical shutter (shutter triggers but auto-reopens too fast to capture)
- ⚠️ Gain switching (0x8020/0x8021) can destabilize the camera — needs more work

**Tested on:** InfiRay T2 Pro (single unit), Arch Linux, x86_64

## License

MIT — see [LICENSE](LICENSE)

## Credits

- [julled/py_InfiRay_T2_Pro](https://github.com/julled/py_InfiRay_T2_Pro) — Original minimal T2 Pro driver, inspired the first-frame skip and auto-load NUC calibration patterns.
- RMHansen (thermal imagery Discord) — Advice on two-point NUC, AGC, and 14-bit thermal data scaling.