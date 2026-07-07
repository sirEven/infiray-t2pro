# InfiRay T2 Pro — Python Driver

A clean, well-tested Python driver for the InfiRay T2 Pro USB thermal camera.
Designed for Linux. Built with TDD (231 tests, all passing).

## Install

```bash
git clone https://github.com/sirEven/infiray-t2pro.git
cd infiray-t2pro
pip install -e .
```

## Quick Start

```python
from infiray_t2pro import T2Pro, Palette
from infiray_t2pro.thermometry import ThermometryLib, calculate_temperature
from infiray_t2pro.snapshot import take_snapshot

# Live preview with temperature overlay
# Press 's' to take a snapshot, 'q' to quit
cam = T2Pro()
cam.start_stream(warmup=10, auto_nuc=120)  # Auto-NUC every 2 min

# Read frames during streaming
frame = cam.read_frame()

# Calculate temperatures from a raw frame
tlib = ThermometryLib()
raw = cam.read_frame_raw()
result = calculate_temperature(tlib, raw)
print(f"Center: {result.center_temp:.1f}°C  Max: {result.max_temp:.1f}°C")

# Take a complete snapshot (PNG + .npy + JSON metadata)
snap = take_snapshot(cam, tlib=tlib, output_dir="snapshots")
print(snap)
# Snapshot: 2026-07-07_21-48-53
#   Center: 27.8°C  Max: 30.7°C  Min: 21.1°C  Avg: 29.2°C
#   FPA: 34.1°C
#   Files: snapshots/2026-07-07_21-48-53

cam.stop_stream()
```

## Session API (Recommended)

For most use cases, `T2ProSession` wraps the entire pipeline — stream, AGC,
denoise, temperature, recording, and snapshots — in a single context manager:

```python
from infiray_t2pro import T2Pro, T2ProSession

cam = T2Pro()
with T2ProSession(cam) as session:
    for _ in range(100):
        frame = session.read_frame()       # Decoded 192×256 float32
        display = session.process(frame)    # AGC + denoise + palette → BGR image
        result = session.read_temperature() # °C temperatures
        print(f"Center: {result.center_temp:.1f}°C")

    # Snapshot: PNG + .npy + JSON
    snap = session.snapshot()

    # Recording: raw frames + temps to disk
    session.start_recording()
    for _ in range(200):
        session.read_frame()
    metadata = session.stop_recording()

# Stream auto-closes on exit. Recording auto-stops if active.
```

Key features:
- Auto-loads `ThermometryLib` and creates `AgcAutoRange`
- `process(frame)` applies FPN correction, denoise, AGC, palette, and upscale
- `read_temperature()` reads raw frame + calculates temps in one call
- `start_recording()` / `stop_recording()` manages `ThermalRecorder`
- `snapshot()` saves complete deliverable (PNG + .npy + JSON)
- `trigger_nuc()`, `reset_agc()`, `set_denoise()`, `set_palette()`, `set_scale()`
- Recording auto-stops on context exit

## Streaming Mode

The driver supports continuous streaming with context manager or explicit start/stop:

```python
# Context manager (auto-cleanup)
with cam.stream(warmup=10, auto_nuc=120) as s:
    for _ in range(100):
        frame = s.read_frame()

# Or manual start/stop
cam.start_stream(warmup=10, auto_nuc=120)
try:
    frame = cam.read_frame()
finally:
    cam.stop_stream()
```

**Auto-NUC:** `auto_nuc=120` triggers a shutter calibration every 120 seconds
during streaming. Essential for unattended drone operations. Set `auto_nuc=None`
(the default) to disable.

## Temperature Calculation

Bundled `libthermometry.so` (official InfiRay Xtherm Linux SDK V6.15) provides
accurate temperature calculation from raw 14-bit thermal data. Ships for both
x86_64 (dev machines) and aarch64 (Raspberry Pi 4/5).

```python
from infiray_t2pro.thermometry import ThermometryLib, calculate_temperature

tlib = ThermometryLib()  # Auto-detects platform
raw = cam.read_frame_raw()
result = calculate_temperature(tlib, raw)

# TemperatureResult fields:
# center_temp, max_temp, max_x, max_y,
# min_temp, min_x, min_y, avg_temp,
# fpa_temp, correction, reflection_temp,
# ambient_temp, humidity, emissivity, distance, shutter_temp_c
```

**Important:** T2 Pro metadata offsets differ from the SDK defaults. The driver
patches `shutter_temp` at `flat[547]` and `core_temp` at `flat[601]` before
calling the C functions. `thermometrySearch` must use `height=192` (sensor only),
not 196 (total rows).

## Snapshot Mode

```python
from infiray_t2pro.snapshot import take_snapshot

# During streaming — saves PNG + .npy + JSON
snap = take_snapshot(cam, tlib=tlib, output_dir="snapshots")

# Or with pre-calculated temperatures
snap = take_snapshot(cam, temp_result=result, output_dir="snapshots")

# Each snapshot creates a timestamped directory:
# snapshots/2026-07-07_21-48-53/
# ├── thermal_2026-07-07_21-48-53.png   # Rendered image with temp overlay
# ├── raw_2026-07-07_21-48-53.npy       # Raw frame data for re-processing
# └── metadata_2026-07-07_21-48-53.json # Full temperature + parameter metadata
```

JSON metadata includes: center/max/min/avg/FPA temperatures, emissivity, distance,
humidity, correction, reflection temp, ambient temp, file references.

## Thermal Recording

Record raw frames and temperature data to disk for post-flight analysis:

```python
from infiray_t2pro.recording import ThermalRecorder

# Context manager — auto-starts/stops recording
with ThermalRecorder(cam, tlib=tlib, output_dir="recordings") as rec:
    for _ in range(100):
        rec.record_frame()  # Raw frame + temperature calculation
    # recorder.json written on exit

# Fast mode — raw frames only, no temperature calculation
with ThermalRecorder(cam, output_dir="recordings") as rec:
    for _ in range(500):
        rec.record_frame_fast()  # Maximum frame rate

# Each session creates a directory:
# recordings/2026-07-07_22-30-00/
# ├── frames/000001.npy  — Raw YUYV frame data
# ├── frames/000002.npy
# ├── ...
# ├── temps.jsonl        — One JSON line per frame with temperature data
# └── recording.json      — Session metadata (fps, duration, frame count)
```

`record_frame()` calculates temperatures via tlib (slower). `record_frame_fast()`
saves only the raw frame (faster, calculate temps offline later).

## Image Processing

### Smooth AGC (Adaptive Gain Control)

Per-frame `cv2.normalize(NORM_MINMAX)` amplifies noise at distance — each frame
stretches its own min/max independently, causing pixelation and flicker. The
driver uses `AgcAutoRange` instead: temporal adaptation at 8% per frame.

```python
from infiray_t2pro.processing import AgcAutoRange

agc = AgcAutoRange(low_percentile=0.5, high_percentile=99.5, adapt_speed=0.08)
normalized = agc.update(frame)  # Adapts range smoothly over time
```

### Bilateral Filter Denoising

Edge-preserving spatial noise reduction for cleaner thermal images:

```python
from infiray_t2pro.processing import denoise_thermal

denoised = denoise_thermal(frame, spatial_sigma=1.5, range_sigma=15.0)
```

### Full Processing Pipeline

```python
from infiray_t2pro.processing import AgcAutoRange, correct_column_fpn, denoise_thermal
from infiray_t2pro.palettes import apply_palette, Palette

agc = AgcAutoRange(adapt_speed=0.08)

frame = cam.read_frame()
corrected = correct_column_fpn(frame)  # Remove vertical stripes
denoised = denoise_thermal(corrected)   # Edge-preserving denoise
rendered = apply_palette(denoised, Palette.INFERNO, scale=5, agc=agc)
```

## Palettes

11 built-in palettes:

| Palette | Description |
|---------|-------------|
| `INFERNO` | Perceptually uniform, dark-to-bright (default) |
| `JET` | Classic rainbow thermal |
| `TURBO` | Improved rainbow, better perceptual uniformity |
| `HOT` | Black-red-yellow-white |
| `CIVIDIS` | Color-vision deficiency friendly |
| `VIRIDIS` | Perceptually uniform, blue-green-yellow |
| `PLASMA` | Purple-pink-yellow |
| `MAGMA` | Dark purple-orange-yellow |
| `WINTER` | Blue-green |
| `WHITE_HOT` | Grayscale, bright = hot |
| `BLACK_HOT` | Grayscale, bright = cold |

Default upscale: 5× (1280×960 from 256×192 sensor).

## Architecture

```
infiray_t2pro/
├── __init__.py        — Public API
├── commands.py        — Vendor command enum + packing logic
├── decode.py          — Raw YUYV frame decoding (16-bit thermal extraction)
├── processing.py      — AgcAutoRange, bilateral denoise, column FPN correction
├── palettes.py        — 11 color palettes + apply_palette with AGC support
├── thermometry.py     — libthermometry.so wrapper, temperature calculation
├── snapshot.py        — take_snapshot() — PNG + .npy + JSON deliverables
├── recording.py       — ThermalRecorder — raw frames + temps to disk
├── session.py         — T2ProSession — high-level pipeline context manager
└── camera.py          — T2Pro class, VideoBackend, streaming, auto-NUC
```

`VideoBackend` abstract class enables testing without hardware.
`V4L2Backend` is the default real implementation.

## Live Preview Controls

`examples/live_preview_temp.py` — full-featured live viewer:

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Take snapshot (PNG + .npy + JSON) |
| `c` | Cycle color palette (11 palettes) |
| `n` | Trigger NUC calibration (cover lens first!) |
| `a` | Toggle AGC mode (smooth vs per-frame) |
| `d` | Toggle bilateral denoise |
| `r` | Reset AGC range |
| `+`/`-` | Zoom in/out (upscale factor) |

## Examples

| Example | Description |
|---------|-------------|
| `live_preview_temp.py` | Full live viewer: temp overlay, AGC, denoise, snapshots, recording |
| `capture_single.py` | Capture and save one frame |
| `live_preview.py` | Basic live thermal video feed |
| `nuc_calibration.py` | NUC calibration workflow |
| `extract_metadata.py` | Capture with metadata row extraction |
| `two_point_nuc.py` | Two-point NUC for high-quality images |
| `live_preview_nuc.py` | Live preview with NUC + FPN correction |

## Tests

```bash
pytest tests/ -v
```

189 tests covering: command packing, palette rendering (11 palettes, 5× upscale),
frame decoding, streaming, error hierarchy, frame validation, NUC calibration,
two-point NUC, AGC (linear, percentile, temporal), column FPN correction,
bilateral denoising, thermometry (temperature calculation, metadata offsets,
T2 Pro patches), auto-NUC, snapshot mode, and camera logic (with mock hardware).

## Requirements

- Linux with UVC support (uvcvideo kernel module)
- OpenCV (`pip install opencv-python`)
- NumPy
- v4l-utils (`v4l2-ctl` command)

## Camera Specs

| Spec | Value |
|------|-------|
| Sensor | 256×192 thermal (16-bit) |
| Frame rate | 25 fps |
| Focus | Manual (rotate lens enclosure by hand) |
| USB | UVC, VID:PID 04b4:0100 (Cypress bridge) |
| Format | YUYV 4:2:2, 196×256×2 bytes (192 image + 4 metadata rows) |
| Temperature range | −20°C to 120°C (high gain) / −20°C to 400°C (low gain) |
| Thermometry | Bundled libthermometry.so (x86_64 + aarch64) |

## Status

**Working:**
- ✅ Raw frame capture (256×192, 25fps, 16-bit)
- ✅ Live preview with 11 color palettes
- ✅ Temperature calculation (°C) from bundled libthermometry.so
- ✅ Smooth temporal AGC — eliminates pixelation/flicker at distance
- ✅ Bilateral filter denoising — edge-preserving noise reduction
- ✅ 5× LANCZOS4 upscale (1280×960 display)
- ✅ Auto-NUC — periodic shutter calibration during streaming
- ✅ Snapshot mode — PNG + .npy + JSON deliverables
- ✅ Thermal recording — raw frames + temperature stream to disk
- ✅ NUC calibration (manual dark frame subtraction)
- ✅ Two-point NUC correction (per-pixel offset + gain)
- ✅ Column FPN removal (vertical stripe correction)
- ✅ Streaming mode with context manager
- ✅ Metadata row extraction and parameter parsing

**Not yet implemented:**
- ⬜ Thermal video recording (raw frames + temp stream to disk)
- ⬜ RPi deployment test (aarch64 lib works on actual Pi)
- ⬜ MAVLink/ArduPilot telemetry bridge for drone integration

**Tested on:** InfiRay T2 Pro (single unit), Arch Linux, x86_64

## License

MIT — see [LICENSE](LICENSE)

## Credits

- [julled/py_InfiRay_T2_Pro](https://github.com/julled/py_InfiRay_T2_Pro) — Original minimal T2 Pro driver, inspired the first-frame skip and auto-load NUC calibration patterns.
- RMHansen ([thermal imagery Discord](https://discord.gg/hBTStM9R9)) — Advice on two-point NUC, AGC, and 14-bit thermal data scaling.