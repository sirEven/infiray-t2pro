# RPi Stream Architecture

Software running on the Raspberry Pi payload — reads camera frames, processes them,
streams to ground, and records a rolling buffer to SD.

**Current scope:** T2 Pro thermal camera only. Generalization to other cameras
(USB RGB photo cameras, etc.) comes later when we have a second input device.

---

## Context

The RPi is mounted on the drone as a separate payload system. It is NOT the flight
controller. It talks to cameras via USB and talks to the ground via Wi-Fi or radio
link. The drone itself (flight controller, ESCs, piloting cam) is a separate system.

The current `infiray_t2pro` library is a camera *driver* — it reads frames, decodes
raw thermal data, applies NUC correction, renders palettes, and triggers vendor
commands. It does NOT stream, record, or do detection. That's what we're building
here: the layer that sits on top of the driver and makes the payload useful in the
air.

---

## Data Flow

```
  Camera (USB)                    RPi Payload                     Ground Station
  ────────────                    ─────────────                   ──────────────

  T2 Pro ──YUYV──→ [read frames] ──→ [process*] ──→ [stream to ground] ──→ viewer
                                        │
                                        └──→ [ring buffer on SD]
                                              (last 3 min, circular)

  * process = camera-specific pipeline stage:
      T2 Pro:  decode → NUC → FPN → palette → RGB output
      RGB cam: grab frame → maybe downscale → RGB output
      (one camera at a time, initially)
```

Three concerns, one input:

1. **Process** — camera-specific frame transformation
2. **Stream** — push processed frames to ground station in real time
3. **Record** — keep last N minutes on SD, promote to permanent on demand

---

## Components

### 1. Frame Source (reads from camera)

Wraps the `infiray_t2pro.T2Pro` driver (or later, any camera backend).

Responsibilities:
- Open/close the camera stream
- Yield frames at native fps (or a configurable rate)
- Expose frame metadata: resolution, fps, dtype, timestamp

For T2 Pro: uses `T2Pro.capture_raw()` internally but keeps the stream open
continuously instead of open/capture/close per frame. The current driver opens
and closes per capture — we need a long-running stream mode.

For future cameras: same interface, different backend. OpenCV `VideoCapture`
covers most USB cameras out of the box.

```python
# Minimal interface sketch (NOT implementation, just shape)
class FrameSource:
    def start(self): ...          # open stream, warm up
    def read(self) -> Frame: ...   # returns (image: np.ndarray, meta: dict)
    def stop(self): ...            # clean shutdown
```

### 2. Processing Pipeline (camera-specific transforms)

Takes raw frames from the source, applies camera-specific processing, outputs
RGB frames suitable for streaming and display.

For T2 Pro the pipeline is:
```
raw YUYV → decode 16-bit → NUC subtract → column FPN → AGC → palette → BGR
```

For an RGB camera the pipeline might be:
```
raw frame → (maybe downscale) → (maybe denoise) → BGR
```

The pipeline is a list of callables that each take a frame and return a frame.
This makes it easy to insert new stages (detection overlay, HUD, etc.) without
changing the core loop.

```python
# Pipeline = list of (name, callable) stages
# Frame carries both the thermal data and the rendered version
# Stages can operate on either

pipeline = [
    ("decode", decode_yuyv_to_thermal),
    ("nuc", apply_nuc_correction),
    ("fpn", correct_column_fpn),
    ("agc", agc_percentile),
    ("palette", lambda f: apply_palette(f, Palette.INFERNO)),
]
```

### 3. Ring Buffer (last N minutes on SD)

Writes video segments to SD card in a circular fashion. When the buffer is
full, oldest segments are deleted. On a trigger (pilot signal, detection event,
or manual), current segments are promoted to permanent storage.

Design choices:
- **Segment-based, not frame-based.** Write 10-second video segments as files.
  This avoids thrashing the SD card with tiny random writes and makes the
  buffer easy to manage (just delete oldest files).
- **Format:** compressed video segments (H.264 or MJPEG containers). Raw 256×192
  at 25fps is ~1.5 MB/s. Even 3 minutes is only ~270 MB raw, less compressed.
  For higher-res cameras later, compression becomes essential.
- **Circular directory:** e.g. `/mnt/sd/buffer/segment_001234.mp4`. Oldest files
  deleted when total size exceeds limit. Simple, robust, no fancy data structures.
- **Promote on trigger:** copy current segments from buffer to a permanent
  directory with timestamp. This is how the pilot says "save that" — the last
  3 minutes get preserved instead of overwritten.
- **SD card wear:** segment writes are sequential, not random. 10-second segments
  at ~1 MB each means ~6 MB/min, 360 MB/hour. Even cheap SD cards handle this
  for hundreds of hours. The circular delete/reuse is also sequential since
  we delete whole files.

```
/mnt/sd/
  buffer/          # circular ring buffer (last 3 min)
    seg_0001.mp4
    seg_0002.mp4
    ...
  saves/           # promoted permanent saves
    2026-07-07_143052/
      seg_0001.mp4
      seg_0002.mp4
      ...
```

### 4. Stream to Ground (real-time video to operator)

Pushes processed (RGB) frames from the pipeline to a ground station in real time.

Options considered:
- **MJPEG over HTTP** — simplest, any browser can view it. High bandwidth, low
  latency. Fine for thermal (256×192). Not great for 1080p later.
- **RTSP** — standard for IP cameras, works with VLC, FFmpeg, OBS. More complex
  to implement but widely supported.
- **WebRTC** — lowest latency, but heavy dependency. Overkill for now.
- **Raw TCP socket** — custom protocol, maximum control, but we build everything.

**Recommendation: start with MJPEG over HTTP.** It's a few dozen lines of code,
works everywhere, and is good enough for 256×192 thermal at 25fps. We can add
RTSP later when we need lower bandwidth or standard drone tooling integration.

The stream runs in its own thread, serving frames from a shared queue that the
pipeline writes to.

---

## Concurrency Model

The RPi 4 has 4 cores. The frame loop, processing, recording, and streaming can
run in parallel using threads (not processes — the GIL is fine here because all
the heavy work is in NumPy/OpenCV which release the GIL).

```
Thread 1: Frame source (reads from camera, pushes to queue)
Thread 2: Processing pipeline (reads from queue, applies stages, pushes to outputs)
Thread 3: Ring buffer writer (reads processed frames, writes segments)
Thread 4: Stream server (serves frames to ground station via HTTP)
```

Frame source → [queue] → pipeline → [queue] → ring buffer
                                 ↓
                            [queue] → stream

The queues are bounded. If the ring buffer or stream falls behind, frames are
dropped. The stream and buffer must not block the pipeline.

---

## Open Questions

- **Stream protocol:** MJPEG is simple. RTSP is standard. Start with MJPEG, add
  RTSP when a real ground station app exists.
- **Ground station:** What receives the stream? Browser? Custom app? QGroundControl?
  This affects the protocol choice but not the RPi side much.
- **Trigger mechanism:** How does the pilot say "save the last 3 minutes"?
  - GPIO pin from flight controller?
  - Network signal from ground station?
  - Keyboard/button on RPi GPIO?
  - For now: probably a simple HTTP endpoint or MQTT signal.
- **Detection integration:** When we add thermal detection (e.g. fawn detection),
  it plugs into the pipeline as a stage. It can also trigger the ring buffer
  promote. But detection is a future feature, not v0.
- **Dual camera:** If we mount thermal + RGB simultaneously, we run two frame
  sources into two pipelines sharing a ring buffer and stream. But this is
  "later" — design for it by keeping the components decoupled, don't implement
  it now.
- **Hardware encoding:** RPi 4 has V4L2 M2M H.264 encoder. For thermal it's
  overkill (256×192 compresses fine in software). For 1080p RGB later, we'll
  want it. Not needed now.

---

## Implementation Order

1. **Stream mode in T2Pro** — modify the driver to support continuous streaming
   (keep VideoCapture open, yield frames in a loop) instead of open/capture/close.
2. **Ring buffer** — segment-based circular recorder. This is the most
   camera-agnostic piece and immediately useful (never lose footage).
3. **MJPEG stream server** — HTTP endpoint serving live video. Again very simple
   and camera-agnostic.
4. **Pipeline orchestrator** — tie frame source, processing, buffer, and stream
   together with queues and threads.
5. **Trigger mechanism** — HTTP endpoint or signal to promote ring buffer
   segments to permanent save.

Steps 1-4 can be tested on the bench with the T2 Pro plugged into the RPi.
Step 5 requires thinking about how the pilot interacts with the payload.

---

## What This Is NOT

- NOT a ground station app (that's a separate project)
- NOT a flight controller integration (separate system, separate concern)
- NOT an autopilot or navigation system
- NOT a multi-camera switcher (yet — single camera only for v0)
- NOT detection or AI (future pipeline stage, not core architecture)

This is the software that runs on the RPi payload and makes the camera feed
useful in the air: processed, streamed, and always recorded.