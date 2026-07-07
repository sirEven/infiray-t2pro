"""Thermal recording — save raw frames and temperature data to disk.

Records a thermal session as a directory containing:
  - frames/000001.npy   — raw 16-bit thermal frames (196×256×2 YUYV)
  - temps.jsonl         — one JSON line per recorded frame with temperature data
  - recording.json      — session metadata (start time, duration, frame count, etc.)

Usage during streaming:
    from infiray_t2pro.camera import T2Pro
    from infiray_t2pro.thermometry import ThermometryLib, calculate_temperature
    from infiray_t2pro.recording import ThermalRecorder

    cam = T2Pro()
    cam.start_stream(warmup=10, auto_nuc=120)
    tlib = ThermometryLib()

    with ThermalRecorder(cam, tlib, output_dir="recordings/session1") as rec:
        while some_condition:
            rec.record_frame()

    print(f"Recorded {rec.frame_count} frames to {rec.output_dir}")

The recorder reads two frames per call (one processed via read_frame,
one raw via read_frame_raw for temperature). For maximum frame rate,
use record_frame_fast() which only saves the raw frame without temperature
calculation.

Files:
  - recording.json: start_time, end_time, frame_count, fps, sensor_size
  - temps.jsonl: one JSON object per line with TemperatureResult fields
  - frames/: sequentially numbered .npy files (raw YUYV data)
"""

import json
import os
import time
import numpy as np
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from .camera import T2Pro
from .thermometry import ThermometryLib, TemperatureResult, calculate_temperature


class ThermalRecorder:
    """Records raw thermal frames and temperature data to disk.

    Designed for unattended drone operations — start recording, fly the
    mission, stop recording. All data is flushed to disk on every frame
    so nothing is lost on disconnect.

    Fail-fast: if disk writes fail, the exception propagates immediately.
    No buffering, no retry.
    """

    def __init__(
        self,
        cam: T2Pro,
        tlib: Optional[ThermometryLib] = None,
        output_dir: str = "recordings",
        session_name: Optional[str] = None,
    ):
        """Initialize the recorder.

        Args:
            cam: T2Pro camera instance (must be streaming).
            tlib: ThermometryLib for temperature calculation.
                  If None, record_frame() skips temperature recording.
            output_dir: Base directory for recordings.
            session_name: Session name (defaults to timestamp).
        """
        self.cam = cam
        self.tlib = tlib
        self.output_dir = output_dir
        self.session_name = session_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_path = ""
        self._recording = False
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self.frame_count: int = 0
        self._temps_file = None

    @property
    def is_recording(self) -> bool:
        """Whether the recorder is actively recording."""
        return self._recording

    @property
    def fps(self) -> float:
        """Average FPS during recording. 0.0 if not started."""
        if self._start_time == 0 or self.frame_count == 0:
            return 0.0
        elapsed = (self._end_time or time.monotonic()) - self._start_time
        if elapsed == 0:
            return 0.0
        return self.frame_count / elapsed

    def __enter__(self):
        """Start recording (context manager)."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop recording (context manager)."""
        self.stop()

    def start(self) -> str:
        """Start recording. Creates session directory and metadata file.

        Returns:
            Path to the session directory.

        Raises:
            RuntimeError: If already recording.
        """
        if self._recording:
            raise RuntimeError("Already recording — call stop() first")

        self.session_path = os.path.join(self.output_dir, self.session_name)
        os.makedirs(self.session_path, exist_ok=True)
        os.makedirs(os.path.join(self.session_path, "frames"), exist_ok=True)

        self._temps_file = open(
            os.path.join(self.session_path, "temps.jsonl"), "w"
        )
        self._start_time = time.monotonic()
        self._end_time = 0.0
        self.frame_count = 0
        self._recording = True

        return self.session_path

    def record_frame(self) -> Optional[TemperatureResult]:
        """Record a single frame with temperature calculation.

        Reads one raw frame from the camera, saves it as .npy, and
        calculates temperatures via libthermometry. Returns the
        TemperatureResult or None if tlib is not set.

        Returns:
            TemperatureResult if tlib is set, None otherwise.

        Raises:
            RuntimeError: If not recording.
            FrameReadError: If the frame read fails.
        """
        if not self._recording:
            raise RuntimeError("Not recording — call start() first")

        raw = self.cam.read_frame_raw()
        frame_num = self.frame_count + 1
        frame_path = os.path.join(
            self.session_path, "frames", f"{frame_num:06d}.npy"
        )
        np.save(frame_path, raw)

        self.frame_count = frame_num

        # Temperature calculation
        temp_result = None
        if self.tlib is not None:
            temp_result = calculate_temperature(self.tlib, raw)
            line = json.dumps({
                "frame": frame_num,
                "time": time.time(),
                **{k: round(v, 4) if isinstance(v, float) else v
                   for k, v in asdict(temp_result).items()},
            })
            self._temps_file.write(line + "\n")
            self._temps_file.flush()

        self._end_time = time.monotonic()
        return temp_result

    def record_frame_fast(self) -> None:
        """Record a raw frame without temperature calculation.

        Faster than record_frame() because it skips the thermometry
        calculation. Use for maximum frame rate when you only need the
        raw thermal data and will calculate temperatures later.

        Raises:
            RuntimeError: If not recording.
            FrameReadError: If the frame read fails.
        """
        if not self._recording:
            raise RuntimeError("Not recording — call start() first")

        raw = self.cam.read_frame_raw()
        frame_num = self.frame_count + 1
        frame_path = os.path.join(
            self.session_path, "frames", f"{frame_num:06d}.npy"
        )
        np.save(frame_path, raw)

        self.frame_count = frame_num
        self._end_time = time.monotonic()

    def stop(self) -> dict:
        """Stop recording and write session metadata.

        Returns:
            Dict with session metadata (path, frame_count, fps, duration, etc.)

        Raises:
            RuntimeError: If not recording.
        """
        if not self._recording:
            raise RuntimeError("Not recording — call start() first")

        self._recording = False
        end_time = time.monotonic()

        if self._temps_file:
            self._temps_file.close()
            self._temps_file = None

        duration = end_time - self._start_time
        metadata = {
            "session": self.session_name,
            "start_time": datetime.fromtimestamp(self._start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
            "duration_s": round(duration, 3),
            "frame_count": self.frame_count,
            "fps": round(self.frame_count / duration, 2) if duration > 0 else 0.0,
            "sensor_size": "256x192",
            "frame_format": "YUYV 4:2:2, 196x256x2 bytes per frame",
            "has_temperatures": self.tlib is not None,
        }

        meta_path = os.path.join(self.session_path, "recording.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata