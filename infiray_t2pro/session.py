"""High-level session context manager for the T2 Pro thermal camera.

T2ProSession wraps the common pipeline: open stream → configure AGC/denoise
→ read frames → process → display/record → close stream. It eliminates the
boilerplate of manually managing ThermometryLib, AgcAutoRange, dark references,
and recording state.

Usage:
    from infiray_t2pro import T2Pro, T2ProSession

    cam = T2Pro()
    with T2ProSession(cam) as session:
        for _ in range(100):
            frame = session.read_frame()        # Decoded 192×256 float32
            display = session.process(frame)     # AGC + denoise + palette
            result = session.read_temperature()  # TemperatureResult
            print(f"Center: {result.center_temp:.1f}°C")

    # With recording:
    with T2ProSession(cam) as session:
        session.start_recording()
        for _ in range(200):
            frame = session.read_frame()
            display = session.process(frame)
        metadata = session.stop_recording()

    # Snapshot:
    with T2ProSession(cam) as session:
        result = session.read_temperature()
        snap = session.snapshot()
"""

import os
import numpy as np
from typing import Optional

from .camera import T2Pro, StreamClosedError
from .palettes import Palette, apply_palette
from .processing import AgcAutoRange, denoise_thermal
from .thermometry import ThermometryLib, TemperatureResult, calculate_temperature
from .snapshot import take_snapshot, SnapshotResult
from .recording import ThermalRecorder


class T2ProSession:
    """High-level context manager for the T2 Pro thermal camera pipeline.

    Wraps stream lifecycle, AGC, denoising, temperature calculation,
    recording, and snapshots into a single cohesive interface.

    Args:
        cam: T2Pro camera instance.
        warmup: Number of warmup frames to discard on stream start.
        auto_nuc: Auto-NUC interval in seconds (None = disabled).
        tlib: ThermometryLib instance. If None, auto-loads the bundled library.
        agc: AgcAutoRange instance. If None, creates one with defaults.
        dark: Dark reference for NUC correction. If None, uses cam.nuc_calib.
        palette: Default color palette for process() and snapshot().
        scale: Default upscale factor for display.
        output_dir: Base directory for snapshots and recordings.
    """

    def __init__(
        self,
        cam: T2Pro,
        warmup: int = 10,
        auto_nuc: Optional[float] = 120,
        tlib: Optional[ThermometryLib] = None,
        agc: Optional[AgcAutoRange] = None,
        dark: Optional[np.ndarray] = None,
        palette: Palette = Palette.INFERNO,
        scale: int = 5,
        output_dir: str = "output",
    ):
        self.cam = cam
        self.warmup = warmup
        self.auto_nuc = auto_nuc
        self.tlib = tlib or ThermometryLib()
        self.agc = agc or AgcAutoRange(
            low_percentile=0.5,
            high_percentile=99.5,
            adapt_speed=0.08,
            min_range=80.0,
        )
        self.dark = dark
        self.palette = palette
        self.scale = scale
        self.output_dir = output_dir

        self._recorder: Optional[ThermalRecorder] = None
        self._last_temp: Optional[TemperatureResult] = None
        self._last_temp_time: float = 0.0
        self._use_denoise: bool = True

    # --- Context manager ---

    def __enter__(self):
        """Start streaming and return self."""
        self.cam.start_stream(warmup=self.warmup, auto_nuc=self.auto_nuc)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop streaming and recording (if active)."""
        if self._recorder is not None:
            try:
                self._recorder.stop()
            except Exception:
                pass
            self._recorder = None
        self.cam.stop_stream()
        return False

    # --- Properties ---

    @property
    def is_streaming(self) -> bool:
        """Whether the stream is currently active."""
        return self.cam.is_streaming

    @property
    def frame_count(self) -> int:
        """Number of frames read in this session."""
        return self.cam.frame_count

    @property
    def is_recording(self) -> bool:
        """Whether a recording is in progress."""
        return self._recorder is not None

    @property
    def fps(self) -> float:
        """Average FPS of the current recording (0.0 if not recording)."""
        if self._recorder is not None:
            return self._recorder.fps
        return 0.0

    # --- Frame reading ---

    def read_frame(self) -> np.ndarray:
        """Read a decoded thermal frame (192×256 float32).

        Delegates to cam.read_frame(). Auto-NUC is handled by the stream.

        Returns:
            2D float32 array of thermal values.

        Raises:
            StreamClosedError: If the stream is not open.
            FrameReadError: If the frame read fails.
        """
        return self.cam.read_frame()

    def read_frame_raw(self) -> np.ndarray:
        """Read a raw frame (196×256×2 uint8) including metadata rows.

        Use this when you need the raw data for temperature calculation.

        Returns:
            Raw YUYV frame array.

        Raises:
            StreamClosedError: If the stream is not open.
            FrameReadError: If the frame read fails.
        """
        return self.cam.read_frame_raw()

    # --- Processing ---

    def process(
        self,
        frame: np.ndarray,
        palette: Optional[Palette] = None,
        scale: Optional[int] = None,
        use_agc: bool = True,
        use_denoise: Optional[bool] = None,
    ) -> np.ndarray:
        """Process a decoded frame into a display-ready BGR image.

        Applies column FPN correction (if dark reference loaded),
        bilateral denoising (if enabled), AGC contrast adaptation,
        palette coloring, and upscaling.

        Args:
            frame: Decoded 192×256 float32 thermal frame.
            palette: Color palette. Defaults to session palette.
            scale: Upscale factor. Defaults to session scale.
            use_agc: Whether to apply temporal AGC. Default True.
            use_denoise: Whether to apply bilateral denoise.
                         None uses the session default (True).

        Returns:
            BGR image array suitable for cv2.imshow().
        """
        if use_denoise is None:
            use_denoise = self._use_denoise

        processed = frame
        if self.dark is not None:
            from .processing import correct_column_fpn
            processed = correct_column_fpn(processed)
        if use_denoise:
            processed = denoise_thermal(processed)

        return apply_palette(
            processed,
            palette or self.palette,
            scale=scale or self.scale,
            agc=self.agc if use_agc else None,
        )

    # --- Temperature ---

    def read_temperature(
        self,
        camera_lens: int = 130,
        range_mode: int = 120,
        shutter_fix: float = 0.0,
    ) -> TemperatureResult:
        """Read a frame and calculate temperatures.

        Reads a raw frame via read_frame_raw() and passes it to
        calculate_temperature(). This consumes one frame from the stream.

        Args:
            camera_lens: Lens type (130=13mm default, 68=6.8mm).
            range_mode: Temperature range (120 or 400).
            shutter_fix: Shutter correction value.

        Returns:
            TemperatureResult with all temperature values.

        Raises:
            StreamClosedError: If the stream is not open.
            FrameReadError: If the frame read fails.
            ThermometryLibNotFoundError: If tlib is not loaded.
        """
        raw = self.cam.read_frame_raw()
        result = calculate_temperature(
            self.tlib, raw,
            camera_lens=camera_lens,
            range_mode=range_mode,
            shutter_fix=shutter_fix,
        )
        self._last_temp = result
        self._last_temp_time = __import__('time').time()
        return result

    # --- Recording ---

    def start_recording(self, session_name: Optional[str] = None) -> str:
        """Start recording raw frames to disk.

        Args:
            session_name: Name for the recording session. Defaults to timestamp.

        Returns:
            Path to the session directory.

        Raises:
            RuntimeError: If already recording.
        """
        if self._recorder is not None:
            raise RuntimeError("Already recording — call stop_recording() first")

        self._recorder = ThermalRecorder(
            self.cam, tlib=self.tlib,
            output_dir=os.path.join(self.output_dir, "recordings"),
            session_name=session_name,
        )
        return self._recorder.start()

    def stop_recording(self) -> dict:
        """Stop recording and save session metadata.

        Returns:
            Dict with session metadata (path, frame_count, fps, duration).

        Raises:
            RuntimeError: If not recording.
        """
        if self._recorder is None:
            raise RuntimeError("Not recording — call start_recording() first")

        metadata = self._recorder.stop()
        self._recorder = None
        return metadata

    # --- Snapshot ---

    def snapshot(self, palette: Optional[Palette] = None) -> SnapshotResult:
        """Take a complete thermal snapshot.

        Saves PNG + .npy + JSON to the output directory. Uses the last
        temperature reading if available, otherwise reads a new frame.

        Args:
            palette: Color palette for the PNG. Defaults to session palette.

        Returns:
            SnapshotResult with file paths and temperature summary.
        """
        # Use last temperature result if recent, otherwise calculate
        if self._last_temp is not None:
            temp_result = self._last_temp
        else:
            temp_result = self.read_temperature()

        return take_snapshot(
            self.cam,
            temp_result=temp_result,
            output_dir=os.path.join(self.output_dir, "snapshots"),
            palette=palette or self.palette,
            scale=self.scale,
            agc=self.agc,
            dark=self.dark,
            apply_denoise=self._use_denoise,
        )

    # --- Controls ---

    def trigger_nuc(self):
        """Trigger NUC calibration (mechanical shutter). Cover lens first!"""
        self.cam.trigger_shutter()

    def reset_agc(self):
        """Reset the AGC range. Use if the image looks washed out."""
        self.agc.reset()

    def set_denoise(self, enabled: bool):
        """Enable or disable bilateral denoise."""
        self._use_denoise = enabled

    def set_palette(self, palette: Palette):
        """Change the default color palette."""
        self.palette = palette

    def set_scale(self, scale: int):
        """Change the default upscale factor."""
        self.scale = scale