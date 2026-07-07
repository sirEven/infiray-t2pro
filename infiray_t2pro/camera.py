"""InfiRay T2 Pro thermal camera driver.

Clean architecture: the T2Pro class uses an injectable VideoBackend so all
logic can be tested without hardware. The default V4L2Backend wraps OpenCV
and v4l2-ctl for real camera access.

Supports two modes:
- Snapshot mode: capture() / capture_raw() — open, grab, close per call.
- Streaming mode: start_stream() / read_frame() / stop_stream() — keep stream
  open and read frames continuously.
"""

import cv2
import numpy as np
import subprocess
import time
import os
from typing import Optional, Tuple
from datetime import datetime

from .commands import Command
from .palettes import Palette, apply_palette
from .decode import decode_frame, extract_metadata, IMAGE_HEIGHT, TOTAL_ROWS, IMAGE_WIDTH
from .thermometry import (
    ThermometryLib, ThermometryLibNotFoundError,
    calculate_temperature as _calc_temp,
    CAMERA_LENS_13, RANGE_MODE_120,
)


class StreamClosedError(RuntimeError):
    """Raised when trying to read a frame from a stream that is not open."""
    pass


class StreamOpenError(RuntimeError):
    """Raised when start_stream() fails — camera not found, open fails, or warmup fails.

    The original error is chained and accessible via __cause__.
    After a StreamOpenError, the stream is guaranteed to be off (is_streaming=False).
    The caller can retry by calling start_stream() again.
    """
    pass


class FrameReadError(RuntimeError):
    """Raised when reading a frame from the backend fails mid-stream.

    The original backend error is chained and accessible via __cause__.
    After a FrameReadError, the stream is no longer active — the caller
    must stop_stream() then start_stream() to reconnect.
    """
    pass


class VideoBackend:
    """Abstract video backend for camera access. Override for testing."""

    def open(self):
        """Open the video stream."""
        raise NotImplementedError

    def read_raw(self) -> np.ndarray:
        """Read a raw frame (196×256×2, uint8)."""
        raise NotImplementedError

    def set_zoom(self, value: int):
        """Set V4L2 zoom_absolute (vendor command channel)."""
        raise NotImplementedError

    def close(self):
        """Close the video stream."""
        raise NotImplementedError


class V4L2Backend(VideoBackend):
    """Real camera backend using OpenCV + v4l2-ctl."""

    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self._cap = None

    def open(self):
        self._cap = cv2.VideoCapture(self.camera_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open /dev/video{self.camera_id}")
        self._cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

    def read_raw(self) -> np.ndarray:
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Stream not open")
        ret, raw = self._cap.read()
        if not ret:
            raise RuntimeError("No frame captured")
        return raw

    def set_zoom(self, value: int):
        subprocess.run(
            ["v4l2-ctl", "-d", f"/dev/video{self.camera_id}",
             "-c", f"zoom_absolute={int(value)}"],
            capture_output=True, timeout=5,
        )

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class T2Pro:
    """InfiRay T2 Pro thermal camera driver.

    Args:
        camera_id: Video device index (0 for /dev/video0).
        backend: Video backend for hardware abstraction. If None, uses V4L2Backend.
        nuc_calib_path: Path to a .npy NUC calibration file. If the file exists,
                        it is loaded automatically on init. If it doesn't exist,
                        calibration starts empty (call calibrate_nuc_manual() to create one).
    """

    def __init__(self, camera_id: int = 0, backend: Optional[VideoBackend] = None,
                 nuc_calib_path: str = "t2pro_nuc_calib.npy"):
        self.camera_id = camera_id
        self._backend = backend if backend is not None else V4L2Backend(camera_id)
        self.nuc_calib: Optional[np.ndarray] = None
        self.palette = Palette.INFERNO
        self._frame_count = 0
        self._is_streaming = False
        self._stream_frame_count = 0
        self._tlib: Optional[ThermometryLib] = None

        # Auto-load NUC calibration if file exists
        if os.path.exists(nuc_calib_path):
            self.nuc_calib = np.load(nuc_calib_path)

    # --- Streaming mode ---

    @property
    def is_streaming(self) -> bool:
        """Whether the stream is currently open for continuous reading."""
        return self._is_streaming

    @property
    def frame_count(self) -> int:
        """Number of frames read since stream was started (excluding warmup)."""
        return self._stream_frame_count

    def start_stream(self, warmup: int = 5) -> None:
        """Open the stream and warm up the camera.

        After calling this, use read_frame() to grab frames continuously.
        Call stop_stream() when done.

        Args:
            warmup: Number of frames to discard after opening. The T2 Pro's
                    first frames can have unstable dynamic range.

        Raises:
            RuntimeError: If already streaming.
            StreamOpenError: If the backend fails to open or warmup frames fail.
        """
        if self._is_streaming:
            raise RuntimeError("Already streaming. Call stop_stream() first.")
        try:
            self._backend.open()
        except Exception as e:
            raise StreamOpenError(f"Failed to open stream: {e}") from e
        self._is_streaming = True
        self._stream_frame_count = 0
        # Discard warmup frames
        try:
            for _ in range(warmup):
                self._backend.read_raw()
        except Exception as e:
            # Warmup failed — close the backend and clean up state
            self._is_streaming = False
            try:
                self._backend.close()
            except Exception:
                pass  # best-effort close
            raise StreamOpenError(f"Failed during warmup: {e}") from e

    def read_frame(self, apply_nuc: bool = True) -> np.ndarray:
        """Read a single frame from the open stream.

        Args:
            apply_nuc: Whether to apply NUC correction (if calibration is loaded).
                       The first frame after start_stream is always raw (no NUC)
                       because it may have unstable dynamic range.

        Returns:
            2D float32 array (192×256) of thermal values.

        Raises:
            StreamClosedError: If stream is not open.
            FrameReadError: If the backend fails or the frame is corrupt
                           (all zeros or uniform value).
        """
        if not self._is_streaming:
            raise StreamClosedError("Stream is not open. Call start_stream() first.")
        try:
            raw = self._backend.read_raw()
        except Exception as e:
            # Stream is broken — mark it dead so caller knows to reconnect.
            self._is_streaming = False
            raise FrameReadError(f"Failed to read frame: {e}") from e
        frame = decode_frame(raw)

        # Validate frame: detect corrupt reads
        if np.all(frame == 0):
            self._is_streaming = False
            raise FrameReadError("Frame is all zeros — dead sensor or corrupt read")
        if np.std(frame) == 0:
            self._is_streaming = False
            raise FrameReadError(f"Frame is uniform (all pixels = {frame.flat[0]:.0f}) — stuck sensor")

        self._stream_frame_count += 1

        if apply_nuc and self.nuc_calib is not None and self._stream_frame_count > 1:
            frame = frame - self.nuc_calib.astype(np.float32)
        return frame

    def read_frame_raw(self) -> np.ndarray:
        """Read a raw frame from the open stream (196×256×2, uint8).

        Use this when you need the raw frame data including metadata rows,
        e.g. for temperature calculation.

        Raises:
            StreamClosedError: If stream is not open.
            FrameReadError: If the backend fails.
        """
        if not self._is_streaming:
            raise StreamClosedError("Stream is not open. Call start_stream() first.")
        try:
            raw = self._backend.read_raw()
        except Exception as e:
            self._is_streaming = False
            raise FrameReadError(f"Failed to read raw frame: {e}") from e
        self._stream_frame_count += 1
        return raw

    def calculate_temperature(
        self,
        tlib: Optional[ThermometryLib] = None,
        camera_lens: int = CAMERA_LENS_13,
        range_mode: int = RANGE_MODE_120,
        shutter_fix: float = 0.0,
    ):
        """Calculate temperature from the next frame in the stream.

        Reads a raw frame, extracts metadata, and converts pixel values to °C
        using libthermometry.so. Requires an active stream (start_stream first).

        Args:
            tlib: ThermometryLib instance. If None, auto-loads the bundled
                  library for the current platform.
            camera_lens: Lens type (CAMERA_LENS_13=130 default, CAMERA_LENS_6_8=68).
            range_mode: Temperature range (RANGE_MODE_120 or RANGE_MODE_400).
            shutter_fix: Shutter correction value (default 0.0).

        Returns:
            TemperatureResult with center/max/min/avg temps and metadata.

        Raises:
            StreamClosedError: If stream is not open.
            FrameReadError: If the backend fails or frame is corrupt.
            ThermometryLibNotFoundError: If libthermometry.so can't be loaded.
        """
        if not self._is_streaming:
            raise StreamClosedError("Stream is not open. Call start_stream() first.")

        if tlib is None:
            if self._tlib is None:
                self._tlib = ThermometryLib()
            tlib = self._tlib

        raw = self.read_frame_raw()
        return _calc_temp(
            tlib, raw,
            camera_lens=camera_lens,
            range_mode=range_mode,
            shutter_fix=shutter_fix,
        )

    def stop_stream(self) -> None:
        """Close the stream. Safe to call multiple times."""
        if self._is_streaming:
            self._backend.close()
            self._is_streaming = False

    def stream(self, warmup: int = 5):
        """Context manager for streaming mode.

        Usage:
            cam = T2Pro()
            with cam.stream() as s:
                for _ in range(100):
                    frame = s.read_frame()
        """
        self.start_stream(warmup=warmup)
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop_stream()

    # --- Snapshot mode (backward compatible) ---

    def capture_raw(self, n_frames: int = 1, trash: int = 5) -> np.ndarray:
        """Capture raw 16-bit thermal frames without NUC correction.

        Opens the stream, captures, and closes. For continuous reading, use
        start_stream() / read_frame() / stop_stream() instead.
        """
        self._backend.open()
        for _ in range(trash):
            self._backend.read_raw()
        frames = []
        for _ in range(n_frames):
            raw = self._backend.read_raw()
            frames.append(decode_frame(raw))
            time.sleep(0.02)
        self._backend.close()
        return np.mean(frames, axis=0)

    def capture(self, n_frames: int = 5, apply_nuc: bool = True) -> np.ndarray:
        """Capture a thermal frame with optional NUC correction.

        The first capture after init skips NUC correction because the first
        frame after opening the stream can have corrupted dynamic range.

        For continuous reading, use start_stream() / read_frame() / stop_stream().
        """
        frame = self.capture_raw(n_frames=n_frames)
        self._frame_count += 1

        if apply_nuc and self.nuc_calib is not None and self._frame_count > 1:
            frame = frame - self.nuc_calib.astype(np.float32)
        return frame

    # --- Commands ---

    def trigger_shutter(self):
        """Trigger the mechanical shutter for auto-calibration."""
        self._backend.set_zoom(int(Command.SHUTTER_CLOSE))
        time.sleep(0.5)

    def send_command(self, command: Command):
        """Send a vendor command via V4L2 zoom_absolute."""
        self._backend.set_zoom(int(command))
        time.sleep(0.3)

    def reset(self):
        """Reset the camera to default mode."""
        self._backend.set_zoom(int(Command.DEFAULT))
        time.sleep(1)

    # --- Calibration ---

    def calibrate_nuc_manual(self, n_frames: int = 10) -> np.ndarray:
        """Capture a NUC calibration frame (cover lens first!)."""
        dark = self.capture_raw(n_frames=n_frames, trash=5)
        self.nuc_calib = dark.astype(np.uint16)
        return dark

    def load_nuc_calibration(self, path: str):
        """Load NUC calibration from a .npy file."""
        self.nuc_calib = np.load(path)

    def save_nuc_calibration(self, path: str = "t2pro_nuc_calib.npy"):
        """Save NUC calibration to a .npy file."""
        np.save(path, self.nuc_calib)

    # --- Metadata ---

    def capture_with_metadata(self) -> Tuple[np.ndarray, np.ndarray]:
        """Capture a frame and extract metadata rows."""
        self._backend.open()
        for _ in range(5):
            self._backend.read_raw()
        raw = self._backend.read_raw()
        self._backend.close()
        metadata = extract_metadata(raw)
        image = decode_frame(raw)
        return image, metadata

    # --- Rendering ---

    def render(self, thermal: np.ndarray, palette: Optional[Palette] = None) -> np.ndarray:
        """Render thermal data as a colored BGR image."""
        return apply_palette(thermal, palette or self.palette)

    def save(self, thermal: np.ndarray, path: str, palette: Optional[Palette] = None):
        """Save thermal data as a PNG image."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        image = self.render(thermal, palette)
        cv2.imwrite(path, image)

    def save_raw(self, thermal: np.ndarray, path: str):
        """Save raw 16-bit thermal data as a .npy file."""
        np.save(path, thermal.astype(np.uint16))

    # --- Live preview ---

    def live_preview(self, use_first_frame: bool = False):
        """Show a live thermal preview window."""
        cv2.namedWindow("InfiRay T2 Pro", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("InfiRay T2 Pro", 1024, 768)

        palettes = list(Palette)
        cmap_idx = 0
        frame_count = 0

        while True:
            frame = self.capture_raw(n_frames=1, trash=1)
            display = apply_palette(frame, palettes[cmap_idx])

            info = f"std={frame.std():.0f} min={frame.min():.0f} max={frame.max():.0f} | #{frame_count}"
            cv2.putText(display, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, "q=quit  s=save  c=colormap", (10, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("InfiRay T2 Pro", display)
            frame_count += 1

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs("thermal_captures", exist_ok=True)
                cv2.imwrite(f"thermal_captures/live_{ts}.png", display)
            elif key == ord('c'):
                cmap_idx = (cmap_idx + 1) % len(palettes)

        cv2.destroyAllWindows()