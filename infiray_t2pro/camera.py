"""InfiRay T2 Pro thermal camera driver.

Clean architecture: the T2Pro class uses an injectable VideoBackend so all
logic can be tested without hardware. The default V4L2Backend wraps OpenCV
and v4l2-ctl for real camera access.
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
    """

    def __init__(self, camera_id: int = 0, backend: Optional[VideoBackend] = None):
        self.camera_id = camera_id
        self._backend = backend if backend is not None else V4L2Backend(camera_id)
        self.nuc_calib: Optional[np.ndarray] = None
        self.palette = Palette.INFERNO

    def capture_raw(self, n_frames: int = 1, trash: int = 5) -> np.ndarray:
        """Capture raw 16-bit thermal frames without NUC correction."""
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
        """Capture a thermal frame with optional NUC correction."""
        frame = self.capture_raw(n_frames=n_frames)
        if apply_nuc and self.nuc_calib is not None:
            frame = frame - self.nuc_calib.astype(np.float32)
        return frame

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

    def live_preview(self, use_first_frame: bool = False):
        """Show a live thermal preview window."""
        cv2.namedWindow("InfiRay T2 Pro", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("InfiRay T2 Pro", 1024, 768)

        palettes = [Palette.INFERNO, Palette.JET, Palette.TURBO, Palette.HOT]
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