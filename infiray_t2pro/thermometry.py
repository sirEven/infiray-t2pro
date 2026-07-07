"""InfiRay T2 Pro thermometry — temperature calculation from raw thermal data.

Wraps the vendor's libthermometry.so via ctypes to convert raw 14-bit pixel
values to temperatures in °C.

The library ships in two architectures alongside this module:
  libs/x86_64/libthermometry.so   — Linux x86_64 (dev machines, MacBook)
  libs/aarch64/libthermometry.so  — Linux aarch64 (Raspberry Pi 4/5)

Auto-detection: ThermometryLib() with no arguments picks the right one based
on the host platform. An explicit path can be passed to override.

Source: InfiRay Xtherm Linux SDK V6.15 (official, glibc-linked, not Android).

Architecture:
- ThermometryLib: ctypes wrapper. Loads libthermometry.so and calls
  thermometryT4Line() / thermometrySearch() with proper argtypes.
- parse_metadata_params(): extracts camera-embedded parameters (correction,
  reflection temp, ambient temp, humidity, emissivity, distance) from the
  4-row metadata section of a raw YUYV frame.
- calculate_temperature(): high-level pipeline — parse metadata, build temp
  table, then compute per-pixel and summary temps.

Fail-fast: if the library is missing or fails to load, clear exceptions are
raised. No silent fallback to wrong temperatures.
"""

import ctypes
import platform
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .decode import IMAGE_WIDTH, IMAGE_HEIGHT, TOTAL_ROWS, METADATA_ROWS, extract_metadata


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CAMERA_LENS_6_8 = 68    # 6.8mm lens
CAMERA_LENS_13 = 130    # 13mm lens (default for T2 Pro)

RANGE_MODE_120 = 120    # -20°C to 120°C
RANGE_MODE_400 = 400    # -20°C to 400°C

# Temperature table size: maps 14-bit pixel values (0–16383) to °C
TEMP_TABLE_SIZE = 16384

# For T2 Pro (256 width): amountPixels = 256 * (4-3) = 256
# User area offset within fourLinePara: amountPixels + 127 = 383
_AMOUNT_PIXELS = IMAGE_WIDTH * (METADATA_ROWS - 3)  # 256 * 1 = 256 for T2 Pro
_USER_AREA_OFFSET = _AMOUNT_PIXELS + 127  # 383


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ThermometryLibNotFoundError(OSError):
    """Raised when libthermometry.so cannot be found or loaded."""
    pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MetadataParams:
    """Parameters extracted from the 4-row metadata section of a raw frame.

    All temperatures in °C. Distance in meters. Humidity and emissivity
    are 0–1 fractions.
    """
    correction: float
    reflection_temp: float
    ambient_temp: float
    humidity: float
    emissivity: float
    distance: int
    shutter_temp_c: float  # Shutter temperature in °C (from raw uint16)
    core_temp_c: float     # Core/shell temperature in °C (from raw uint16)


@dataclass
class TemperatureResult:
    """Result of temperature calculation from a single frame.

    All temperatures in °C.
    """
    center_temp: float    # Temperature at frame center
    max_temp: float       # Maximum temperature in frame
    max_x: int            # X coordinate of max temp pixel
    max_y: int            # Y coordinate of max temp pixel
    min_temp: float       # Minimum temperature in frame
    min_x: int            # X coordinate of min temp pixel
    min_y: int            # Y coordinate of min temp pixel
    avg_temp: float       # Average temperature across frame
    point1_temp: float    # Predefined point 1 temperature
    point2_temp: float    # Predefined point 2 temperature
    point3_temp: float    # Predefined point 3 temperature
    fpa_temp: float       # FPA/chipset temperature
    correction: float     # Correction value used
    reflection_temp: float
    ambient_temp: float
    humidity: float
    emissivity: float
    distance: int
    shutter_temp_c: float


# ---------------------------------------------------------------------------
# Metadata parsing
# ---------------------------------------------------------------------------

def _read_float_from_uint16(arr: np.ndarray, offset: int) -> float:
    """Read a float32 value from two consecutive uint16 slots.

    Args:
        arr: Flat uint16 array view of metadata.
        offset: Starting index (the float occupies arr[offset] and arr[offset+1]).

    Returns:
        The float value.
    """
    raw = np.array([arr[offset], arr[offset + 1]], dtype=np.uint16)
    return raw.view(np.float32)[0]


def parse_metadata_params(metadata: np.ndarray) -> MetadataParams:
    """Parse user parameters from the 4-row metadata section.

    The T2 Pro embeds correction, reflection temp, ambient temp, humidity,
    emissivity, and distance in Row 1 at offset 127+ (matching the SDK layout).
    Shutter and core temperatures are at Row 2, Cols 35 and 89 (T2 Pro specific).

    Args:
        metadata: (4, 256, 2) uint8 array from extract_metadata().

    Returns:
        MetadataParams with all extracted values.
    """
    # Flatten metadata to uint16 view
    flat_uint16 = metadata.view(np.uint16).reshape(METADATA_ROWS * IMAGE_WIDTH)

    # Read shutter temperature from T2 Pro specific offset (Row 2, Col 35)
    # Convert: value/10.0 - 273.15 → °C
    _T2PRO_SHUT_FLAT = 2 * IMAGE_WIDTH + 35
    _T2PRO_CORE_FLAT = 2 * IMAGE_WIDTH + 89
    shut_temper_raw = int(flat_uint16[_T2PRO_SHUT_FLAT])
    shutter_temp_c = shut_temper_raw / 10.0 - 273.15 if shut_temper_raw != 0 else 0.0

    # Read core/shell temperature from T2 Pro specific offset (Row 2, Col 89)
    core_temper_raw = int(flat_uint16[_T2PRO_CORE_FLAT])
    core_temp_c = core_temper_raw / 10.0 - 273.15 if core_temper_raw != 0 else 0.0

    # Read user parameters starting at _USER_AREA_OFFSET (Row 1, Col 127)
    offset = _USER_AREA_OFFSET
    correction = float(_read_float_from_uint16(flat_uint16, offset))
    offset += 2
    reflection_temp = float(_read_float_from_uint16(flat_uint16, offset))
    offset += 2
    ambient_temp = float(_read_float_from_uint16(flat_uint16, offset))
    offset += 2
    humidity = float(_read_float_from_uint16(flat_uint16, offset))
    offset += 2
    emissivity = float(_read_float_from_uint16(flat_uint16, offset))
    offset += 2
    distance = int(flat_uint16[offset])

    return MetadataParams(
        correction=correction,
        reflection_temp=reflection_temp,
        ambient_temp=ambient_temp,
        humidity=humidity,
        emissivity=emissivity,
        distance=distance,
        shutter_temp_c=shutter_temp_c,
        core_temp_c=core_temp_c,
    )


# ---------------------------------------------------------------------------
# ThermometryLib: ctypes wrapper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

_LIBS_DIR = Path(__file__).parent / "libs"

def _default_lib_path() -> str:
    """Return the path to the bundled libthermometry.so for the current platform."""
    machine = platform.machine().lower()
    # Normalize common aliases
    if machine in ("x86_64", "amd64"):
        subdir = "x86_64"
    elif machine in ("aarch64", "arm64"):
        subdir = "aarch64"
    else:
        raise ThermometryLibNotFoundError(
            f"Unsupported architecture '{machine}' — need x86_64 or aarch64"
        )

    lib_path = _LIBS_DIR / subdir / "libthermometry.so"
    if not lib_path.exists():
        raise ThermometryLibNotFoundError(
            f"libthermometry.so not found at {lib_path}. "
            f"Expected bundled library for {subdir}."
        )
    return str(lib_path)


class ThermometryLib:
    """Wrapper around libthermometry.so for temperature calculation.

    Loads the shared library and provides typed Python methods that call
    the C functions with proper ctypes argtypes.

    Usage:
        # Auto-detect platform:
        tlib = ThermometryLib()
        # Or explicit path:
        tlib = ThermometryLib("/path/to/libthermometry.so")
        result = calculate_temperature(tlib, raw_frame, ...)
    """

    def __init__(self, lib_path: Optional[str] = None):
        """Load libthermometry.so.

        Args:
            lib_path: Absolute path to libthermometry.so. If None, auto-detects
                      the correct bundled library for the current platform.

        Raises:
            ThermometryLibNotFoundError: If the library cannot be found or loaded.
        """
        if lib_path is None:
            lib_path = _default_lib_path()

        try:
            self._lib = ctypes.cdll.LoadLibrary(lib_path)
        except OSError as e:
            raise ThermometryLibNotFoundError(
                f"Cannot load libthermometry.so from {lib_path}: {e}"
            ) from e

        # Set up function signatures
        self._setup_thermometry_t4_line()
        self._setup_thermometry_search()
        self._setup_thermometry_search_cmm()

    def _setup_thermometry_t4_line(self):
        """Configure argtypes for thermometryT4Line."""
        fn = self._lib.thermometryT4Line
        fn.argtypes = [
            ctypes.c_int,                        # width
            ctypes.c_int,                        # height
            ctypes.POINTER(ctypes.c_float),       # temperatureTable (output)
            ctypes.POINTER(ctypes.c_ushort),      # fourLinePara
            ctypes.POINTER(ctypes.c_float),       # floatFpaTmp (output)
            ctypes.POINTER(ctypes.c_float),       # correction (in/out)
            ctypes.POINTER(ctypes.c_float),       # Refltmp (in/out)
            ctypes.POINTER(ctypes.c_float),       # Airtmp (in/out)
            ctypes.POINTER(ctypes.c_float),       # humi (in/out)
            ctypes.POINTER(ctypes.c_float),       # emiss (in/out)
            ctypes.POINTER(ctypes.c_ushort),      # distance (in/out)
            ctypes.c_int,                        # cameraLens
            ctypes.c_float,                      # shutterFix
            ctypes.c_int,                        # rangeMode
        ]
        fn.restype = None

    def _setup_thermometry_search(self):
        """Configure argtypes for thermometrySearch."""
        fn = self._lib.thermometrySearch
        fn.argtypes = [
            ctypes.c_int,                        # width
            ctypes.c_int,                        # height
            ctypes.POINTER(ctypes.c_float),       # temperatureTable
            ctypes.POINTER(ctypes.c_ushort),      # orgData
            ctypes.POINTER(ctypes.c_float),       # temperatureData (output)
            ctypes.c_int,                        # rangeMode
            ctypes.c_int,                        # outputMode
        ]
        fn.restype = None

    def _setup_thermometry_search_cmm(self):
        """Configure argtypes for thermometrySearchCMM."""
        fn = self._lib.thermometrySearchCMM
        fn.argtypes = [
            ctypes.c_int,                        # width
            ctypes.c_int,                        # height
            ctypes.POINTER(ctypes.c_float),       # temperatureTable
            ctypes.POINTER(ctypes.c_ushort),      # fourLinePara
            ctypes.POINTER(ctypes.c_float),       # temperatureData (output)
            ctypes.c_int,                        # rangeMode
        ]
        fn.restype = None

    def thermometry_t4_line(
        self,
        width: int,
        height: int,
        temp_table: np.ndarray,
        four_line_para: np.ndarray,
        correction: float,
        reflection_temp: float,
        ambient_temp: float,
        humidity: float,
        emissivity: float,
        distance: int,
        camera_lens: int = CAMERA_LENS_13,
        shutter_fix: float = 0.0,
        range_mode: int = RANGE_MODE_120,
    ) -> float:
        """Call thermometryT4Line to build the temperature lookup table.

        This fills the temp_table array with °C values indexed by 14-bit
        pixel value. Also returns the FPA temperature via the output params.

        Args:
            width: Image width (256 for T2 Pro).
            height: Image height including metadata rows (196 for T2 Pro).
            temp_table: float32 array of size 16384 (output, filled by lib).
            four_line_para: uint16 array of metadata (4 rows × 256 cols).
            correction: Temperature correction value (typically -3.0 to 3.0).
            reflection_temp: Reflection temperature in °C (usually = ambient).
            ambient_temp: Ambient/air temperature in °C.
            humidity: Relative humidity (0.0–1.0).
            emissivity: Surface emissivity (0.0–1.0).
            distance: Distance to target in meters.
            camera_lens: Lens type (68=6.8mm, 130=13mm, default 130).
            shutter_fix: Shutter correction value (default 0.0).
            range_mode: Temperature range (120 or 400).

        Returns:
            FPA (chipset) temperature in °C.
        """
        # Ensure arrays are contiguous and correct dtype
        temp_table = np.ascontiguousarray(temp_table, dtype=np.float32)
        four_line_para = np.ascontiguousarray(four_line_para, dtype=np.uint16)

        # Output parameters
        fpa_temp = ctypes.c_float(0.0)
        c_correction = ctypes.c_float(correction)
        c_refl = ctypes.c_float(reflection_temp)
        c_air = ctypes.c_float(ambient_temp)
        c_humi = ctypes.c_float(humidity)
        c_emiss = ctypes.c_float(emissivity)
        c_distance = ctypes.c_ushort(distance)

        self._lib.thermometryT4Line(
            width, height,
            temp_table.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            four_line_para.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort)),
            ctypes.byref(fpa_temp),
            ctypes.byref(c_correction),
            ctypes.byref(c_refl),
            ctypes.byref(c_air),
            ctypes.byref(c_humi),
            ctypes.byref(c_emiss),
            ctypes.byref(c_distance),
            camera_lens,
            shutter_fix,
            range_mode,
        )

        return float(fpa_temp.value)

    def thermometry_search(
        self,
        width: int,
        height: int,
        temp_table: np.ndarray,
        org_data: np.ndarray,
        range_mode: int = RANGE_MODE_120,
        output_mode: int = 4,
    ) -> np.ndarray:
        """Call thermometrySearch to compute temperatures from pixel data.

        Args:
            width: Image width.
            height: Image height (excluding metadata rows).
            temp_table: Temperature lookup table from thermometry_t4_line.
            org_data: uint16 array of raw pixel values (height × width).
            range_mode: Temperature range (120 or 400).
            output_mode: 4 for full frame + 10 stats, 5 for 10 stats only.

        Returns:
            Float32 array of temperature values. For mode 4:
            [0]=center, [1]=max_x, [2]=max_y, [3]=max, [4]=min_x, [5]=min_y,
            [6]=min, [7]=point1, [8]=point2, [9]=avg, then full frame.
        """
        temp_table = np.ascontiguousarray(temp_table, dtype=np.float32)
        org_data = np.ascontiguousarray(org_data, dtype=np.uint16)

        # For mode 4: 10 stats + width*(height-4) temperature values
        result_size = 10 + width * (height - 4) if output_mode == 4 else 11
        temp_data = np.zeros(result_size, dtype=np.float32)

        self._lib.thermometrySearch(
            width, height,
            temp_table.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            org_data.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort)),
            temp_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            range_mode,
            output_mode,
        )

        return temp_data

    def thermometry_search_cmm(
        self,
        width: int,
        height: int,
        temp_table: np.ndarray,
        four_line_para: np.ndarray,
        range_mode: int = RANGE_MODE_120,
    ) -> np.ndarray:
        """Call thermometrySearchCMM for mode-5 temperature calculation.

        This is the alternative to thermometry_search for 8005 mode.
        Uses fourLinePara instead of full orgData.

        Args:
            width: Image width.
            height: Image height (including metadata rows).
            temp_table: Temperature lookup table.
            four_line_para: uint16 metadata array.
            range_mode: Temperature range (120 or 400).

        Returns:
            Float32 array of 11 temperature values:
            [0]=center, [1]=max_x, [2]=max_y, [3]=max, [4]=min_x, [5]=min_y,
            [6]=min, [7]=point1, [8]=point2, [9]=point3, [10]=avg.
        """
        temp_table = np.ascontiguousarray(temp_table, dtype=np.float32)
        four_line_para = np.ascontiguousarray(four_line_para, dtype=np.uint16)

        temp_data = np.zeros(11, dtype=np.float32)

        self._lib.thermometrySearchCMM(
            width, height,
            temp_table.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            four_line_para.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort)),
            temp_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            range_mode,
        )

        return temp_data


# ---------------------------------------------------------------------------
# High-level temperature calculation
# ---------------------------------------------------------------------------

def calculate_temperature(
    tlib: ThermometryLib,
    raw_frame: np.ndarray,
    camera_lens: int = CAMERA_LENS_13,
    range_mode: int = RANGE_MODE_120,
    shutter_fix: float = 0.0,
) -> TemperatureResult:
    """Calculate temperature from a raw YUYV frame using libthermometry.

    This is the main entry point for temperature calculation. It:
    1. Extracts the 4 metadata rows from the raw frame.
    2. Parses camera-embedded parameters (emissivity, distance, etc.).
    3. Calls thermometryT4Line to build the temperature lookup table.
    4. Calls thermometrySearch to compute center/max/min/avg temperatures.

    Args:
        tlib: Loaded ThermometryLib instance.
        raw_frame: (196, 256, 2) uint8 array from camera.read_frame().
        camera_lens: Lens type constant (CAMERA_LENS_13 or CAMERA_LENS_6_8).
        range_mode: Temperature range (RANGE_MODE_120 or RANGE_MODE_400).
        shutter_fix: Shutter correction value (default 0.0).

    Returns:
        TemperatureResult with all temperature values and metadata params.
    """
    # 1. Extract metadata rows
    metadata = extract_metadata(raw_frame)

    # 2. Parse embedded parameters
    params = parse_metadata_params(metadata)

    # 3. Convert metadata to uint16 fourLinePara for the C function
    four_line_para = metadata.view(np.uint16).reshape(METADATA_ROWS * IMAGE_WIDTH).copy()
    four_line_para = np.ascontiguousarray(four_line_para, dtype=np.uint16)

    # 3b. Fix shutter/core temp offsets for T2 Pro.
    # The SDK expects shut_temper at fourLinePara[amountPixels+1] and core_temper
    # at fourLinePara[amountPixels+2] (amountPixels=256 for 256-wide cameras).
    # The T2 Pro stores these at Row 2, Col 35 (shutter) and Col 89 (core)
    # in the metadata. Patch the expected offsets so the C function reads them.
    _AMOUNT_PIXELS_256 = 256
    _SHUT_OFFSET = _AMOUNT_PIXELS_256 + 1  # 257
    _CORE_OFFSET = _AMOUNT_PIXELS_256 + 2  # 258
    _T2PRO_SHUT_FLAT = 2 * 256 + 35  # Row 2, Col 35 → flat index 547
    _T2PRO_CORE_FLAT = 2 * 256 + 89  # Row 2, Col 89 → flat index 601

    if four_line_para[_T2PRO_SHUT_FLAT] != 0:
        four_line_para[_SHUT_OFFSET] = four_line_para[_T2PRO_SHUT_FLAT]
    if four_line_para[_T2PRO_CORE_FLAT] != 0:
        four_line_para[_CORE_OFFSET] = four_line_para[_T2PRO_CORE_FLAT]

    # 4. Build temperature table
    temp_table = np.zeros(TEMP_TABLE_SIZE, dtype=np.float32)

    # Height for thermometryT4Line includes metadata rows (196 total)
    fpa_temp = tlib.thermometry_t4_line(
        width=IMAGE_WIDTH,
        height=TOTAL_ROWS,
        temp_table=temp_table,
        four_line_para=four_line_para,
        correction=params.correction,
        reflection_temp=params.reflection_temp,
        ambient_temp=params.ambient_temp,
        humidity=params.humidity,
        emissivity=params.emissivity,
        distance=params.distance,
        camera_lens=camera_lens,
        shutter_fix=shutter_fix,
        range_mode=range_mode,
    )

    # 5. Get the full raw frame as uint16 (including metadata rows).
    # thermometrySearch needs the full orgData buffer but height must be
    # IMAGE_HEIGHT (192, thermal only) — NOT TOTAL_ROWS (196). Including
    # metadata rows in the height parameter causes the C function to produce
    # incorrect stats (all -273°C).
    org_data = raw_frame.view(np.uint16).reshape(TOTAL_ROWS * IMAGE_WIDTH).copy()
    org_data = np.ascontiguousarray(org_data, dtype=np.uint16)

    # 6. Compute temperatures using mode 4 (full frame + stats)
    temp_data = tlib.thermometry_search(
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        temp_table=temp_table,
        org_data=org_data,
        range_mode=range_mode,
        output_mode=4,
    )

    # 7. Build result from temp_data
    return TemperatureResult(
        center_temp=float(temp_data[0]),
        max_temp=float(temp_data[3]),
        max_x=int(temp_data[1]),
        max_y=int(temp_data[2]),
        min_temp=float(temp_data[6]),
        min_x=int(temp_data[4]),
        min_y=int(temp_data[5]),
        avg_temp=float(temp_data[9]),
        point1_temp=float(temp_data[7]),
        point2_temp=float(temp_data[8]),
        point3_temp=0.0,  # mode 4 only has 2 extra points
        fpa_temp=fpa_temp,
        correction=params.correction,
        reflection_temp=params.reflection_temp,
        ambient_temp=params.ambient_temp,
        humidity=params.humidity,
        emissivity=params.emissivity,
        distance=params.distance,
        shutter_temp_c=params.shutter_temp_c,
    )