"""Typed NumPy bridge for the native C++20 Splat Constructor.

Build ``splat_constructor_bridge.dll`` with ``scripts/build_splat_constructor.ps1``
before calling :func:`construct_splat_state`.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from enum import IntEnum
import os
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray


class DepthSemantics(IntEnum):
    """Depth interpretation shared with the C++ ``ConstructionFrame``."""

    RELATIVE_INVERSE = 0
    METRIC_CAMERA_Z = 1
    PSEUDO_INVERSE_FROM_SEGMENTATION = 2


class DepthUnits(IntEnum):
    UNITLESS = 0
    METERS = 1


@dataclass(frozen=True)
class ConstructionConfig:
    sample_stride: int = 2
    near_depth: float = 1.0
    far_depth: float = 4.0
    mask_threshold: int = 1
    opacity: float = 0.9
    scale_multiplier: float = 0.75
    thickness_multiplier: float = 0.25
    depth_edge_threshold: float = 0.1
    edge_scale_multiplier: float = 0.5
    minimum_reconstruction_weight: float = 0.0


@dataclass(frozen=True)
class ConstructionFrame:
    frame_id: str
    timestamp_ns: int
    source_id: str
    rgb: NDArray[np.uint8]
    depth: NDArray[np.float32]
    foreground_mask: NDArray[np.uint8]
    intrinsics: tuple[float, float, float, float]
    depth_semantics: DepthSemantics = DepthSemantics.RELATIVE_INVERSE
    depth_units: DepthUnits = DepthUnits.UNITLESS
    depth_validity: NDArray[np.float32] | None = None
    foreground_weight: NDArray[np.float32] | None = None
    camera_to_world: NDArray[np.float32] = field(
        default_factory=lambda: np.eye(4, dtype=np.float32)
    )


@dataclass(frozen=True)
class SplatState:
    frame_id: str
    timestamp_ns: int
    source_id: str
    intrinsics: tuple[float, float, float, float]
    depth_semantics: DepthSemantics
    depth_units: DepthUnits
    camera_to_world: NDArray[np.float32]
    positions: NDArray[np.float32]
    sh_dc: NDArray[np.float32]
    opacity_logits: NDArray[np.float32]
    log_scales: NDArray[np.float32]
    rotations: NDArray[np.float32]
    reconstruction_weights: NDArray[np.float32]
    local_ids: NDArray[np.uint64]
    source_pixels: NDArray[np.uint32]


class _CConfig(ctypes.Structure):
    _fields_ = [
        ("sample_stride", ctypes.c_size_t),
        ("near_depth", ctypes.c_float),
        ("far_depth", ctypes.c_float),
        ("mask_threshold", ctypes.c_uint8),
        ("opacity", ctypes.c_float),
        ("scale_multiplier", ctypes.c_float),
        ("thickness_multiplier", ctypes.c_float),
        ("depth_edge_threshold", ctypes.c_float),
        ("edge_scale_multiplier", ctypes.c_float),
        ("minimum_reconstruction_weight", ctypes.c_float),
    ]


class _CInput(ctypes.Structure):
    _fields_ = [
        ("frame_id", ctypes.c_char_p),
        ("timestamp_ns", ctypes.c_uint64),
        ("source_id", ctypes.c_char_p),
        ("width", ctypes.c_size_t),
        ("height", ctypes.c_size_t),
        ("rgb", ctypes.POINTER(ctypes.c_uint8)),
        ("depth", ctypes.POINTER(ctypes.c_float)),
        ("foreground_mask", ctypes.POINTER(ctypes.c_uint8)),
        ("depth_validity", ctypes.POINTER(ctypes.c_float)),
        ("foreground_weight", ctypes.POINTER(ctypes.c_float)),
        ("fx", ctypes.c_float),
        ("fy", ctypes.c_float),
        ("cx", ctypes.c_float),
        ("cy", ctypes.c_float),
        ("depth_semantics", ctypes.c_int),
        ("depth_units", ctypes.c_int),
        ("camera_to_world", ctypes.POINTER(ctypes.c_float)),
    ]


_FLOAT_POINTER: Final = ctypes.POINTER(ctypes.c_float)
_UINT32_POINTER: Final = ctypes.POINTER(ctypes.c_uint32)
_UINT64_POINTER: Final = ctypes.POINTER(ctypes.c_uint64)
_UINT8_POINTER: Final = ctypes.POINTER(ctypes.c_uint8)


def default_library_path() -> Path:
    return Path(__file__).resolve().parents[2] / "build" / "splat_constructor" / (
        "splat_constructor_bridge.dll" if os.name == "nt" else "libsplat_constructor_bridge.so"
    )


def _require_array(
    value: NDArray[np.generic], dtype: np.dtype[np.generic], shape: tuple[int, ...], name: str
) -> NDArray[np.generic]:
    array = np.asarray(value)
    if array.dtype != dtype or array.shape != shape:
        raise ValueError(f"{name} must have dtype {dtype} and shape {shape}; got {array.dtype} {array.shape}")
    return np.ascontiguousarray(array)


def _optional_plane(
    value: NDArray[np.float32] | None, height: int, width: int, name: str
) -> NDArray[np.float32] | None:
    if value is None:
        return None
    return _require_array(value, np.dtype(np.float32), (height, width), name).reshape(-1)  # type: ignore[return-value]


def _validate_frame(
    frame: ConstructionFrame,
) -> tuple[NDArray[np.uint8], NDArray[np.float32], NDArray[np.uint8], NDArray[np.float32] | None, NDArray[np.float32] | None, NDArray[np.float32]]:
    if not frame.frame_id or not frame.source_id or frame.timestamp_ns <= 0:
        raise ValueError("frame_id, source_id, and positive timestamp_ns are required")
    rgb = np.asarray(frame.rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be a uint8 H x W x 3 array")
    height, width, _ = rgb.shape
    rgb = np.ascontiguousarray(rgb)
    depth = _require_array(frame.depth, np.dtype(np.float32), (height, width), "depth")
    mask = np.asarray(frame.foreground_mask)
    if mask.dtype == np.bool_:
        mask = mask.astype(np.uint8)
    mask = _require_array(mask, np.dtype(np.uint8), (height, width), "foreground_mask")
    if not np.isin(mask, (0, 1)).all():
        raise ValueError("foreground_mask must contain binary {0, 1} values")
    if not np.isfinite(depth).all() or (depth < 0.0).any():
        raise ValueError("depth must be finite and non-negative")
    if frame.depth_semantics != DepthSemantics.METRIC_CAMERA_Z and (depth > 1.0).any():
        raise ValueError("relative depth must be normalized to [0, 1]")
    if (frame.depth_semantics == DepthSemantics.METRIC_CAMERA_Z) != (
        frame.depth_units == DepthUnits.METERS
    ):
        raise ValueError("metric camera-Z requires meters; relative depth requires unitless values")
    depth_validity = _optional_plane(frame.depth_validity, height, width, "depth_validity")
    foreground_weight = _optional_plane(frame.foreground_weight, height, width, "foreground_weight")
    for name, plane in (("depth_validity", depth_validity), ("foreground_weight", foreground_weight)):
        if plane is not None and (not np.isfinite(plane).all() or (plane < 0.0).any() or (plane > 1.0).any()):
            raise ValueError(f"{name} must be finite in [0, 1]")
    pose = _require_array(frame.camera_to_world, np.dtype(np.float32), (4, 4), "camera_to_world")
    return rgb, depth, mask, depth_validity, foreground_weight, pose


class _NativeLibrary:
    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(
                f"native bridge not found at {path}; run ./scripts/build_splat_constructor.ps1"
            )
        self._library = ctypes.CDLL(str(path))
        library = self._library
        library.splat_construct.argtypes = [ctypes.POINTER(_CInput), ctypes.POINTER(_CConfig)]
        library.splat_construct.restype = ctypes.c_void_p
        library.splat_state_destroy.argtypes = [ctypes.c_void_p]
        library.splat_state_destroy.restype = None
        library.splat_last_error.restype = ctypes.c_char_p
        library.splat_state_size.argtypes = [ctypes.c_void_p]
        library.splat_state_size.restype = ctypes.c_size_t
        for name, pointer in (
            ("splat_state_positions", _FLOAT_POINTER),
            ("splat_state_sh_dc", _FLOAT_POINTER),
            ("splat_state_opacity_logits", _FLOAT_POINTER),
            ("splat_state_log_scales", _FLOAT_POINTER),
            ("splat_state_rotations", _FLOAT_POINTER),
            ("splat_state_reconstruction_weights", _FLOAT_POINTER),
            ("splat_state_local_ids", _UINT64_POINTER),
            ("splat_state_source_pixels", _UINT32_POINTER),
        ):
            getattr(library, name).argtypes = [ctypes.c_void_p]
            getattr(library, name).restype = pointer

    def construct(self, frame: ConstructionFrame, config: ConstructionConfig) -> SplatState:
        rgb, depth, mask, validity, foreground, pose = _validate_frame(frame)
        height, width = depth.shape
        fx, fy, cx, cy = frame.intrinsics
        native_input = _CInput(
            frame.frame_id.encode("utf-8"),
            frame.timestamp_ns,
            frame.source_id.encode("utf-8"),
            width,
            height,
            rgb.ctypes.data_as(_UINT8_POINTER),
            depth.ctypes.data_as(_FLOAT_POINTER),
            mask.ctypes.data_as(_UINT8_POINTER),
            None if validity is None else validity.ctypes.data_as(_FLOAT_POINTER),
            None if foreground is None else foreground.ctypes.data_as(_FLOAT_POINTER),
            fx,
            fy,
            cx,
            cy,
            int(frame.depth_semantics),
            int(frame.depth_units),
            pose.ctypes.data_as(_FLOAT_POINTER),
        )
        native_config = _CConfig(
            config.sample_stride,
            config.near_depth,
            config.far_depth,
            config.mask_threshold,
            config.opacity,
            config.scale_multiplier,
            config.thickness_multiplier,
            config.depth_edge_threshold,
            config.edge_scale_multiplier,
            config.minimum_reconstruction_weight,
        )
        handle = self._library.splat_construct(ctypes.byref(native_input), ctypes.byref(native_config))
        if not handle:
            detail = self._library.splat_last_error()
            message = detail.decode("utf-8", errors="replace") if detail else "unknown native error"
            raise RuntimeError(f"native Splat Constructor failed: {message}")
        try:
            count = self._library.splat_state_size(handle)
            return SplatState(
                frame_id=frame.frame_id,
                timestamp_ns=frame.timestamp_ns,
                source_id=frame.source_id,
                intrinsics=frame.intrinsics,
                depth_semantics=frame.depth_semantics,
                depth_units=frame.depth_units,
                camera_to_world=pose.copy(),
                positions=_copy_native(self._library.splat_state_positions(handle), count * 3, np.float32).reshape(count, 3),
                sh_dc=_copy_native(self._library.splat_state_sh_dc(handle), count * 3, np.float32).reshape(count, 3),
                opacity_logits=_copy_native(self._library.splat_state_opacity_logits(handle), count, np.float32),
                log_scales=_copy_native(self._library.splat_state_log_scales(handle), count * 3, np.float32).reshape(count, 3),
                rotations=_copy_native(self._library.splat_state_rotations(handle), count * 4, np.float32).reshape(count, 4),
                reconstruction_weights=_copy_native(self._library.splat_state_reconstruction_weights(handle), count, np.float32),
                local_ids=_copy_native(self._library.splat_state_local_ids(handle), count, np.uint64),
                source_pixels=_copy_native(self._library.splat_state_source_pixels(handle), count * 2, np.uint32).reshape(count, 2),
            )
        finally:
            self._library.splat_state_destroy(handle)


def _copy_native(pointer: ctypes._Pointer[ctypes._SimpleCData], count: int, dtype: np.dtype[np.generic]) -> NDArray[np.generic]:
    if count == 0:
        return np.empty((0,), dtype=dtype)
    if not pointer:
        raise RuntimeError("native Splat Constructor returned a null attribute buffer")
    return np.ctypeslib.as_array(pointer, shape=(count,)).copy()


def construct_splat_state(
    frame: ConstructionFrame,
    config: ConstructionConfig = ConstructionConfig(),
    *,
    library_path: str | Path | None = None,
) -> SplatState:
    """Construct a copied NumPy ``SplatState`` through the native C++ core."""

    path = Path(library_path) if library_path is not None else default_library_path()
    return _NativeLibrary(path).construct(frame, config)
