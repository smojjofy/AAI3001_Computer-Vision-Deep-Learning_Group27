"""Shadow person-mask evaluation for the live pipeline.

This module runs the research person-mask candidate beside the accepted live
mask. It never changes the published mask, the published splat, the published
pose joints, or the published timings. It writes one JSON record for each
sample. The records go to a separate file, which the caller writes.

The module has three parts:

- :func:`build_research_input` builds the 15-channel research tensor from
  arrays that are already in memory. It calls the same helpers as the replay
  loader, so the live path and the training path use the same arithmetic.
- :class:`ShadowMaskRunner` holds one frozen checkpoint and runs the model.
- :class:`ShadowMaskPipeline` keeps a short rolling depth window, builds a
  provisional background baseline, and returns one record for each frame.

Two limits are important. The provisional baseline is the older end of a short
rolling window. It is not an empty-scene calibration, because the person is
inside that window. Use the records for comparison only. They are not a
deployment result.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import queue
import sys
from collections import deque
from collections.abc import Iterable, Mapping
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import cv2
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from modules.inference.checkpoint import load_checkpoint
from modules.inference.paired_research.data import (
    CHANNEL_ORDER,
    HEIGHT,
    WIDTH,
    _depth_arrays,
    _flow_array,
    _pose_array,
    _stack_channels,
)
from modules.inference.paired_research.model import PairedMaskMotionNet

ROOT = Path(__file__).resolve().parents[3]
APPLICATIONS_ROOT = ROOT / "applications"
STREAM_HANDLER_SRC = ROOT / "modules" / "stream_handler" / "src"
OPTICAL_FLOW_SRC = ROOT / "modules" / "inference" / "optical_flow" / "src"

CHECKPOINT_TASK = "paired_mask_motion_research"
SCHEMA_VERSION = "1"
KIND = "live_shadow_mask"

# Fixed decision rules. Change them only with a new checkpoint.
MASK_THRESHOLD = 0.5
SOURCE_UPSAMPLE_RULE = "bilinear logits, align_corners=False, threshold zero"

# The bounded mask mode permits a correction only inside the baseline mask and
# within a fixed research-pixel radius of it. The model uses a square maximum
# window of size 2 * radius + 1 at 320 x 180, so the shadow restriction must use
# the same square window.
PERMIT_RADIUS_PX = 12
PERMIT_KERNEL_PX = 2 * PERMIT_RADIUS_PX + 1

# The reference search copies the live window rule.
MIN_VALID_FLOW_FRACTION = 0.25

# A provisional background frame with too little usable depth near the image
# border is usually an estimate of the wall behind the person. Drop such frames.
BORDER_WIDTH_PX = 12
BORDER_MIN_USABLE_FRACTION = 0.5

# The provisional baseline keeps a pixel when at least this many window frames
# have usable depth at that pixel. This is the rule of
# ``BackgroundCalibrator.build``.
MIN_PIXEL_VALID_RATIO = 0.5

STATUSES = ("ok", "skipped", "failed")


# --------------------------------------------------------------------------
# Environment helpers
# --------------------------------------------------------------------------


def _add_sys_path(path: Path) -> None:
    """Add ``path`` to ``sys.path`` when it is not there."""
    text = str(path)
    if text not in sys.path:
        sys.path.append(text)


def _import_live_modules() -> tuple[Any, Any, Any]:
    """Import the live flow estimator and the mask selector modules.

    The applications package and the component source trees are not installed.
    This helper adds their roots to ``sys.path`` first. Returns a tuple of
    (DIS flow estimator class, select function, gate function, config class).
    """
    for path in (APPLICATIONS_ROOT, STREAM_HANDLER_SRC, OPTICAL_FLOW_SRC):
        _add_sys_path(path)
    from optical_flow import DISFlowEstimator  # type: ignore[import-not-found]
    from live_pipeline.adapters import DepthThresholdConfig  # type: ignore[import-not-found]
    from live_pipeline.mask_selection import (  # type: ignore[import-not-found]
        select_pose_gradient_mask,
    )
    from live_pipeline.pose_gradient_mask import (  # type: ignore[import-not-found]
        gate_validity_to_joint_depth,
    )

    return DISFlowEstimator, DepthThresholdConfig, select_pose_gradient_mask, gate_validity_to_joint_depth


def _frame_packet(fields: Mapping[str, Any]) -> Any:
    """Build a minimal capture packet for the flow estimator.

    The flow estimator reads ``frame_id``, ``source_id``,
    ``capture_timestamp_ns``, ``timestamp_origin``, and ``rgb`` only.
    """
    modules = None
    try:
        _add_sys_path(STREAM_HANDLER_SRC)
        from stream_handler import FramePacket, FrameQuality, TimestampOrigin  # type: ignore[import-not-found]

        modules = (FramePacket, FrameQuality, TimestampOrigin)
    except Exception:  # pragma: no cover - depends on the environment
        modules = None
    if modules is not None:
        packet_class, quality_class, origin_class = modules
        return packet_class(
            frame_id=int(fields["frame_id"]),
            source_id=str(fields["source_id"]),
            capture_timestamp_ns=int(fields["capture_timestamp_ns"]),
            timestamp_origin=origin_class.DECODE_MONOTONIC,
            rgb=fields["rgb"],
            quality=quality_class(decode_latency_ns=0, frame_age_ns=0, frame_age_available=False),
        )
    return _SimpleFrame(
        frame_id=int(fields["frame_id"]),
        source_id=str(fields["source_id"]),
        capture_timestamp_ns=int(fields["capture_timestamp_ns"]),
        rgb=fields["rgb"],
    )


class _SimpleFrame:
    """A small stand-in for ``FramePacket`` when that package is not present."""

    def __init__(self, *, frame_id: int, source_id: str, capture_timestamp_ns: int, rgb: np.ndarray) -> None:
        self.frame_id = int(frame_id)
        self.source_id = str(source_id)
        self.capture_timestamp_ns = int(capture_timestamp_ns)
        self.timestamp_origin = "decode_monotonic"
        self.rgb = rgb


# --------------------------------------------------------------------------
# Part 1 - the 15-channel research input
# --------------------------------------------------------------------------


def _checked_rgb(name: str, image: np.ndarray) -> np.ndarray:
    """Return an RGB image after a dtype and shape check."""
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8, got {array.dtype}")
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{name} must have shape [H, W, 3], got {array.shape}")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must be at least 1 x 1 pixels")
    return array


def _checked_plane(name: str, plane: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Return a float32 plane after a shape check."""
    array = np.asarray(plane)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array.astype(np.float32)


def _checked_plane3(name: str, plane: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Return a float32 three-dimensional array after a shape check."""
    array = np.asarray(plane)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array.astype(np.float32)


def _kept_depth_is_non_finite(depth: np.ndarray, validity: np.ndarray) -> bool:
    """Return True when a depth sample holds a value that the rule cannot keep.

    The research depth rule keeps a sample when ``depth`` is finite and
    ``validity`` is above 0.5. An infinite depth has no physical meaning in
    metres, and a NaN depth with a validity above 0.5 is a contradiction. This
    function reports both cases.
    """
    if bool(np.isinf(depth).any()):
        return True
    return bool(np.any(np.isnan(depth) & (np.asarray(validity) > 0.5)))


def _rgb_to_research(image: np.ndarray) -> np.ndarray:
    """Resize one RGB uint8 image to 320 x 180 and scale the values to [0, 1]."""
    resized = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def _mask_to_research(mask: np.ndarray) -> np.ndarray:
    """Resize one bool or uint8 mask to 320 x 180 with nearest sampling."""
    small = cv2.resize(mask.astype(np.uint8) * 255, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
    return small.astype(np.float32) / 255.0


def build_research_input(
    *,
    reference_rgb: np.ndarray,
    selected_rgb: np.ndarray,
    reference_depth: np.ndarray,
    reference_validity: np.ndarray,
    selected_depth: np.ndarray,
    selected_validity: np.ndarray,
    flow: np.ndarray,
    flow_valid: np.ndarray,
    reference_joints: list[dict],
    selected_joints: list[dict],
    reference_frame_id: int,
    selected_frame_id: int,
    baseline_mask: np.ndarray,
) -> torch.Tensor:
    """Build the 15-channel research input for one reference and selected pair.

    Both RGB images are RGB uint8 ``[H, W, 3]``. The caller converts a BGR image
    before the call. Depths and validities are ``[H, W]`` in metres, and invalid
    depth holds NaN. ``flow`` is ``[H, W, 2]`` in pixels with the direction
    ``previous_to_current``, which is reference to selected. ``flow_valid`` is
    uint8 or bool ``[H, W]``. ``baseline_mask`` is bool or uint8 ``[H, W]`` at
    the image size. The joint lists use the mask-input record fields of
    ``shared/schemas/learning-replay.md``.

    Returns a float32 tensor of shape ``(15, 180, 320)``. The channel order is
    :data:`CHANNEL_ORDER`. The arithmetic is the arithmetic of the replay
    loader.

    Raises ``ValueError`` on a wrong type, a wrong shape, or a non-finite value.
    """
    reference_image = _checked_rgb("reference_rgb", reference_rgb)
    selected_image = _checked_rgb("selected_rgb", selected_rgb)
    if reference_image.shape != selected_image.shape:
        raise ValueError(
            "reference_rgb and selected_rgb must have the same shape, got "
            f"{reference_image.shape} and {selected_image.shape}"
        )
    source_shape = (int(selected_image.shape[0]), int(selected_image.shape[1]))
    reference_id = int(reference_frame_id)
    selected_id = int(selected_frame_id)
    if reference_id == selected_id:
        raise ValueError(
            f"reference_frame_id and selected_frame_id must differ, both are {selected_id}"
        )

    depth_reference = _checked_plane("reference_depth", reference_depth, source_shape)
    depth_selected = _checked_plane("selected_depth", selected_depth, source_shape)
    validity_reference = _checked_plane("reference_validity", reference_validity, source_shape)
    validity_selected = _checked_plane("selected_validity", selected_validity, source_shape)
    flow_plane = _checked_plane3("flow", flow, (*source_shape, 2))
    flow_valid_plane = _checked_plane("flow_valid", flow_valid, source_shape)

    # A non-finite value is not allowed at a sample that the rule keeps. Invalid
    # depth uses NaN with a validity of zero, and the depth rule turns it into
    # zero. An invalid flow sample uses a finite vector or a NaN vector with a
    # valid flag of zero.
    if _kept_depth_is_non_finite(depth_reference, validity_reference):
        raise ValueError("reference_depth must be finite, with a validity of zero for an invalid sample")
    if _kept_depth_is_non_finite(depth_selected, validity_selected):
        raise ValueError("selected_depth must be finite, with a validity of zero for an invalid sample")
    for name, value in (("reference_validity", validity_reference), ("selected_validity", validity_selected)):
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must have finite values")
    if np.isinf(flow_plane).any():
        raise ValueError("flow must not hold an infinite value")
    if not np.isfinite(flow_valid_plane).all():
        raise ValueError("flow_valid must have finite values")

    mask = np.asarray(baseline_mask)
    if mask.shape != source_shape:
        raise ValueError(f"baseline_mask must have shape {source_shape}, got {mask.shape}")
    if mask.dtype != np.bool_ and mask.dtype != np.uint8:
        raise ValueError(f"baseline_mask must be bool or uint8, got {mask.dtype}")
    if not np.isin(np.unique(mask), (0, 1)).all():
        raise ValueError("baseline_mask must hold only 0 and 1")

    channels = _stack_channels(
        _rgb_to_research(reference_image),
        _rgb_to_research(selected_image),
        _depth_arrays(depth_reference, validity_reference, source_shape)[0],
        _depth_arrays(depth_selected, validity_selected, source_shape)[0],
        _depth_arrays(depth_reference, validity_reference, source_shape)[1],
        _depth_arrays(depth_selected, validity_selected, source_shape)[1],
        _flow_array(flow_plane, flow_valid_plane, source_shape),
        _pose_array(list(reference_joints)),
        _pose_array(list(selected_joints)),
        _mask_to_research(mask),
    )
    return torch.from_numpy(channels.transpose(2, 0, 1).copy())


def research_permitted_region(baseline_mask: np.ndarray) -> np.ndarray:
    """Return the region of the bounded model at 320 x 180.

    The input mask is sampled nearest to 320 x 180. A permitted pixel is then a
    pixel inside that baseline or at most 12 research pixels away from it. The
    window is a square of 25 pixels, which is the window of the model.
    """
    mask = np.asarray(baseline_mask).astype(bool)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("baseline_mask must be a non-empty 2-D array")
    kernel = np.ones((PERMIT_KERNEL_PX, PERMIT_KERNEL_PX), np.uint8)
    return cv2.dilate(_mask_to_research(mask).astype(np.uint8), kernel).astype(bool)


def source_permitted_region(baseline_mask: np.ndarray) -> np.ndarray:
    """Return the region the bounded model may change, at the image size.

    The rule moves the research rule of :func:`research_permitted_region` to the
    image size. A pixel is permitted when the baseline mask put a pixel inside
    the 25-pixel research window at that location. The image-size candidate is
    equal to the nearest upsampling of the permitted research pixels.
    """
    mask = np.asarray(baseline_mask)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("baseline_mask must be a non-empty 2-D array")
    if mask.shape == (HEIGHT, WIDTH):
        return research_permitted_region(mask)
    research = research_permitted_region(mask)
    return (
        cv2.resize(research.astype(np.uint8) * 255, (int(mask.shape[1]), int(mask.shape[0])),
                   interpolation=cv2.INTER_NEAREST)
        > 127
    )


def dilate_baseline_ellipse(baseline_mask: np.ndarray, kernel_px: int = PERMIT_KERNEL_PX) -> np.ndarray:
    """Return the image-size dilation of the baseline mask with an ellipse kernel.

    This helper exists to compare kernel shapes. The shadow pipeline uses
    :func:`source_permitted_region`, which follows the square window of the
    model.
    """
    mask = np.asarray(baseline_mask).astype(np.uint8)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("baseline_mask must be a non-empty 2-D array")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
    return cv2.dilate(mask, kernel).astype(bool)


# --------------------------------------------------------------------------
# Part 2 - the shadow runner
# --------------------------------------------------------------------------


def describe_failure(error: BaseException) -> str:
    """Return a short one-line text for an exception."""
    text = " ".join(str(error).split())
    if len(text) > 300:
        text = text[:297] + "..."
    return f"{type(error).__name__}: {text}" if text else type(error).__name__


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of the file bytes as lowercase hexadecimal."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ShadowMaskRunner:
    """Run one frozen research checkpoint on live input tensors.

    The constructor reads the checkpoint one time, checks the task, the trained
    heads, and optionally the file digest, and moves the model to the device.
    Every prediction runs under ``torch.inference_mode()``.

    Decision rules, copied from ``evaluate.py`` and fixed here:

    - At 320 x 180 the mask is ``sigmoid(logits) >= 0.5``.
    - At the image size the mask is the bilinear upsampling of the logits, with
      ``align_corners=False``, tested with ``>= 0``.

    The model also returns a motion head. Shadow mode discards that output. It
    never uses the motion output.

    This class does not measure time. The caller measures time around a call.
    A model failure travels to the caller as an exception; this class does not
    catch it.
    """

    def __init__(
        self,
        *,
        checkpoint: Path,
        device: str = "auto",
        expected_sha256: str | None = None,
        log: Callable[[str], None] = print,
    ) -> None:
        path = Path(checkpoint)
        if path.suffix != ".pt" or not path.is_file():
            raise ValueError(f"checkpoint file does not exist: {path}")
        actual_sha256 = file_sha256(path)
        if expected_sha256 is not None:
            expected = str(expected_sha256).strip().lower()
            if expected != actual_sha256:
                raise ValueError(
                    "checkpoint SHA-256 mismatch: expected "
                    f"{expected}, got {actual_sha256}"
                )
        payload = load_checkpoint(path, expected_task=CHECKPOINT_TASK, map_location="cpu")
        config = payload.get("model_config")
        if not isinstance(config, Mapping):
            raise ValueError("checkpoint has no model_config")
        trained_heads = list(config.get("trained_heads", []))
        if "mask" not in trained_heads:
            raise ValueError(
                f"checkpoint has no trained mask head; trained_heads is {trained_heads}"
            )
        in_channels = int(config["in_channels"])
        if in_channels != len(CHANNEL_ORDER):
            raise ValueError(
                f"checkpoint in_channels must be {len(CHANNEL_ORDER)}, got {in_channels}"
            )
        mask_mode = str(config.get("mask_mode", "absolute"))
        gate_radius_px = int(config.get("gate_radius_px", PERMIT_RADIUS_PX))
        if mask_mode == "bounded_residual" and gate_radius_px != PERMIT_RADIUS_PX:
            raise ValueError(
                "checkpoint gate_radius_px must be "
                f"{PERMIT_RADIUS_PX} for shadow mode, got {gate_radius_px}"
            )
        if device == "auto":
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            if device not in ("cpu", "cuda"):
                raise ValueError("device must be auto, cpu, or cuda")
            if device == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA is not available for the shadow runner")
            selected_device = device

        model = PairedMaskMotionNet(
            in_channels=in_channels,
            base_channels=int(config["base_channels"]),
            mask_mode=mask_mode,
            gate_radius_px=gate_radius_px,
            residual_scale=float(config.get("residual_scale", 4.0)),
        )
        model.load_state_dict(payload["model_state_dict"])
        model.to(selected_device)
        model.eval()
        self._model: nn.Module = model
        self._device = selected_device
        self._sha256 = actual_sha256
        self._config: dict[str, Any] = dict(config)
        self._log = log

    @property
    def checkpoint_sha256(self) -> str:
        """SHA-256 of the checkpoint file bytes."""
        return self._sha256

    @property
    def device(self) -> str:
        """Torch device string of the loaded model."""
        return self._device

    @property
    def model_config(self) -> dict[str, Any]:
        """Plain copy of the checkpoint ``model_config``."""
        return dict(self._config)

    @property
    def model(self) -> nn.Module:
        """The loaded model. A test can read or change its parameters."""
        return self._model

    def predict(self, inputs: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Return the 320 x 180 mask and the source-size logit plane.

        ``inputs`` is one input tensor of shape ``[15, 180, 320]`` or a batch of
        shape ``[1, 15, 180, 320]``. Returns ``(mask, source_logits)``:

        - ``mask`` is bool ``[180, 320]``.
        - ``source_logits`` is float32 ``[H, W]`` for the size of the input
          tensor. For an input of 320 x 180 the return is the same size.

        Use :meth:`predict_for_image` to get the mask at the image size.
        """
        mask, logits, _source_mask = self.predict_for_image(inputs, (HEIGHT, WIDTH))
        return mask, logits

    def predict_for_image(
        self, inputs: torch.Tensor, image_shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the research mask, the image-size logits, and the image mask.

        ``inputs`` is a float tensor ``[15, 180, 320]`` or ``[1, 15, 180, 320]``.
        ``image_shape`` is ``(height, width)`` of the image. Returns:

        - the research mask, bool ``[180, 320]``;
        - the image-size logit plane, float32 ``[H, W]``;
        - the image-size mask, bool ``[H, W]``, from ``logits >= 0``.
        """
        tensor = torch.as_tensor(inputs)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4 or tensor.shape[1] != len(CHANNEL_ORDER):
            raise ValueError("inputs must have shape [15, H, W] or [1, 15, H, W]")
        height, width = int(image_shape[0]), int(image_shape[1])
        if height < 1 or width < 1:
            raise ValueError("image_shape must be positive")
        tensor = tensor.to(device=self._device, dtype=torch.float32)
        with torch.inference_mode():
            mask_logits, _motion = self._model(tensor)
            research_mask = (torch.sigmoid(mask_logits)[0, 0] >= MASK_THRESHOLD).cpu().numpy()
            upsampled = F.interpolate(
                mask_logits, size=(height, width), mode="bilinear", align_corners=False
            )
        source_logits = upsampled[0, 0].cpu().numpy().astype(np.float32)
        return research_mask, source_logits, source_logits >= 0


# --------------------------------------------------------------------------
# Part 3 - the live shadow pipeline
# --------------------------------------------------------------------------


def percentile(values: Iterable[float], p: float) -> float | None:
    """Return the ``p``-th percentile with the nearest-rank rule.

    The rank is ``ceil(p / 100 * n)``, clamped to ``[1, n]``. The result is one
    measured value, not an interpolation. Returns None for an empty input.
    Raises ``ValueError`` when ``p`` is outside ``[0, 100]`` or a value is NaN.
    """
    if not np.isfinite(p) or not (0.0 <= float(p) <= 100.0):
        raise ValueError("p must be in [0, 100]")
    prepared: list[float] = []
    for value in values:
        number = float(value)
        if math.isnan(number):
            raise ValueError("percentile values must not be NaN")
        prepared.append(number)
    if not prepared:
        return None
    prepared.sort()
    rank = min(len(prepared), max(1, int(math.ceil(float(p) / 100.0 * len(prepared)))))
    return float(prepared[rank - 1])


def _fraction(numerator: int, denominator: int) -> float | None:
    """Return numerator / denominator, or None when the denominator is zero."""
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def compare_masks(candidate: np.ndarray, accepted: np.ndarray) -> dict[str, Any]:
    """Compare the candidate shadow mask with the accepted baseline mask.

    A candidate-only pixel is outside the accepted mask by definition, so
    "fraction of added pixels outside the accepted mask" is always 1.0 and says
    nothing. This function reports size changes instead, because a candidate
    that mostly shrinks the person is a different failure from a candidate that
    adds background.

    - ``added_fraction_of_accepted`` and ``removed_fraction_of_accepted``
      divide the added and removed counts by the accepted pixel count.
    - ``net_area_change_fraction`` is the candidate size minus the accepted
      size, divided by the accepted size. A negative value means the candidate
      mask covers less of the person.
    - ``shrink_ratio`` is removed divided by added. A value above 1 means the
      candidate removes more person pixels than it adds.
    """
    candidate_mask = np.asarray(candidate)
    accepted_mask = np.asarray(accepted)
    if candidate_mask.shape != accepted_mask.shape:
        raise ValueError(
            f"candidate mask shape {candidate_mask.shape} does not match the "
            f"accepted mask shape {accepted_mask.shape}"
        )
    candidate_mask = candidate_mask.astype(bool)
    accepted_mask = accepted_mask.astype(bool)
    accepted_pixels = int(np.count_nonzero(accepted_mask))
    candidate_pixels = int(np.count_nonzero(candidate_mask))
    intersection = int(np.count_nonzero(np.logical_and(candidate_mask, accepted_mask)))
    union = int(np.count_nonzero(np.logical_or(candidate_mask, accepted_mask)))
    added = candidate_pixels - intersection
    removed = accepted_pixels - intersection
    return {
        "intersection": intersection,
        "union": union,
        "iou": None if union == 0 else intersection / union,
        "added_pixels": added,
        "removed_pixels": removed,
        "added_fraction_of_accepted": _fraction(added, accepted_pixels),
        "removed_fraction_of_accepted": _fraction(removed, accepted_pixels),
        "net_area_change_fraction": (
            None if accepted_pixels == 0 else (candidate_pixels - accepted_pixels) / accepted_pixels
        ),
        "shrink_ratio": None if added == 0 else removed / added,
    }


def _empty_compare() -> dict[str, Any]:
    """Return a compare block with no counts."""
    return {
        "intersection": 0,
        "union": 0,
        "iou": None,
        "added_pixels": 0,
        "removed_pixels": 0,
        "added_fraction_of_accepted": None,
        "removed_fraction_of_accepted": None,
        "net_area_change_fraction": None,
        "shrink_ratio": None,
    }


def common_joints_for_mask(result: Any) -> list[dict] | None:
    """Build the exact joint records that the live mask selector receives.

    This is the loop of ``RefreshPipeline.process``. The values are not
    rounded. Returns None when the pose data cannot be read.
    """
    try:
        joint_depths = getattr(result, "joint_depth", None)
        joints: list[dict] = []
        for index, landmark in enumerate(result.pose.landmarks):
            depth_value = None
            if joint_depths is not None and index < len(joint_depths):
                candidate = joint_depths[index]
                if candidate is not None and np.isfinite(candidate):
                    depth_value = float(candidate)
            joints.append({
                "index": index,
                "x": float(landmark.x),
                "y": float(landmark.y),
                "visibility": None if landmark.visibility is None else float(landmark.visibility),
                "depth_m": depth_value,
            })
    except (AttributeError, TypeError, ValueError):
        return None
    return joints


def _json_joints(joints: list[dict] | None) -> list[dict]:
    """Return JSON-safe joint records without the depth value."""
    records: list[dict] = []
    for joint in joints or []:
        records.append({
            "index": int(joint.get("index", len(records))),
            "x": float(joint.get("x", 0.0)),
            "y": float(joint.get("y", 0.0)),
            "visibility": None if joint.get("visibility") is None else float(joint["visibility"]),
        })
    return records


def boundary_usable(depth: np.ndarray) -> bool:
    """True when enough border pixels of a depth plane hold usable depth.

    A plane with too little usable depth near the border is usually an estimate
    of the wall behind the person. The provisional baseline drops such planes.
    """
    array = np.asarray(depth)
    height, width = array.shape
    border = BORDER_WIDTH_PX
    if height <= 2 * border or width <= 2 * border:
        return True
    ring = np.zeros(array.shape, dtype=bool)
    ring[:border, :] = True
    ring[-border:, :] = True
    ring[:, :border] = True
    ring[:, -border:] = True
    usable = np.isfinite(array) & (array > 0.0)
    return float(usable[ring].mean()) >= BORDER_MIN_USABLE_FRACTION


def provisional_median_background(depths: Iterable[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a provisional background median for a list of depth planes.

    A pixel holds the median depth of the planes that have usable depth there,
    when at least half of the planes have usable depth at that pixel. Other
    pixels hold NaN. This is the pixel rule of ``BackgroundCalibrator.build``.
    A plane with too little usable border depth is dropped first, because such a
    plane is usually an estimate of the wall behind the person.

    The result is provisional. The planes are the older end of a short rolling
    window, so a person inside that window is part of the result. The result is
    not an empty-scene calibration.

    Raises ``ValueError`` for an empty list, for different plane sizes, and when
    no pixel has usable depth.
    """
    planes = [np.asarray(depth) for depth in depths]
    if not planes:
        raise ValueError("no depth planes were given")
    shape = planes[0].shape
    if any(plane.shape != shape for plane in planes):
        raise ValueError("all depth planes must have the same shape")
    kept = [plane for plane in planes if boundary_usable(plane)]
    if not kept:
        kept = list(planes)
    stack = np.stack([plane.astype(np.float32) for plane in kept], axis=0)
    valid = np.isfinite(stack) & (stack > 0.0)
    usable = valid.mean(axis=0) >= MIN_PIXEL_VALID_RATIO
    if not usable.any():
        raise ValueError("no pixel has usable depth in at least half of the planes")
    with np.errstate(invalid="ignore"):
        column_median = np.nanmedian(np.where(valid, stack, np.nan), axis=0)
    baseline = np.where(usable, column_median, np.nan).astype(np.float32)
    return baseline, {
        "frames_used": len(kept),
        "frames_dropped": len(planes) - len(kept),
        "valid_fraction": float(usable.mean()),
    }


class _WindowFrame:
    """One accepted frame of the rolling shadow window."""

    __slots__ = ("frame_id", "capture_timestamp_ns", "source_id", "rgb", "depth", "valid", "joints")

    def __init__(
        self,
        *,
        frame_id: int,
        capture_timestamp_ns: int,
        source_id: str,
        rgb: np.ndarray,
        depth: np.ndarray,
        valid: np.ndarray,
        joints: list[dict],
    ) -> None:
        self.frame_id = int(frame_id)
        self.capture_timestamp_ns = int(capture_timestamp_ns)
        self.source_id = str(source_id)
        self.rgb = rgb
        self.depth = depth
        self.valid = valid
        self.joints = joints


class ShadowMaskPipeline:
    """Collect live frames and return one shadow record for each frame.

    This class is synchronous. The caller owns the thread. :meth:`poll` runs
    the flow estimator and the model, so the caller must put it on a worker
    thread and never on the publish path.

    Window rules:

    1. Keep at most ``min_images`` depth planes since the last reset.
    2. Clear the window when the generation changes, when no person is found,
       or when the depth size changes.
    3. Use the newest window frame whose flow from the newest frame is below
       ``reference_threshold``. Use the oldest frame when no frame passes.

    Status rules:

    - ``skipped`` with ``warming_up`` before the window holds ``min_images``
      frames.
    - ``skipped`` with ``reference_is_selected`` when the reference frame is the
      selected frame.
    - ``failed`` when a check or a model call fails. ``error`` and
      ``error_type`` then hold a short text.
    - ``ok`` for every other sample.

    The pipeline never changes the accepted mask. It rebuilds that mask from
    the result only, so the record can describe itself.
    """

    def __init__(
        self,
        runner: Any,
        *,
        min_images: int = 3,
        reference_threshold: float = 0.02,
        flow_estimator: Any | None = None,
        log: Callable[[str], None] = print,
    ) -> None:
        if isinstance(min_images, bool) or not isinstance(min_images, int) or min_images < 2:
            raise ValueError("min_images must be an integer of 2 or more")
        if not np.isfinite(reference_threshold) or float(reference_threshold) < 0.0:
            raise ValueError("reference_threshold must be finite and non-negative")
        self.runner = runner
        self.min_images = int(min_images)
        self.reference_threshold = float(reference_threshold)
        self._log = log
        self._flow_estimator = flow_estimator
        self.reset()

    # -- state ---------------------------------------------------------

    def reset(self) -> None:
        """Clear the window, the generation, and the counters."""
        self._window: deque[_WindowFrame] = deque(maxlen=self.min_images)
        self._generation = 0
        self._shape: tuple[int, int] | None = None
        self._counters: dict[str, Any] = {
            "samples": 0, "ok": 0, "skipped": 0, "failed": 0,
            "skip_reasons": {}, "error_types": {},
        }

    @property
    def generation(self) -> int:
        """Generation of the frames in the current window."""
        return self._generation

    @property
    def window_size(self) -> int:
        """Number of frames in the rolling window."""
        return len(self._window)

    @property
    def counters(self) -> dict[str, Any]:
        """Counters since the last reset, as plain data."""
        return {
            "samples": int(self._counters["samples"]),
            "ok": int(self._counters["ok"]),
            "skipped": int(self._counters["skipped"]),
            "failed": int(self._counters["failed"]),
            "skip_reasons": dict(self._counters["skip_reasons"]),
            "error_types": dict(self._counters["error_types"]),
        }

    def set_generation(self, generation: int) -> None:
        """Clear the window when the capture run changes."""
        value = int(generation)
        if value != self._generation:
            self._generation = value
            self._window.clear()
            self._log(f"shadow: generation {value} starts, window cleared")

    def end(self) -> dict[str, Any]:
        """Close the pipeline: clear the window and return the counters."""
        self._window.clear()
        return self.counters

    # -- flow ----------------------------------------------------------

    def _flow(self) -> Any:
        if self._flow_estimator is None:
            estimator_class = _import_live_modules()[0]
            self._flow_estimator = estimator_class()
        return self._flow_estimator

    @staticmethod
    def flow_score(flow: Any) -> float | None:
        """Return the flow score of the live window selector, or None.

        The live ``WindowSelector._score_candidate`` computes the mean of the
        flow magnitude in pixels over the valid pixels, divided by the image
        diagonal. This function returns the same value. It returns None when
        the result has no usable flow, which the live selector reports as an
        invalid candidate.
        """
        if flow is None or getattr(flow, "status", None) != "ok":
            return None
        vectors = np.asarray(getattr(flow, "flow", None))
        valid = np.asarray(getattr(flow, "valid_mask", None))
        if vectors.ndim != 3 or vectors.shape[2] != 2 or valid.shape != vectors.shape[:2]:
            return None
        if vectors.size == 0 or valid.size == 0 or float(valid.mean()) < MIN_VALID_FLOW_FRACTION:
            return None
        magnitudes = np.linalg.norm(vectors[valid.astype(bool)], axis=1)
        if magnitudes.size == 0 or not np.isfinite(magnitudes).all():
            return None
        height, width = vectors.shape[:2]
        diagonal = float(np.hypot(width, height))
        if diagonal <= 0.0:
            return None
        return float(magnitudes.mean()) / diagonal

    def _reference_score(self, flow: Any) -> float:
        """Return the flow score for the reference test, or infinity."""
        score = self.flow_score(flow)
        return math.inf if score is None else score

    def _estimate(self, previous: _WindowFrame, current: _WindowFrame) -> Any:
        """Estimate previous-to-current flow for two window frames."""
        previous_packet = _frame_packet({
            "frame_id": previous.frame_id,
            "source_id": previous.source_id,
            "capture_timestamp_ns": previous.capture_timestamp_ns,
            "rgb": previous.rgb,
        })
        current_packet = _frame_packet({
            "frame_id": current.frame_id,
            "source_id": current.source_id,
            "capture_timestamp_ns": current.capture_timestamp_ns,
            "rgb": current.rgb,
        })
        return self._flow().estimate(previous_packet, current_packet)

    def choose_reference(self) -> tuple[_WindowFrame, float | None] | None:
        """Return the flow reference frame and the flow score of that choice.

        Returns None when the window holds fewer than two frames. The method
        tests the window frames from the newest to the oldest. It returns the
        first frame whose flow from the newest frame is below
        ``reference_threshold``. When no frame passes, it returns the oldest
        frame. The flow uses the direction ``previous_to_current``.

        The second value is ``flow_score`` of the reference-to-newest flow.
        """
        frames = list(self._window)
        if len(frames) < 2:
            return None
        newest = frames[-1]
        threshold = self.reference_threshold
        for earlier in reversed(frames[:-1]):
            try:
                flow = self._estimate(earlier, newest)
            except Exception:
                continue
            if self._reference_score(flow) < threshold:
                return earlier, self.flow_score(flow)
        return frames[0], None

    @staticmethod
    def frame_identity(frame: _WindowFrame) -> tuple[int, int, str]:
        """Return the (frame_id, capture_timestamp_ns, source_id) of a window frame."""
        return (frame.frame_id, frame.capture_timestamp_ns, frame.source_id)

    # -- provisional baseline -----------------------------------------

    def provisional_background(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Return a provisional background median for the current window.

        See :func:`provisional_median_background`. The result is provisional: it
        is the older end of a short rolling window, not an empty-scene
        calibration.
        """
        return provisional_median_background([frame.depth for frame in self._window])

    # -- accepted mask -------------------------------------------------

    @staticmethod
    def accepted_mask(depth: np.ndarray, joints: list[dict] | None) -> tuple[np.ndarray, str, str | None]:
        """Return the accepted live mask for this frame, built like the pipeline.

        The call is the call of ``RefreshPipeline.process``: the pose-gradient
        selector with the default configuration and gradient stride, then the
        joint-depth validity gate when the selector used the pose gradient.
        Returns ``(mask, source, fallback_reason)``.
        """
        _estimator, config_class, selector, _gate = _import_live_modules()
        selection = selector(
            np.asarray(depth, dtype=np.float32),
            pose_joints=joints,
            config=config_class(),
        )
        mask = np.asarray(selection.info.mask).astype(bool)
        if selection.source == "pose_gradient":
            _gate(mask, np.asarray(depth, dtype=np.float32), joints)
        return mask, str(selection.source), selection.fallback_reason

    # -- record helpers ------------------------------------------------

    def _record(self, packet: Any) -> dict[str, Any]:
        """Return a record with identity fields and empty results."""
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "generation": int(self._generation),
            "frame_id": int(getattr(packet, "frame_id", -1)),
            "source_id": str(getattr(packet, "source_id", "")),
            "capture_timestamp_ns": str(getattr(packet, "capture_timestamp_ns", "")),
            "reference_frame_id": None,
            "reference_timestamp_ns": None,
            "flow_score": None,
            "window_frames": len(self._window),
            "background": None,
            "shadow_status": "failed",
            "skip_reason": None,
            "error": None,
            "error_type": None,
            "timings_ms": {"data": 0.0, "model": 0.0, "total": 0.0},
            "accepted": {
                "source": None, "fallback_reason": None,
                "pixels": 0, "area_fraction": 0.0,
            },
            "candidate": {
                "mask_mode": str(self.runner.model_config.get("mask_mode", "absolute")),
                "threshold": MASK_THRESHOLD,
                "source_upsample_rule": SOURCE_UPSAMPLE_RULE,
                "pixels": 0,
                "area_fraction": 0.0,
                "baseline_restricted": True,
            },
            "compare": _empty_compare(),
        }

    def _count(self, record: dict[str, Any]) -> dict[str, Any]:
        """Add a record to the counters and return it."""
        status = record["shadow_status"]
        counters = self._counters
        counters["samples"] = int(counters["samples"]) + 1
        if status == "ok":
            counters["ok"] = int(counters["ok"]) + 1
        elif status == "skipped":
            counters["skipped"] = int(counters["skipped"]) + 1
            reason = str(record.get("skip_reason") or "unknown")
            counters["skip_reasons"][reason] = int(counters["skip_reasons"].get(reason, 0)) + 1
        else:
            counters["failed"] = int(counters["failed"]) + 1
            error_type = str(record.get("error_type") or "Exception")
            counters["error_types"][error_type] = int(counters["error_types"].get(error_type, 0)) + 1
        return record

    @staticmethod
    def _finish(record: dict[str, Any], status: str, started: float, timings: dict[str, float] | None = None) -> None:
        """Set the status, the skip reason, and the timings of a record."""
        record["shadow_status"] = status
        record["skip_reason"] = None
        if timings is not None:
            values = dict(timings)
        else:
            values = {"data": 0.0, "model": 0.0}
        values["total"] = (perf_counter() - started) * 1000.0
        record["timings_ms"] = {key: round(float(value), 3) for key, value in values.items()}

    # -- main entry ----------------------------------------------------

    def poll_result(self, packet: Any, outcome: Any, *, generation: int | None = None) -> dict[str, Any]:
        """Evaluate one accepted refresh in shadow mode.

        ``outcome`` is the live ``RefreshOutcome`` of that refresh. The shadows
        run uses the accepted depth, depth validity, and pose joints of the
        outcome, so the shadow pair matches the frame that the live pipeline
        built. The accepted mask is rebuilt from the same inputs.

        Use this method when the caller has the refresh outcome. Use :meth:`poll`
        when the caller has the depth-stack result instead.

        This method never raises. Any exception becomes a ``failed`` record.
        """
        try:
            view = _outcome_result(outcome)
        except Exception as error:
            record = self._record(packet)
            record["shadow_status"] = "failed"
            record["error"] = describe_failure(error)
            record["error_type"] = type(error).__name__
            return self._count(record)
        return self.poll(packet, view, generation=generation)

    def poll(self, packet: Any, result: Any, *, generation: int | None = None) -> dict[str, Any]:
        """Evaluate one selected frame in shadow mode and return its record.

        ``packet`` is a capture packet with ``rgb`` (RGB uint8), ``frame_id``,
        ``source_id``, and ``capture_timestamp_ns``. ``result`` is the depth
        stack result of that frame, with ``depth.depth``, ``depth.valid``,
        ``pose.landmarks``, ``joint_depth``, and ``detected``. ``generation``
        names the capture run; a change clears the window.

        This method never raises. Any exception becomes a ``failed`` record.
        """
        started = perf_counter()
        if generation is not None:
            try:
                self.set_generation(int(generation))
            except Exception:
                pass
        record = self._record(packet)
        try:
            return self._poll_inner(packet, result, record, started)
        except Exception as error:
            record["shadow_status"] = "failed"
            record["skip_reason"] = None
            record["error"] = describe_failure(error)
            record["error_type"] = type(error).__name__
            record["timings_ms"]["total"] = round((perf_counter() - started) * 1000.0, 3)
            return self._count(record)

    def _poll_inner(
        self, packet: Any, result: Any, record: dict[str, Any], started: float
    ) -> dict[str, Any]:
        """Do the work of :meth:`poll`. An exception travels to the caller."""
        depth = np.asarray(result.depth.depth)
        if depth.ndim != 2:
            raise ValueError(f"result depth must be 2-D, got shape {depth.shape}")
        depth = depth.astype(np.float32)
        shape = (int(depth.shape[0]), int(depth.shape[1]))
        rgb = np.asarray(packet.rgb)
        if rgb.dtype != np.uint8:
            raise ValueError(f"packet rgb must be uint8, got {rgb.dtype}")
        if rgb.shape[:2] != shape:
            raise ValueError(f"packet rgb size {rgb.shape[:2]} does not match depth {shape}")

        if self._shape is not None and shape != self._shape:
            self._log(f"shadow: depth size changed to {shape}, window cleared")
            self._window.clear()
        self._shape = shape

        landmarks = list(getattr(result.pose, "landmarks", []) or [])
        if not bool(getattr(result, "detected", False)) or not landmarks:
            self._window.clear()
            self._finish(record, "skipped", started)
            record["skip_reason"] = "no_pose"
            return self._count(record)
        joints = common_joints_for_mask(result)
        if joints is None:
            self._window.clear()
            self._finish(record, "skipped", started)
            record["skip_reason"] = "no_pose"
            return self._count(record)

        valid = np.asarray(getattr(result.depth, "valid", None))
        if valid.shape != shape:
            valid = np.isfinite(depth).astype(np.float32)
        self._window.append(_WindowFrame(
            frame_id=int(packet.frame_id),
            capture_timestamp_ns=int(packet.capture_timestamp_ns),
            source_id=str(packet.source_id),
            rgb=np.array(rgb, copy=True),
            depth=depth.copy(),
            valid=np.asarray(valid, dtype=np.float32),
            joints=joints,
        ))
        record["window_frames"] = len(self._window)
        if len(self._window) < self.min_images:
            self._finish(record, "skipped", started)
            record["skip_reason"] = "warming_up"
            return self._count(record)

        data_started = perf_counter()
        data_timings = {"data": 0.0, "model": 0.0}
        chosen = self.choose_reference()
        if chosen is None:
            self._finish(record, "skipped", started)
            record["skip_reason"] = "flow_unavailable"
            return self._count(record)
        reference, chosen_score = chosen
        record["reference_frame_id"] = int(reference.frame_id)
        record["reference_timestamp_ns"] = str(int(reference.capture_timestamp_ns))
        if self.frame_identity(reference) == self.frame_identity(self._window[-1]):
            self._finish(record, "skipped", started)
            record["skip_reason"] = "reference_is_selected"
            return self._count(record)

        try:
            flow = self._estimate(reference, self._window[-1])
        except Exception:
            flow = None
        if flow is None or getattr(flow, "status", None) != "ok":
            self._finish(record, "skipped", started)
            record["skip_reason"] = "flow_unavailable"
            return self._count(record)
        score = chosen_score if chosen_score is not None else self.flow_score(flow)
        record["flow_score"] = None if score is None else round(float(score), 6)
        flow_vectors = np.asarray(getattr(flow, "flow", None))
        flow_valid = np.asarray(getattr(flow, "valid_mask", None))
        if flow_vectors.shape != (*shape, 2) or flow_valid.shape != shape:
            self._finish(record, "skipped", started)
            record["skip_reason"] = "flow_unavailable"
            return self._count(record)

        try:
            background, background_record = self.provisional_background()
        except ValueError as error:
            record["shadow_status"] = "skipped"
            record["skip_reason"] = "background_unavailable"
            record["error"] = describe_failure(error)
            record["error_type"] = type(error).__name__
            record["timings_ms"]["total"] = round((perf_counter() - started) * 1000.0, 3)
            return self._count(record)
        record["background"] = background_record
        _ = background

        accepted, accepted_source, fallback_reason = self.accepted_mask(depth, joints)
        if accepted.shape != shape:
            raise ValueError(f"accepted mask shape {accepted.shape} does not match {shape}")
        accepted_pixels = int(np.count_nonzero(accepted))
        record["accepted"] = {
            "source": accepted_source,
            "fallback_reason": fallback_reason,
            "pixels": accepted_pixels,
            "area_fraction": accepted_pixels / float(accepted.size),
        }

        inputs = build_research_input(
            reference_rgb=reference.rgb,
            selected_rgb=np.asarray(rgb),
            reference_depth=reference.depth,
            reference_validity=reference.valid,
            selected_depth=depth,
            selected_validity=np.asarray(valid, dtype=np.float32),
            flow=flow_vectors,
            flow_valid=flow_valid,
            reference_joints=reference.joints,
            selected_joints=joints,
            reference_frame_id=reference.frame_id,
            selected_frame_id=int(packet.frame_id),
            baseline_mask=accepted,
        )
        data_done = perf_counter()
        data_timings["data"] = (data_done - data_started) * 1000.0

        research_mask, source_logits, predicted = self.runner.predict_for_image(inputs, shape)
        model_done = perf_counter()
        data_timings["model"] = (model_done - data_done) * 1000.0
        _ = research_mask, source_logits
        predicted = np.asarray(predicted)
        if predicted.shape != accepted.shape:
            raise ValueError(
                f"candidate mask shape {predicted.shape} does not match the "
                f"accepted mask shape {accepted.shape}"
            )
        candidate = np.logical_and(predicted.astype(bool), source_permitted_region(accepted))
        candidate_pixels = int(np.count_nonzero(candidate))
        record["candidate"]["pixels"] = candidate_pixels
        record["candidate"]["area_fraction"] = candidate_pixels / float(candidate.size)
        record["compare"] = compare_masks(candidate, accepted)
        self._finish(record, "ok", started, data_timings)
        return self._count(record)


# --------------------------------------------------------------------------
# Accepted refresh outcome as a poll input
# --------------------------------------------------------------------------


def _outcome_result(outcome: Any) -> Any:
    """Wrap a live ``RefreshOutcome`` as the result object that :meth:`poll` reads.

    The shadow run reads four items from the depth-stack result: the depth
    plane, its validity, the pose landmarks, and the joint depths. A live
    ``RefreshOutcome`` already holds the accepted depth, the accepted depth
    validity, and the unrounded pose joints with their joint depths. This
    function builds a small object with the same attribute names, so the shadow
    pair uses the accepted values and no second depth or pose inference runs.
    """
    depth = np.asarray(getattr(outcome, "depth_m"), dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"outcome depth must be 2-D, got shape {depth.shape}")
    mask = getattr(outcome, "mask", None)
    valid = getattr(mask, "validity", None)
    if valid is None:
        valid = np.isfinite(depth).astype(np.float32)
    valid = np.asarray(valid, dtype=np.float32)
    if valid.shape != depth.shape:
        raise ValueError(
            f"outcome validity shape {valid.shape} does not match depth shape {depth.shape}"
        )
    joints = getattr(outcome, "mask_pose_joints", None)
    if joints is None:
        raise ValueError("outcome has no exact mask joints")
    landmarks = [_Joint(joint) for joint in joints]
    depths = np.array(
        [np.nan if joint.get("depth_m") is None else float(joint["depth_m"]) for joint in joints],
        dtype=np.float64,
    )
    return _Result(depth=_Depth(depth=depth, valid=valid), pose=_Pose(landmarks=landmarks), joint_depth=depths)


class _Joint:
    """One landmark with the attribute names that the pose result uses."""

    __slots__ = ("x", "y", "visibility")

    def __init__(self, joint: Mapping[str, Any]) -> None:
        self.x = float(joint["x"])
        self.y = float(joint["y"])
        visibility = joint.get("visibility")
        self.visibility = None if visibility is None else float(visibility)


class _Pose:
    """Landmark list in the shape that :func:`common_joints_for_mask` reads."""

    __slots__ = ("landmarks",)

    def __init__(self, *, landmarks: list[_Joint]) -> None:
        self.landmarks = landmarks


class _Depth:
    """Depth plane and validity plane in the shape that :meth:`poll` reads."""

    __slots__ = ("depth", "valid")

    def __init__(self, *, depth: np.ndarray, valid: np.ndarray) -> None:
        self.depth = depth
        self.valid = valid


class _Result:
    """The four result attributes that :meth:`poll` reads."""

    __slots__ = ("depth", "pose", "joint_depth", "detected")

    def __init__(self, *, depth: _Depth, pose: _Pose, joint_depth: np.ndarray) -> None:
        self.depth = depth
        self.pose = pose
        self.joint_depth = joint_depth
        self.detected = True


# --------------------------------------------------------------------------
# JSONL record file
# --------------------------------------------------------------------------


class ShadowProcessWriter:
    """Run the shadow pipeline in one child process and write its records.

    The shadow pipeline decodes, resizes, and filters images with NumPy. A
    worker thread shares the interpreter lock with the capture loop, so the
    capture loop loses about 180 frames in a 50 s run and the refresh latency
    rises from 152 ms to 461 ms. A child process has its own interpreter, so it
    cannot take the lock from the live path.

    The parent sends one request at a time. A request that arrives while the
    child is busy is dropped and counted, so the live path never waits.

    The child writes the JSONL file. The parent never opens it for writing, so
    two processes cannot interleave lines.
    """

    def __init__(
        self,
        *,
        checkpoint: Path,
        log_path: Path,
        expected_sha256: str | None = None,
        device: str = "auto",
        min_images: int = 3,
        reference_threshold: float = 0.02,
        session_id: str | None = None,
    ) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._busy = False
        self._dropped = 0
        self._writes = 0
        self._errors = 0
        self._rejected = 0
        self._closed = False
        self._context = multiprocessing.get_context("spawn")
        self._request: Any = self._context.Queue(maxsize=1)
        self._done: Any = self._context.Queue()
        self._process = self._context.Process(
            target=_shadow_child_main,
            args=(
                str(checkpoint),
                expected_sha256,
                device,
                int(min_images),
                float(reference_threshold),
                str(self.log_path),
                session_id,
                self._request,
                self._done,
            ),
            name="shadow-mask-child",
            daemon=True,
        )
        self._process.start()

    @property
    def counters(self) -> dict[str, Any]:
        """Requests sent, records written, drops, and child errors."""
        self._drain()
        return {
            "writes": int(self._writes),
            "dropped_busy": int(self._dropped),
            "child_errors": int(self._errors),
            "rejected_requests": int(self._rejected),
            "busy": bool(self._busy),
            "child_alive": bool(self._process.is_alive()),
        }

    def _drain(self) -> None:
        """Take every finished message from the child. Never blocks."""
        while True:
            try:
                message = self._done.get_nowait()
            except queue.Empty:
                return
            if not isinstance(message, dict):
                continue
            if message.get("kind") == "wrote":
                self._writes += 1
            elif message.get("kind") == "error":
                self._errors += 1
                print(
                    json.dumps({"shadow_error": f"{message.get('error_type')}: {message.get('error')}"}),
                    flush=True,
                )
            self._busy = False

    def submit(self, packet: Any, outcome: Any, *, generation: int) -> bool:
        """Send one refresh to the child. Returns False when the request is dropped.

        This method never raises and never blocks on the child.
        """
        if self._closed:
            return False
        self._drain()
        if self._busy:
            self._dropped += 1
            return False
        request = {
            "rgb": _copy_plane(getattr(packet, "rgb", None)),
            "frame_id": int(getattr(packet, "frame_id", 0)),
            "source_id": str(getattr(packet, "source_id", "")),
            "capture_timestamp_ns": int(getattr(packet, "capture_timestamp_ns", 0)),
            "generation": int(generation),
            "depth_m": _copy_plane(getattr(outcome, "depth_m", None)),
            "validity": _copy_plane(getattr(getattr(outcome, "mask", None), "validity", None)),
            "joints": _json_joints(getattr(outcome, "mask_pose_joints", None)),
        }
        if request["rgb"] is None or request["depth_m"] is None:
            # A request without a frame is a caller fault, not a child fault.
            self._rejected += 1
            return False
        try:
            self._request.put_nowait(request)
        except queue.Full:
            self._dropped += 1
            return False
        self._busy = True
        return True

    def close(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        """Stop the child and return the final counters."""
        if self._closed:
            return self.counters
        self._closed = True
        try:
            self._request.put_nowait({"kind": "stop"})
        except queue.Full:
            pass
        self._process.join(timeout=float(timeout_s))
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5.0)
        self._drain()
        return self.counters


def _copy_plane(array: Any) -> np.ndarray | None:
    """Return a copy of an array for the child process, or None."""
    if array is None:
        return None
    result = np.asarray(array)
    return np.array(result, copy=True)


def _shadow_child_main(
    checkpoint: str,
    expected_sha256: str | None,
    device: str,
    min_images: int,
    reference_threshold: float,
    log_path: str,
    session_id: str | None,
    request_queue: Any,
    done_queue: Any,
) -> None:
    """Child process body: one request in, one record out."""
    try:
        runner = ShadowMaskRunner(
            checkpoint=Path(checkpoint),
            device=device,
            expected_sha256=expected_sha256,
        )
        pipeline = ShadowMaskPipeline(runner, min_images=min_images, reference_threshold=reference_threshold)
        writer = ShadowRecordWriter(Path(log_path), session_id=session_id)
    except Exception as error:  # the parent sees the error and stops
        done_queue.put(
            {"kind": "error", "error": str(error), "error_type": type(error).__name__}
        )
        return
    try:
        while True:
            request = request_queue.get()
            if not isinstance(request, dict) or request.get("kind") == "stop":
                return
            try:
                outcome = _ChildOutcome(request)
                packet = _frame_packet(request)
                record = pipeline.poll_result(packet, outcome, generation=int(request["generation"]))
                writer.write(record)
                done_queue.put({"kind": "wrote", "status": record.get("shadow_status")})
            except Exception as error:
                done_queue.put(
                    {"kind": "error", "error": str(error), "error_type": type(error).__name__}
                )
    finally:
        writer.close()


class _ChildOutcome:
    """The four outcome attributes that the child process needs."""

    __slots__ = ("depth_m", "mask", "mask_pose_joints", "frame_id", "source_id")

    def __init__(self, request: Mapping[str, Any]) -> None:
        self.depth_m = request["depth_m"]
        self.mask = _Depth(
            depth=request["depth_m"],
            valid=np.asarray(request["validity"], dtype=np.float32),
        )
        self.mask_pose_joints = request["joints"]
        self.frame_id = int(request["frame_id"])
        self.source_id = str(request["source_id"])


class ShadowRecordWriter:
    """Append shadow records to one JSONL file.

    The file opens in append mode as UTF-8 text with Unix line endings, so a
    second run does not destroy the records of the first run. Each record gets
    a ``session_id`` field. The field holds the value that the record already
    has, or the session ID of this writer.
    """

    def __init__(self, path: Path, *, session_id: str | None = None) -> None:
        self.path = Path(path)
        self.session_id = session_id
        if self.path.parent and not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "a", encoding="utf-8", newline="\n")
        self._closed = False

    @property
    def closed(self) -> bool:
        """True after :meth:`close`."""
        return self._closed

    def write(self, record: Mapping[str, Any]) -> None:
        """Append one record, flush it, and give it a session ID."""
        if self._closed:
            raise ValueError("the shadow record writer is closed")
        payload = dict(record)
        if payload.get("session_id") is None:
            payload["session_id"] = self.session_id
        text = json.dumps(payload, allow_nan=False)
        self._handle.write(text + "\n")
        self._handle.flush()

    def close(self) -> None:
        """Close the file. A second call does nothing."""
        if not self._closed:
            self._handle.close()
            self._closed = True

    def __enter__(self) -> "ShadowRecordWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def read_shadow_records(path: Path) -> list[dict]:
    """Read one JSONL shadow record file. Empty lines are ignored."""
    records: list[dict] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        record = json.loads(stripped)
        if not isinstance(record, dict):
            raise ValueError(f"shadow record on line {number} is not a JSON object")
        records.append(record)
    return records


def _collect(values: list[float], value: Any) -> None:
    """Add one finite number to a list of values."""
    if value is None or isinstance(value, bool):
        return
    try:
        number = float(value)
    except (TypeError, ValueError):
        return
    if not math.isfinite(number):
        return
    values.append(number)


def _summary(values: list[float]) -> dict[str, Any]:
    """Return the count, p50, and p95 of a list of values."""
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "p50": percentile(ordered, 50.0),
        "p95": percentile(ordered, 95.0),
    }


def summarize_shadow_records(records: Iterable[dict]) -> dict:
    """Summarize shadow records: counts, skip reasons, errors, and percentiles.

    Percentiles use the nearest-rank rule of :func:`percentile`. A metric with
    no value reports a count of zero and None for each percentile.
    """
    rows = [record for record in records if isinstance(record, Mapping)]
    status_counts: dict[str, int] = {status: 0 for status in STATUSES}
    skip_reasons: dict[str, int] = {}
    error_types: dict[str, int] = {}
    values: dict[str, list[float]] = {
        "timings_ms_total": [],
        "timings_ms_data": [],
        "timings_ms_model": [],
        "compare_iou": [],
        "compare_removed_fraction_of_accepted": [],
        "compare_net_area_change_fraction": [],
        "compare_shrink_ratio": [],
        "candidate_area_fraction": [],
        "accepted_area_fraction": [],
    }
    for record in rows:
        status = record.get("shadow_status")
        if status in status_counts:
            status_counts[str(status)] += 1
        reason = record.get("skip_reason")
        if status == "skipped" and reason:
            skip_reasons[str(reason)] = skip_reasons.get(str(reason), 0) + 1
        error_type = record.get("error_type")
        if error_type:
            error_types[str(error_type)] = error_types.get(str(error_type), 0) + 1
        timings = record.get("timings_ms") or {}
        compare = record.get("compare") or {}
        candidate = record.get("candidate") or {}
        accepted = record.get("accepted") or {}
        _collect(values["timings_ms_total"], timings.get("total"))
        _collect(values["timings_ms_data"], timings.get("data"))
        _collect(values["timings_ms_model"], timings.get("model"))
        _collect(values["compare_iou"], compare.get("iou"))
        _collect(values["compare_removed_fraction_of_accepted"], compare.get("removed_fraction_of_accepted"))
        _collect(values["compare_net_area_change_fraction"], compare.get("net_area_change_fraction"))
        _collect(values["compare_shrink_ratio"], compare.get("shrink_ratio"))
        _collect(values["candidate_area_fraction"], candidate.get("area_fraction"))
        _collect(values["accepted_area_fraction"], accepted.get("area_fraction"))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "records": len(rows),
        "status_counts": status_counts,
        "skip_reasons": skip_reasons,
        "error_types": error_types,
    }
    for name, numbers in values.items():
        summary[name] = _summary(numbers)
    return summary
