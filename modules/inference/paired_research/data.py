"""Load checked replay pairs with the fixed 15-channel research transform."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

WIDTH = 320
HEIGHT = 180
CHANNEL_ORDER = (
    "reference_rgb_r", "reference_rgb_g", "reference_rgb_b",
    "selected_rgb_r", "selected_rgb_g", "selected_rgb_b",
    "reference_depth_m_over_10", "selected_depth_m_over_10",
    "reference_depth_valid", "selected_depth_valid",
    "dis_flow_x_over_width", "dis_flow_y_over_height",
    "reference_pose", "selected_pose", "baseline_mask",
)


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"path leaves replay directory: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _image(path: Path, flags: int) -> np.ndarray:
    result = cv2.imread(str(path), flags)
    if result is None:
        raise ValueError(f"cannot read image: {path}")
    return result


def _depth_arrays(
    depth: np.ndarray, validity: np.ndarray, source_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the 320 x 180 depth and validity channels from source-size arrays.

    This pure function holds the research depth rule. The replay loader and the
    live path call it, so both paths use the same arithmetic.
    """
    depth = np.asarray(depth).astype(np.float32)
    valid = np.asarray(validity).astype(np.float32)
    if depth.shape != source_shape or valid.shape != source_shape:
        raise ValueError("depth dimensions do not match RGB")
    valid = (np.isfinite(depth) & (valid > 0.5)).astype(np.float32)
    scaled = np.clip(np.where(valid > 0, depth, 0.0), 0.0, 10.0) / 10.0
    weight = cv2.resize(valid, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    numerator = cv2.resize(scaled * valid, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    output_valid = (weight > 0.5).astype(np.float32)
    output_depth = np.divide(numerator, weight, out=np.zeros_like(numerator), where=weight > 0.5)
    return output_depth * output_valid, output_valid


def _pose_array(joints: list[dict]) -> np.ndarray:
    """Return the 320 x 180 pose map: one filled disk for each visible joint.

    A joint must have finite values, visibility 0.15 or more, and x and y in
    [0, 1). The disk value is the visibility, clamped to 1.
    """
    result = np.zeros((HEIGHT, WIDTH), np.float32)
    for joint in joints:
        x, y = float(joint["x"]), float(joint["y"])
        visibility = float(joint["visibility"])
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(visibility)):
            continue
        if visibility < 0.15 or not (0 <= x < 1 and 0 <= y < 1):
            continue
        cv2.circle(result, (min(WIDTH - 1, int(x * WIDTH)), min(HEIGHT - 1, int(y * HEIGHT))), 2, min(1.0, visibility), -1)
    return result


def _flow_array(
    flow: np.ndarray, valid: np.ndarray, source_shape: tuple[int, int]
) -> np.ndarray:
    """Return the 320 x 180 flow channels from source-size flow and validity."""
    flow = np.asarray(flow).astype(np.float32)
    valid = np.asarray(valid).astype(np.uint8)
    if flow.shape != (*source_shape, 2) or valid.shape != source_shape:
        raise ValueError("flow dimensions do not match RGB")
    finite = np.isfinite(flow).all(axis=2)
    source_valid = ((valid > 0) & finite).astype(np.uint8)
    normalized = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
    normalized[..., 0] /= source_shape[1]
    normalized[..., 1] /= source_shape[0]
    resized = cv2.resize(normalized, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    resized_valid = cv2.resize(source_valid, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
    resized[resized_valid == 0] = 0
    return resized


def _depth(path: Path, source_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        depth = archive["depth_m"]
        valid = archive["validity"]
    if depth.shape != source_shape or valid.shape != source_shape:
        raise ValueError(f"depth dimensions do not match RGB: {path}")
    return _depth_arrays(depth, valid, source_shape)


def _pose(path: Path, frame_id: int) -> np.ndarray:
    record = json.loads(path.read_text(encoding="utf-8"))
    if int(record["frame_id"]) != frame_id:
        raise ValueError(f"pose frame identity does not match: {path}")
    return _pose_array(record["joints"])


def _flow(path: Path, source_shape: tuple[int, int]) -> np.ndarray:
    with np.load(path) as archive:
        flow = archive["flow"]
        valid = archive["valid_mask"]
    if flow.shape != (*source_shape, 2) or valid.shape != source_shape:
        raise ValueError(f"flow dimensions do not match RGB: {path}")
    return _flow_array(flow, valid, source_shape)


def _stack_channels(
    reference: np.ndarray, current: np.ndarray,
    reference_depth: np.ndarray, current_depth: np.ndarray,
    reference_valid: np.ndarray, current_valid: np.ndarray,
    flow: np.ndarray, reference_pose: np.ndarray, current_pose: np.ndarray,
    baseline: np.ndarray,
) -> np.ndarray:
    """Stack the ten research planes into one finite [180, 320, 15] array.

    The channel order is ``CHANNEL_ORDER``. The function raises ``ValueError``
    when the result is not finite or does not have the research shape.
    """
    channels = np.concatenate((
        reference, current, reference_depth[..., None], current_depth[..., None],
        reference_valid[..., None], current_valid[..., None], flow,
        reference_pose[..., None], current_pose[..., None], baseline[..., None],
    ), axis=2)
    if channels.shape != (HEIGHT, WIDTH, len(CHANNEL_ORDER)) or not np.isfinite(channels).all():
        raise ValueError("invalid research channels")
    return channels


def _pair(replay: Path, labels: Path, row: dict, entry: dict) -> tuple[Tensor, Tensor]:
    key = entry["frame_key"]
    selected_id = int(row["selected_frame_id"])
    reference_id = int(row["reference_frame_id"])
    flow_meta = row.get("flow_metadata") or {}
    if (int(flow_meta.get("previous_frame_id", -1)) != reference_id
            or int(flow_meta.get("current_frame_id", -1)) != selected_id
            or flow_meta.get("direction") != "previous_to_current"):
        raise ValueError(f"flow frame identity does not match: {key}")
    current_bgr = _image(_inside(replay, row["selected_rgb"]), cv2.IMREAD_COLOR)
    reference_bgr = _image(_inside(replay, row["reference_rgb"]), cv2.IMREAD_COLOR)
    source_shape = current_bgr.shape[:2]
    if reference_bgr.shape != current_bgr.shape:
        raise ValueError(f"reference and selected RGB dimensions differ: {key}")
    current = cv2.cvtColor(cv2.resize(current_bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    reference = cv2.cvtColor(cv2.resize(reference_bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    reference_depth, reference_valid = _depth(_inside(replay, row["reference_depth"]), source_shape)
    current_depth, current_valid = _depth(_inside(replay, row["selected_depth"]), source_shape)
    flow = _flow(_inside(replay, row["flow"]), source_shape)
    current_pose = _pose(_inside(replay, row["mask_pose"]), selected_id)
    reference_pose_name = Path(row["reference_rgb"]).name.replace("_rgb.png", "_mask_pose.json")
    if reference_pose_name == Path(row["reference_rgb"]).name:
        raise ValueError(f"reference RGB name has no frame key: {key}")
    reference_pose = _pose(_inside(replay, reference_pose_name), reference_id)
    baseline = _image(_inside(replay, row["baseline_mask"]), cv2.IMREAD_GRAYSCALE)
    target = _image(_inside(labels, entry["label"]), cv2.IMREAD_GRAYSCALE)
    if baseline.shape != source_shape or target.shape != source_shape:
        raise ValueError(f"mask dimensions do not match RGB: {key}")
    if not np.isin(np.unique(target), (0, 255)).all():
        raise ValueError(f"checked mask is not binary: {key}")
    baseline = cv2.resize(baseline, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
    target = (cv2.resize(target, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST) > 127).astype(np.float32)
    try:
        channels = _stack_channels(
            reference, current, reference_depth, current_depth,
            reference_valid, current_valid, flow, reference_pose,
            current_pose, baseline,
        )
    except ValueError as error:
        raise ValueError(f"invalid training channels: {key}") from error
    return torch.from_numpy(channels.transpose(2, 0, 1).copy()), torch.from_numpy(target[None, ...].copy())


class PairedReplayDataset(Dataset):
    """Keep transformed pairs in memory for a small diagnostic training run."""

    def __init__(
        self, replay_dir: Path, labels_root: Path, filter_path: Path,
        *, expected_split: str = "train",
    ) -> None:
        self.replay_dir = Path(replay_dir).resolve()
        self.labels_root = Path(labels_root).resolve()
        manifest = json.loads((self.replay_dir / "manifest.json").read_text(encoding="utf-8"))
        if expected_split not in ("train", "validation", "test") or manifest.get("split") != expected_split:
            raise ValueError(f"replay split must be {expected_split}")
        self.manifest = manifest
        filtered = json.loads(Path(filter_path).read_text(encoding="utf-8"))
        if Path(filtered["output_dir"]).resolve() != self.replay_dir:
            raise ValueError("training filter is for a different replay")
        if Path(filtered["labels_root"]).resolve() != self.labels_root:
            raise ValueError("training filter is for a different label store")
        if not filtered["eligible"]:
            raise ValueError("training filter has no eligible pairs")
        rows = [json.loads(line) for line in (self.replay_dir / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
        self.frame_keys: list[str] = []
        self.rows: list[dict] = []
        self.label_paths: list[Path] = []
        self.samples: list[tuple[Tensor, Tensor]] = []
        for entry in filtered["eligible"]:
            row = rows[int(entry["index"])]
            if (row.get("split") != expected_split or row.get("status") != "ok"
                    or row.get("selected_frame_id") != entry["frame_id"]
                    or row.get("reference_frame_id") != entry["reference_frame_id"]):
                raise ValueError(f"filter row identity does not match: {entry['frame_key']}")
            self.samples.append(_pair(self.replay_dir, self.labels_root, row, entry))
            self.frame_keys.append(entry["frame_key"])
            self.rows.append(row)
            self.label_paths.append(_inside(self.labels_root, entry["label"]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.samples[index]
