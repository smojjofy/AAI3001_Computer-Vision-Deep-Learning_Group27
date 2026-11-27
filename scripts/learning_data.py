"""Replay learning-data tools: review set, labelled mask evaluation, training filter.

These helpers read one replay output, write review and report material to
separate directories, and keep manually checked masks in a store that is never
the replay data. ``baseline_mask`` is model output and is never a label.

Example flow, from ``live-splat-pipeline/``:

    python -m scripts.learning_data review-set --output-dir <clip>
    python -m scripts.learning_data review-set --tool-labeler --session label_01
    python -m scripts.learning_data evaluate --output-dir <clip>
    python -m scripts.learning_data training-filter --output-dir <clip>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from applications.live_pipeline.adapters import DepthThresholdConfig  # noqa: E402
from applications.live_pipeline.mask_selection import select_pose_gradient_mask  # noqa: E402
from scripts.inspect_replay_masks import (  # noqa: E402
    _read_image,
    _summary,
    load_rows,
    load_mask_joints,
    region_box,
    _visible_pixels,
)

LABEL_SCHEMA_VERSION = "1"
LOW_AGREEMENT_IOU = 0.9
# Stride 4 is the live default. The others are comparison methods.
COMPARISON_STRIDES = (1, 2, 4)
LIVE_STRIDE = 4


# ---------------------------------------------------------------- utilities


def frame_key(row: dict) -> str:
    """Stable per-row key. It matches the inspection report overlay names."""
    frame_id = row.get("selected_frame_id")
    generation = row.get("generation")
    if frame_id is None:
        return "unknown"
    return f"g{generation}_f{int(frame_id):08d}"


def mask_iou(first: np.ndarray, second: np.ndarray) -> float | None:
    a = np.asarray(first, dtype=bool)
    b = np.asarray(second, dtype=bool)
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return None
    return float(np.logical_and(a, b).sum() / union)


def load_depth(output_dir: Path, row: dict) -> np.ndarray | None:
    path = row.get("selected_depth")
    if not path:
        return None
    candidate = output_dir / path
    if not candidate.is_file():
        return None
    with np.load(candidate) as data:
        if "depth_m" not in data:
            return None
        return np.asarray(data["depth_m"], dtype=np.float32)


def load_mask_image(path: Path) -> np.ndarray | None:
    image = _read_image(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return image > 0


# ------------------------------------------------------------- review set


def _load_stride_rows(report_dir: Path) -> dict[int, dict]:
    path = report_dir / "report.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["frame_id"]: row for row in payload.get("rows", []) if row.get("frame_id") is not None}


def select_review_frames(
    rows: list[dict],
    stride_report: dict[int, dict],
    *,
    low_agreement_iou: float = LOW_AGREEMENT_IOU,
    high_agreement_count: int = 12,
    missing_person_count: int = 6,
) -> list[dict]:
    """Choose the frames for manual review, with the reason for each frame.

    The set holds every low-agreement frame that has a comparable mask, a
    spread of high-agreement frames across the clip, all no-pose fallback rows,
    and evenly spaced missing-person rows. A frame can have more than one
    reason.
    """
    selected: dict[int, dict] = {}

    def add(index: int, reason: str) -> None:
        entry = selected.setdefault(index, {"index": index, "reasons": []})
        if reason not in entry["reasons"]:
            entry["reasons"].append(reason)

    comparable = []
    for index, row in enumerate(rows):
        frame_id = row.get("selected_frame_id")
        info = stride_report.get(frame_id) if frame_id is not None else None
        if info is None or info.get("mask_iou") is None:
            continue
        comparable.append((index, float(info["mask_iou"])))

    low = sorted((item for item in comparable if item[1] < low_agreement_iou), key=lambda item: item[1])
    for index, _iou in low:
        add(index, "low_agreement")

    high = [item for item in comparable if item[1] >= low_agreement_iou]
    if high and high_agreement_count > 0:
        step = max(1, len(high) // high_agreement_count)
        for index, _iou in high[::step][:high_agreement_count]:
            add(index, "high_agreement_sample")

    for index, row in enumerate(rows):
        if row.get("mask_source") == "depth_threshold_fallback":
            add(index, "no_pose_fallback")

    missing = [index for index, row in enumerate(rows) if row.get("status") != "ok"]
    if missing and missing_person_count > 0:
        step = max(1, len(missing) // missing_person_count)
        for index in missing[::step][:missing_person_count]:
            add(index, "missing_person_sample")

    return sorted(
        (
            {
                **entry,
                "frame_key": frame_key(rows[entry["index"]]),
                "frame_id": rows[entry["index"]].get("selected_frame_id"),
                "timestamp_s": round(
                    float(rows[entry["index"]].get("selected_timestamp_ns", 0)) / 1e9, 3
                ),
                "status": rows[entry["index"]].get("status"),
                "mask_source": rows[entry["index"]].get("mask_source"),
                "fallback_reason": rows[entry["index"]].get("fallback_reason"),
                "stride_iou": (
                    None
                    if rows[entry["index"]].get("selected_frame_id") not in stride_report
                    else stride_report[rows[entry["index"]]["selected_frame_id"]].get("mask_iou")
                ),
            }
            for entry in selected.values()
        ),
        key=lambda item: item["index"],
    )


def _view_pair(
    rgb: np.ndarray,
    mask: np.ndarray | None,
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    """RGB beside an RGB and mask overlay. A missing mask shows the RGB only."""
    if mask is None:
        overlay = rgb.copy()
        cv2.putText(overlay, "no mask", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return np.hstack([rgb, overlay])
    inside = np.asarray(mask, dtype=bool)
    overlay = rgb.copy()
    tint = np.array([0, 200, 0], dtype=np.float32)
    overlay[inside] = (overlay[inside].astype(np.float32) * (1 - alpha) + tint * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(inside.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    return np.hstack([rgb, overlay])


def build_review_set(
    output_dir: Path,
    *,
    review_dir: Path | None = None,
    stride_report_dir: Path | None = None,
    low_agreement_iou: float = LOW_AGREEMENT_IOU,
    high_agreement_count: int = 12,
    missing_person_count: int = 6,
    log=print,
) -> dict:
    """Write review images, an HTML page, and a review index for checked masks."""
    output_dir = Path(output_dir).resolve()
    reviewed = Path(review_dir).resolve() if review_dir is not None else Path(str(output_dir) + ".review")
    if reviewed == output_dir or output_dir in reviewed.parents:
        raise ValueError("review directory must be outside the replay data directory")
    stride_dir = (
        Path(stride_report_dir).resolve()
        if stride_report_dir is not None
        else Path(str(output_dir) + ".stride_compare")
    )
    rows = load_rows(output_dir)
    stride_rows = _load_stride_rows(stride_dir)
    chosen = select_review_frames(
        rows, stride_rows,
        low_agreement_iou=low_agreement_iou,
        high_agreement_count=high_agreement_count,
        missing_person_count=missing_person_count,
    )
    reviewed.mkdir(parents=True, exist_ok=True)
    images_dir = reviewed / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for entry in chosen:
        row = rows[entry["index"]]
        rgb = None
        if row.get("selected_rgb"):
            rgb = _read_image(output_dir / row["selected_rgb"], cv2.IMREAD_COLOR)
        mask = None
        if row.get("baseline_mask"):
            mask = load_mask_image(output_dir / row["baseline_mask"])
        image_name = None
        if rgb is not None:
            if mask is not None and mask.shape != rgb.shape[:2]:
                mask = None
            view = _view_pair(rgb, mask)
            image_name = f"{entry['frame_key']}_review.png"
            if not cv2.imwrite(str(images_dir / image_name), view):
                image_name = None
        item = {
            **entry,
            "image": None if image_name is None else f"images/{image_name}",
            "rgb": row.get("selected_rgb"),
            "baseline_mask": row.get("baseline_mask"),
            "mask_pose": row.get("mask_pose"),
            "depth": row.get("selected_depth"),
        }
        items.append(item)

    index = {
        "schema_version": LABEL_SCHEMA_VERSION,
        "output_dir": str(output_dir),
        "review_dir": str(reviewed),
        "criteria": {
            "low_agreement_iou_below": low_agreement_iou,
            "high_agreement_count": high_agreement_count,
            "missing_person_count": missing_person_count,
        },
        "rows_total": len(rows),
        "rows_compared_for_stride": len(stride_rows),
        "items": items,
        "note": (
            "baseline_mask is model output, not a checked label. Write checked masks to the "
            "separate label store and score a method against those labels."
        ),
    }
    (reviewed / "review_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    (reviewed / "index.html").write_text(_render_review_html(index), encoding="utf-8")
    log(f"review set written: {reviewed} ({len(items)} frames)")
    return index


def _render_review_html(index: dict) -> str:
    from html import escape

    reasons = {}
    for item in index["items"]:
        for reason in item["reasons"]:
            reasons[reason] = reasons.get(reason, 0) + 1
    lines = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<title>Checked-mask review set</title>",
        "<style>body{font-family:system-ui,sans-serif;background:#14171a;color:#e6e6e6;margin:24px}"
        "table{border-collapse:collapse;margin:12px 0}td,th{border:1px solid #39424a;padding:4px 8px;"
        "font-size:13px;text-align:left}th{background:#20262b}img{max-width:1000px;border:1px solid #39424a}"
        ".note{color:#9aa7b0;max-width:1000px}</style></head><body>",
        "<h1>Checked-mask review set</h1>",
        f"<p class='note'>Replay data: <code>{escape(index['output_dir'])}</code><br>",
        "Left is the selected RGB frame. Right is the saved <code>baseline_mask</code> over the same "
        "frame. The saved mask is model output, not a checked label. Draw the true person mask in "
        "the label store, then run the evaluation.</p>",
        f"<p>frames: {len(index['items'])} | reasons: {escape(json.dumps(reasons))} | "
        f"stride-compared rows in this clip: {index['rows_compared_for_stride']}</p>",
        "<table><tr><th>frame</th><th>time s</th><th>reasons</th><th>status</th>"
        "<th>mask_source</th><th>stride union</th><th>view</th></tr>",
    ]
    for item in index["items"]:
        lines.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>"
            "<td>{}</td></tr>".format(
                escape(str(item["frame_id"])), escape(str(item["timestamp_s"])),
                escape(", ".join(item["reasons"])), escape(str(item["status"])),
                escape(str(item["mask_source"])), escape(str(item["stride_iou"])),
                (
                    f"<a href='{escape(item['image'])}'><img src='{escape(item['image'])}'></a>"
                    if item["image"]
                    else "no image"
                ),
            )
        )
    lines.append("</table></body></html>")
    return "\n".join(lines)


# --------------------------------------------------------- label store


def label_path(labels_root: Path, dataset: str, frame_key_value: str) -> Path:
    """One PNG for each checked frame mask, in a store that is not replay data."""
    return Path(labels_root) / dataset / "masks" / f"{frame_key_value}.png"


def load_review_index(review_dir: Path) -> dict:
    path = Path(review_dir) / "review_index.json"
    if not path.is_file():
        raise FileNotFoundError(f"review index does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_checked_mask(
    labels_root: Path,
    dataset: str,
    frame_key_value: str,
    mask: np.ndarray,
    *,
    source: str,
    notes: str = "",
) -> Path:
    """Write one checked mask and its metadata. The mask is binary, 0 or 255."""
    path = label_path(labels_root, dataset, frame_key_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), (np.asarray(mask) > 0).astype(np.uint8) * 255):
        raise OSError(f"could not write checked mask: {path}")
    metadata = {
        "schema_version": LABEL_SCHEMA_VERSION,
        "dataset": dataset,
        "frame_key": frame_key_value,
        "source": source,
        "notes": notes,
        "person_pixels": int((np.asarray(mask) > 0).sum()),
    }
    (path.parent.parent / "mask_meta" / f"{frame_key_value}.json").parent.mkdir(
        parents=True, exist_ok=True
    )
    (path.parent.parent / "mask_meta" / f"{frame_key_value}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return path


def labelled_frames(labels_root: Path, dataset: str) -> dict[str, np.ndarray]:
    """Load every checked mask for one dataset."""
    directory = Path(labels_root) / dataset / "masks"
    if not directory.is_dir():
        return {}
    masks: dict[str, np.ndarray] = {}
    for path in sorted(directory.glob("*.png")):
        mask = load_mask_image(path)
        if mask is not None:
            masks[path.stem] = mask
    return masks


# ------------------------------------------------------------ evaluation


def _method_mask(
    strategy: str,
    row: dict,
    *,
    joints: list[dict] | None,
    depth: np.ndarray | None,
    baseline: np.ndarray | None,
    config: DepthThresholdConfig,
) -> tuple[np.ndarray | None, dict]:
    """Return the mask for one method and a short metadata record."""
    if depth is None or joints is None:
        return None, {"error": "missing depth or joints"}
    if strategy == "baseline":
        return baseline, {"stride": None}
    if strategy.startswith("stride"):
        stride = int(strategy.removeprefix("stride"))
        selection = select_pose_gradient_mask(
            depth, pose_joints=joints, config=config, gradient_stride=stride
        )
        return np.asarray(selection.info.mask) > 0, {
            "stride": stride,
            "source": selection.source,
            "fallback_reason": selection.fallback_reason,
        }
    raise ValueError(f"unknown strategy: {strategy}")


def compare_to_label(
    mask: np.ndarray | None,
    label: np.ndarray,
    *,
    joints: list[dict] | None,
    regions: dict[str, tuple[int, int, int, int]],
    rgb: np.ndarray | None,
    pad_px: int,
    background_min_pixels: int = 200,
) -> dict:
    """Person coverage, background leakage, and region metrics for one mask.

    Person coverage is the part of the checked person that the mask includes.
    Background leakage is the part of the mask that falls outside the checked
    person, with a largest-connected-region size so a speckle is not mixed with a
    solid false region. A mask can contain a nearby object that touches the
    person; that object appears as a large leakage component.
    """
    label = np.asarray(label, dtype=bool)
    label_pixels = int(label.sum())
    result: dict[str, Any] = {
        "label_pixels": label_pixels,
        "mask_pixels": None,
        "person_coverage": None,
        "background_leakage": None,
        "iou": None,
        "largest_leak_px": None,
        "regions": {},
    }
    if mask is None:
        result["error"] = "no mask for this method"
        return result
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != label.shape:
        result["error"] = "mask and label size differ"
        return result
    mask_pixels = int(mask.sum())
    result["mask_pixels"] = mask_pixels
    if label_pixels:
        result["person_coverage"] = round(float(np.logical_and(mask, label).sum()) / label_pixels, 4)
    leak = np.logical_and(mask, ~label)
    leak_pixels = int(leak.sum())
    result["background_leakage"] = round(float(leak_pixels) / label_pixels, 4) if label_pixels else None
    result["iou"] = mask_iou(mask, label)
    if leak_pixels:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(leak.astype(np.uint8), connectivity=8)
        if count > 1:
            result["largest_leak_px"] = int(stats[1:, cv2.CC_STAT_AREA].max())
    else:
        result["largest_leak_px"] = 0
    # Hand and head regions are judged by how much of the checked person inside
    # each region the method keeps.
    for name, group in ((name, group) for name, group in regions):
        if joints is None:
            continue
        height, width = label.shape
        points = _visible_pixels(joints, group, width=width, height=height)
        if len(points) < 3:
            result["regions"][name] = None
            continue
        box = region_box(points, width=width, height=height, pad_px=pad_px)
        if box is None:
            result["regions"][name] = None
            continue
        left, right, top, bottom = box
        window_label = label[top : bottom + 1, left : right + 1]
        window_mask = mask[top : bottom + 1, left : right + 1]
        label_in_region = int(window_label.sum())
        result["regions"][name] = {
            "label_pixels": label_in_region,
            "kept_fraction": (
                None
                if label_in_region == 0
                else round(float(np.logical_and(window_mask, window_label).sum()) / label_in_region, 4)
            ),
            "leak_pixels": int(np.logical_and(window_mask, ~window_label).sum()),
        }
    return result


def evaluate_labelled(
    output_dir: Path,
    labels_root: Path,
    dataset: str,
    *,
    report_dir: Path | None = None,
    pad_px: int = 2,
    log=print,
) -> dict:
    """Compare the live mask and the stride variants against checked masks."""
    output_dir = Path(output_dir).resolve()
    reviewed = Path(str(output_dir) + ".eval")
    resolved_report = Path(report_dir).resolve() if report_dir is not None else reviewed
    if resolved_report == output_dir or output_dir in resolved_report.parents:
        raise ValueError("report directory must be outside the replay data directory")
    rows = load_rows(output_dir)
    by_key = {frame_key(row): row for row in rows}
    masks = labelled_frames(labels_root, dataset)
    if not masks:
        raise FileNotFoundError(
            f"no checked masks in {Path(labels_root) / dataset / 'masks'}; label the review set first"
        )
    config = DepthThresholdConfig()
    records = []
    methods = [("baseline", "baseline")] + [(f"stride{s}", f"stride{s}") for s in COMPARISON_STRIDES]
    for key, label in masks.items():
        row = by_key.get(key)
        record: dict[str, Any] = {"frame_key": key, "label_source": "checked"}
        if row is None:
            record["error"] = "checked mask has no replay row"
            records.append(record)
            continue
        joints, joint_source = load_mask_joints(output_dir, row)
        depth = load_depth(output_dir, row)
        baseline = (
            load_mask_image(output_dir / row["baseline_mask"]) if row.get("baseline_mask") else None
        )
        rgb = _read_image(output_dir / row["selected_rgb"], cv2.IMREAD_COLOR) if row.get("selected_rgb") else None
        record.update(
            {
                "frame_id": row.get("selected_frame_id"),
                "timestamp_s": round(float(row.get("selected_timestamp_ns", 0)) / 1e9, 3),
                "mask_source": row.get("mask_source"),
                "fallback_reason": row.get("fallback_reason"),
                "joint_record": joint_source,
                "has_depth": depth is not None,
            }
        )
        # Check the head first because it is the
        # reported weak region; hands are added when the joints are visible.
        region_groups = (
            ("head", tuple(range(0, 13))),
            ("left_hand", (15, 17, 19, 21)),
            ("right_hand", (16, 18, 20, 22)),
        )
        for name, strategy in methods:
            strategy_started = time.perf_counter()
            method_mask, meta = _method_mask(
                strategy, row, joints=joints, depth=depth, baseline=baseline, config=config
            )
            elapsed_ms = (time.perf_counter() - strategy_started) * 1000.0
            record[name] = {
                **meta,
                "timing_ms": round(elapsed_ms, 2) if strategy != "baseline" else None,
                **compare_to_label(
                    method_mask, label, joints=joints, regions=region_groups, rgb=rgb, pad_px=pad_px
                ),
            }
        records.append(record)

    report = {
        "schema_version": LABEL_SCHEMA_VERSION,
        "output_dir": str(output_dir),
        "labels_root": str(Path(labels_root)),
        "dataset": dataset,
        "frames": len(records),
        "live_stride": LIVE_STRIDE,
        "summary": _evaluate_summary(records),
        "records": records,
    }
    resolved_report.mkdir(parents=True, exist_ok=True)
    (resolved_report / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (resolved_report / "index.html").write_text(_render_eval_html(report), encoding="utf-8")
    log(f"evaluation written: {resolved_report}")
    return report


def _evaluate_summary(records: list[dict]) -> dict:
    valid = [record for record in records if "error" not in record]
    summary: dict[str, Any] = {"frames_scored": len(valid), "errors": len(records) - len(valid), "methods": {}}
    names = [name for name in (valid[0] if valid else {}) if name.startswith(("baseline", "stride"))]
    for name in names:
        rows = [record[name] for record in valid if record.get(name, {}).get("mask_pixels") is not None]
        coverage = [row["person_coverage"] for row in rows if row.get("person_coverage") is not None]
        leakage = [row["background_leakage"] for row in rows if row.get("background_leakage") is not None]
        iou = [row["iou"] for row in rows if row.get("iou") is not None]
        largest = [row["largest_leak_px"] for row in rows if row.get("largest_leak_px") is not None]
        timings = [row["timing_ms"] for row in rows if row.get("timing_ms") is not None]
        head = [
            row["regions"]["head"]["kept_fraction"]
            for row in rows
            if row.get("regions", {}).get("head") and row["regions"]["head"].get("kept_fraction") is not None
        ]
        summary["methods"][name] = {
            "frames": len(rows),
            "person_coverage": _summary(coverage),
            "background_leakage": _summary(leakage),
            "iou": _summary(iou),
            "largest_leak_px": _summary([float(value) for value in largest]),
            "head_kept_fraction": _summary(head),
            "timing_ms": _timing_summary(timings),
        }
    return summary


def _timing_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"frames": 0, "mean": None, "median": None, "max": None}
    return {
        "frames": len(values),
        "mean": round(statistics.fmean(values), 2),
        "median": round(statistics.median(values), 2),
        "max": round(max(values), 2),
    }


def _render_eval_html(report: dict) -> str:
    from html import escape

    lines = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<title>Checked-mask evaluation</title>",
        "<style>body{font-family:system-ui,sans-serif;background:#14171a;color:#e6e6e6;margin:24px}"
        "table{border-collapse:collapse;margin:12px 0}td,th{border:1px solid #39424a;padding:4px 8px;"
        "font-size:13px;text-align:left}th{background:#20262b}"
        ".note{color:#9aa7b0;max-width:1000px}</style></head><body>",
        "<h1>Checked-mask evaluation</h1>",
        f"<p class='note'>Replay data: <code>{escape(report['output_dir'])}</code><br>",
        f"Labels: <code>{escape(report['labels_root'])}</code> dataset "
        f"<code>{escape(report['dataset'])}</code><br>",
        "Person coverage is the part of the checked person the mask includes. Background leakage "
        "is mask area outside the checked person, relative to checked person area. Hand regions are "
        "judged against checked person pixels inside the region.</p>",
        f"<p>frames scored: {report['summary']['frames_scored']} | errors: "
        f"{report['summary']['errors']}</p>",
        "<h2>Methods</h2><table><tr><th>method</th><th>frames</th><th>coverage p05/p50/p95</th>"
        "<th>leakage p05/p50/p95</th><th>union p50</th><th>largest leak px p50</th>"
        "<th>head kept p50</th><th>time ms median</th></tr>",
    ]
    for name, entry in report["summary"]["methods"].items():
        coverage = entry["person_coverage"]
        leakage = entry["background_leakage"]
        lines.append(
            "<tr><td>{}</td><td>{}</td><td>{}/{}/{}</td><td>{}/{}/{}</td><td>{}</td>"
            "<td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(name), entry["frames"],
                coverage["p05"], coverage["p50"], coverage["p95"],
                leakage["p05"], leakage["p50"], leakage["p95"],
                entry["iou"]["p50"], entry["largest_leak_px"]["p50"],
                entry["head_kept_fraction"]["p50"], entry["timing_ms"]["median"],
            )
        )
    lines.append("</table><h2>Frames</h2><table><tr><th>frame</th><th>time s</th>"
                 "<th>label px</th><th>baseline coverage/leak</th><th>stride1 coverage/leak</th>"
                 "<th>stride2 coverage/leak</th><th>stride4 coverage/leak</th></tr>")
    for record in report["records"]:
        if "error" in record:
            lines.append(f"<tr><td colspan='7'>{escape(record['frame_key'])}: {escape(record['error'])}</td></tr>")
            continue
        label_px = record.get("baseline", {}).get("label_pixels")
        cells = []
        for name in ("baseline", "stride1", "stride2", "stride4"):
            entry = record.get(name) or {}
            cells.append("{}/{}".format(entry.get("person_coverage"), entry.get("background_leakage")))
        lines.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(str(record.get("frame_id"))), escape(str(record.get("timestamp_s"))),
                escape(str(label_px)), *[escape(cell) for cell in cells],
            )
        )
    lines.append("</table></body></html>")
    return "\n".join(lines)


# ------------------------------------------------------- training filter


def training_eligibility(
    output_dir: Path,
    *,
    labels_root: Path | None = None,
    dataset: str | None = None,
    log=print,
) -> dict:
    """Decide which replay pairs may enter the first 15-channel set.

    A pair needs a complete reference and selected frame, flow,
    the selected depth, the exact ``mask_pose``, a successful selected row, and a
    checked person-mask label. Failed rows, the bootstrap row, and the no-pose
    fallback rows are excluded with their own reasons. ``baseline_mask`` is an
    input, never a label, so a pair is eligible only with a checked mask.
    """
    output_dir = Path(output_dir).resolve()
    rows = load_rows(output_dir)
    labels = labelled_frames(labels_root, dataset) if labels_root is not None and dataset else {}
    reasons: dict[str, int] = {}
    eligible = []
    for index, row in enumerate(rows):
        key = frame_key(row)
        missing: list[str] = []
        if row.get("status") != "ok":
            missing.append(f"selected_status_{row.get('status')}")
        if row.get("action") == "bootstrap" or row.get("reference_frame_id") is None:
            missing.append("no_reference")
        if row.get("mask_source") == "depth_threshold_fallback":
            missing.append("no_pose_fallback_mask")
        if not row.get("mask_pose"):
            missing.append("no_exact_mask_pose")
        if not row.get("selected_depth"):
            missing.append("no_selected_depth")
        elif not (output_dir / row["selected_depth"]).is_file():
            missing.append("selected_depth_file_missing")
        if row.get("action") == "refresh":
            if not row.get("flow"):
                missing.append("refresh_row_has_no_flow")
        if labels_root is None or dataset is None:
            missing.append("no_label_store")
        elif key not in labels:
            missing.append("no_checked_mask")
        if missing:
            for reason in missing:
                reasons[reason] = reasons.get(reason, 0) + 1
            continue
        eligible.append(
            {
                "index": index,
                "frame_key": key,
                "frame_id": row.get("selected_frame_id"),
                "timestamp_s": round(float(row.get("selected_timestamp_ns", 0)) / 1e9, 3),
                "reference_frame_id": row.get("reference_frame_id"),
                "label": f"{dataset}/masks/{key}.png",
            }
        )
    report = {
        "schema_version": LABEL_SCHEMA_VERSION,
        "output_dir": str(output_dir),
        "labels_root": None if labels_root is None else str(Path(labels_root)),
        "dataset": dataset,
        "rows_total": len(rows),
        "eligible_rows": len(eligible),
        "excluded_rows": len(rows) - len(eligible),
        "exclusion_reasons": dict(sorted(reasons.items())),
        "eligible": eligible,
        "note": (
            "A row can have more than one exclusion reason, so the reason counts do not add up "
            "to excluded_rows. baseline_mask is an input and is never a label."
        ),
    }
    path = Path(str(output_dir) + ".training_filter.json")
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(
        "training filter: eligible {} of {} rows; reasons {}".format(
            report["eligible_rows"], report["rows_total"], report["exclusion_reasons"]
        )
    )
    return report


# ------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    review = sub.add_parser("review-set", help="write the checked-mask review set")
    review.add_argument("--output-dir", type=Path, required=True)
    review.add_argument("--review-dir", type=Path, default=None)
    review.add_argument("--stride-report-dir", type=Path, default=None)
    review.add_argument("--low-agreement-iou", type=float, default=LOW_AGREEMENT_IOU)
    review.add_argument("--high-agreement-count", type=int, default=12)
    review.add_argument("--missing-person-count", type=int, default=6)

    evaluate = sub.add_parser("evaluate", help="score methods against checked masks")
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--labels-root", type=Path, default=None)
    evaluate.add_argument("--dataset", default=None)
    evaluate.add_argument("--report-dir", type=Path, default=None)

    training = sub.add_parser("training-filter", help="list training-eligible pairs")
    training.add_argument("--output-dir", type=Path, required=True)
    training.add_argument("--labels-root", type=Path, default=None)
    training.add_argument("--dataset", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "review-set":
        build_review_set(
            args.output_dir,
            review_dir=args.review_dir,
            stride_report_dir=args.stride_report_dir,
            low_agreement_iou=args.low_agreement_iou,
            high_agreement_count=args.high_agreement_count,
            missing_person_count=args.missing_person_count,
        )
    elif args.command == "evaluate":
        if args.dataset is None:
            parser.error("evaluate needs --dataset")
        labels_root = args.labels_root or (args.output_dir.resolve().parent.parent / "labels")
        evaluate_labelled(
            args.output_dir, labels_root, args.dataset, report_dir=args.report_dir
        )
    else:
        labels_root = args.labels_root
        if labels_root is None and args.dataset is not None:
            labels_root = args.output_dir.resolve().parent.parent / "labels"
        training_eligibility(args.output_dir, labels_root=labels_root, dataset=args.dataset)


if __name__ == "__main__":
    main()
