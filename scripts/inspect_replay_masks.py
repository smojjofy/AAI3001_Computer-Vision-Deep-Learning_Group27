"""Inspect one replay output: overlays, mask sources, and joint-region coverage.

Example:
    python -m scripts.inspect_replay_masks \
        --output-dir data/processed/learning_replay/person01/clip01

The tool reads replay data and never changes it. It writes one RGB and mask
overlay for each usable row, one HTML index page, and one machine-readable
``report.json`` to a separate report directory. Joint-region coverage is a
diagnostic proxy, not an accuracy result: a box around a joint group includes
background by design, and human review of the paired images remains necessary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import html
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from applications.live_pipeline.pose_gradient_mask import (  # noqa: E402
    HEAD_JOINTS,
    LEFT_HAND_JOINTS,
    MIN_JOINT_VISIBILITY,
    RIGHT_HAND_JOINTS,
)
from applications.live_pipeline.telemetry import percentile  # noqa: E402

# A convex hull needs three points. Use the same minimum for a region box so a
# single stray landmark cannot define a region.
MIN_REGION_JOINTS = 3
REGION_JOINT_SETS = (
    ("head", HEAD_JOINTS),
    ("left_hand", LEFT_HAND_JOINTS),
    ("right_hand", RIGHT_HAND_JOINTS),
)
REGION_COLORS = {"head": (0, 0, 255), "left_hand": (255, 128, 0), "right_hand": (0, 128, 255)}


@dataclass
class RegionResult:
    """Coverage of one joint-derived region for one row."""

    coverage: float
    joint_coverage: float
    width_px: int
    height_px: int
    visible_joints: int


@dataclass
class RowResult:
    """Everything measured for one replay row."""

    index: int
    frame_id: int | None
    status: str | None
    error: str | None
    mask_source: str | None
    fallback_reason: str | None
    mask_area_fraction: float | None
    mask_present: bool
    pose_joint_source: str
    exact_mask_joints: bool
    usable_pose: bool
    # The replay's own ``pose_detected`` value: True, False, or None when absent.
    pose_detected: bool | None = None
    overlay: str | None = None
    problems: list[str] = field(default_factory=list)
    regions: dict[str, RegionResult | None] = field(default_factory=dict)
    excluded: dict[str, str] = field(default_factory=dict)


def load_rows(output_dir: Path) -> list[dict]:
    rows_path = output_dir / "rows.jsonl"
    if not rows_path.is_file():
        raise FileNotFoundError(f"replay rows file does not exist: {rows_path}")
    rows: list[dict] = []
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _read_image(path: Path, flags: int) -> np.ndarray | None:
    if not path.is_file():
        return None
    image = cv2.imread(str(path), flags)
    if image is None:
        return None
    return image


def _visible_pixels(
    joints: list[dict] | None,
    group: tuple[int, ...],
    *,
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    """Joint centres of one group that clear the visibility floor and the image."""
    points: list[tuple[int, int]] = []
    for joint in joints or []:
        index = joint.get("index")
        if index is None or int(index) not in group:
            continue
        visibility = joint.get("visibility")
        if visibility is None or float(visibility) < MIN_JOINT_VISIBILITY:
            continue
        x = joint.get("x")
        y = joint.get("y")
        if x is None or y is None:
            continue
        px = int(round(float(x) * (width - 1)))
        py = int(round(float(y) * (height - 1)))
        if 0 <= px < width and 0 <= py < height:
            points.append((px, py))
    return points


def region_box(
    points: list[tuple[int, int]],
    *,
    width: int,
    height: int,
    pad_px: int,
) -> tuple[int, int, int, int] | None:
    """Axis-aligned bounding box around the visible joints, clipped to the image."""
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    left = max(0, min(xs) - pad_px)
    right = min(width - 1, max(xs) + pad_px)
    top = max(0, min(ys) - pad_px)
    bottom = min(height - 1, max(ys) + pad_px)
    if right < left or bottom < top:
        return None
    return left, right, top, bottom


def measure_region(
    joints: list[dict] | None,
    group: tuple[int, ...],
    mask: np.ndarray,
    *,
    pad_px: int,
) -> RegionResult | None:
    """Coverage of the mask inside a box around the visible joints of one group."""
    height, width = mask.shape
    points = _visible_pixels(joints, group, width=width, height=height)
    if len(points) < MIN_REGION_JOINTS:
        return None
    box = region_box(points, width=width, height=height, pad_px=pad_px)
    if box is None:
        return None
    left, right, top, bottom = box
    window = mask[top : bottom + 1, left : right + 1]
    if window.size == 0:
        return None
    joint_values = [float(mask[py, px]) for px, py in points]
    return RegionResult(
        coverage=float(window.mean()),
        joint_coverage=float(np.mean(joint_values)),
        width_px=right - left + 1,
        height_px=bottom - top + 1,
        visible_joints=len(points),
    )


def load_mask_joints(output_dir: Path, row: dict) -> tuple[list[dict] | None, str]:
    """Return the joints mask selection used, and where they came from.

    Newer rows store the exact unrounded joints in ``mask_pose``. Older rows only
    have the display joints in ``selected_pose``, which are rounded.
    """
    for key, source in (("mask_pose", "mask_pose"), ("selected_pose", "selected_pose_rounded")):
        path = row.get(key)
        if not path:
            continue
        candidate = output_dir / path
        if not candidate.is_file():
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        joints = payload.get("joints")
        if isinstance(joints, list) and joints:
            return joints, source
    return None, "absent"


def _usable_pose(joints: list[dict] | None) -> bool:
    for joint in joints or []:
        visibility = joint.get("visibility")
        if visibility is not None and float(visibility) >= MIN_JOINT_VISIBILITY:
            return True
    return False


def _pose_detected_value(row: dict) -> bool | None:
    """Return the replay ``pose_detected`` value as a boolean, or None.

    A null value means the field is absent or unknown. It is not False.
    """
    value = row.get("pose_detected")
    if value is None:
        return None
    return bool(value)


def _draw_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    boxes: dict[str, tuple[int, int, int, int]],
) -> np.ndarray:
    output = rgb.copy()
    inside = mask.astype(bool)
    green = np.array([0, 200, 0], dtype=np.float32)
    output[inside] = (0.55 * output[inside].astype(np.float32) + 0.45 * green).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (0, 255, 0), 1)
    for name, (left, right, top, bottom) in boxes.items():
        cv2.rectangle(output, (left, top), (right, bottom), REGION_COLORS[name], 1)
        cv2.putText(
            output, name, (left, max(8, top - 3)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, REGION_COLORS[name], 1, cv2.LINE_AA,
        )
    return output


def inspect_row(
    output_dir: Path,
    row: dict,
    index: int,
    *,
    report_dir: Path,
    pad_px: int,
    write_overlay: bool = True,
) -> RowResult:
    """Measure one replay row. Missing files are reported, never counted as zero."""
    generation = row.get("generation")
    frame_id = row.get("selected_frame_id")
    key = f"{index:04d}_g{generation}_f{int(frame_id):08d}" if frame_id is not None else f"{index:04d}"
    result = RowResult(
        index=index,
        frame_id=frame_id,
        status=row.get("status"),
        error=row.get("error"),
        mask_source=row.get("mask_source"),
        fallback_reason=row.get("fallback_reason"),
        mask_area_fraction=row.get("mask_area_fraction"),
        mask_present=False,
        pose_joint_source="absent",
        exact_mask_joints=False,
        usable_pose=False,
        pose_detected=_pose_detected_value(row),
    )
    joints, joint_source = load_mask_joints(output_dir, row)
    result.pose_joint_source = joint_source
    result.exact_mask_joints = joint_source == "mask_pose"
    result.usable_pose = _usable_pose(joints)

    rgb: np.ndarray | None = None
    rgb_path = row.get("selected_rgb")
    if rgb_path:
        rgb = _read_image(output_dir / rgb_path, cv2.IMREAD_COLOR)
        if rgb is None:
            result.problems.append(f"selected_rgb missing or unreadable: {rgb_path}")

    mask: np.ndarray | None = None
    mask_path = row.get("baseline_mask")
    if mask_path:
        loaded = _read_image(output_dir / mask_path, cv2.IMREAD_GRAYSCALE)
        if loaded is None:
            result.problems.append(f"baseline_mask missing or unreadable: {mask_path}")
        else:
            mask = loaded > 0
            result.mask_present = True
    if mask is not None and rgb is not None and mask.shape != rgb.shape[:2]:
        result.problems.append(
            "baseline_mask and selected_rgb size differ: "
            f"{mask.shape[1]}x{mask.shape[0]} against {rgb.shape[1]}x{rgb.shape[0]}"
        )
        mask_for_regions: np.ndarray | None = None
    else:
        mask_for_regions = mask

    boxes: dict[str, tuple[int, int, int, int]] = {}
    for name, group in REGION_JOINT_SETS:
        result.regions[name] = None
        if mask_for_regions is None:
            if not mask_path:
                result.excluded[name] = "no_mask"
            elif not result.mask_present:
                result.excluded[name] = "missing_file"
            else:
                result.excluded[name] = "size_mismatch"
            continue
        if joints is None:
            result.excluded[name] = "no_joint_record"
            continue
        height, width = mask_for_regions.shape
        points = _visible_pixels(joints, group, width=width, height=height)
        if len(points) < MIN_REGION_JOINTS:
            result.excluded[name] = "insufficient_visible_joints"
            continue
        box = region_box(points, width=width, height=height, pad_px=pad_px)
        if box is None:
            result.excluded[name] = "insufficient_visible_joints"
            continue
        region = measure_region(joints, group, mask_for_regions, pad_px=pad_px)
        if region is None:
            result.excluded[name] = "insufficient_visible_joints"
            continue
        result.regions[name] = region
        boxes[name] = box

    if write_overlay and rgb is not None and mask_for_regions is not None:
        overlay = _draw_overlay(rgb, mask_for_regions, boxes)
        overlay_dir = report_dir / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = overlay_dir / f"{key}_overlay.png"
        if not cv2.imwrite(str(overlay_path), overlay):
            result.problems.append(f"could not write overlay: {overlay_path}")
        else:
            result.overlay = f"overlays/{overlay_path.name}"
    return result


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"rows": 0, "p05": None, "p50": None, "p95": None}
    return {
        "rows": len(values),
        "p05": round(percentile(values, 5.0), 4),
        "p50": round(percentile(values, 50.0), 4),
        "p95": round(percentile(values, 95.0), 4),
    }


def build_summary(rows: list[RowResult], *, pad_px: int) -> dict:
    """Aggregate the row results with separate denominators for every reason."""
    by_source: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (str(row.mask_source), str(row.fallback_reason))
        entry = by_source.setdefault(
            key,
            {
                "mask_source": row.mask_source,
                "fallback_reason": row.fallback_reason,
                "rows": 0,
                "person_missing_rows": 0,
                "error_rows": 0,
                "pose_detected_false_rows": 0,
                "pose_detected_null_rows": 0,
                "mask_area_fractions": [],
            },
        )
        entry["rows"] += 1
        if row.status == "person_missing":
            entry["person_missing_rows"] += 1
        if row.status == "error":
            entry["error_rows"] += 1
        # These two counters read the replay's own pose_detected value. A null
        # value is counted separately and is never treated as False.
        if row.pose_detected is False:
            entry["pose_detected_false_rows"] += 1
        elif row.pose_detected is None:
            entry["pose_detected_null_rows"] += 1
        if row.mask_area_fraction is not None:
            entry["mask_area_fractions"].append(float(row.mask_area_fraction))

    source_rows = []
    for entry in by_source.values():
        areas = entry.pop("mask_area_fractions")
        entry["mask_area"] = _summary(areas)
        entry["rows"] = int(entry["rows"])
        source_rows.append(entry)
    source_rows.sort(key=lambda item: (str(item["mask_source"]), str(item["fallback_reason"])))

    regions: dict[str, dict] = {}
    for name, _group in REGION_JOINT_SETS:
        coverages = [
            row.regions[name].coverage
            for row in rows
            if row.regions.get(name) is not None
        ]
        joint_coverages = [
            row.regions[name].joint_coverage
            for row in rows
            if row.regions.get(name) is not None
        ]
        exclusions: dict[str, int] = {}
        for row in rows:
            reason = row.excluded.get(name)
            if reason is not None:
                exclusions[reason] = exclusions.get(reason, 0) + 1
        regions[name] = {
            "joint_set": list(dict(REGION_JOINT_SETS)[name]),
            "denominator_rows": len(coverages),
            "coverage": _summary(coverages),
            "joint_center_coverage": _summary(joint_coverages),
            "excluded": dict(sorted(exclusions.items())),
        }

    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[str(row.status)] = status_counts.get(str(row.status), 0) + 1
    joint_sources: dict[str, int] = {}
    for row in rows:
        joint_sources[row.pose_joint_source] = joint_sources.get(row.pose_joint_source, 0) + 1

    return {
        "schema_version": "1",
        "visibility_floor": MIN_JOINT_VISIBILITY,
        "region_pad_px": pad_px,
        "min_region_joints": MIN_REGION_JOINTS,
        "region_joint_sets": {name: list(group) for name, group in REGION_JOINT_SETS},
        "rows_total": len(rows),
        "row_status": dict(sorted(status_counts.items())),
        "mask_presence": {
            "present": sum(1 for row in rows if row.mask_present),
            "absent": sum(1 for row in rows if not row.mask_present),
        },
        "pose_joint_record": dict(sorted(joint_sources.items())),
        "pose_detected_values": {
            "true": sum(1 for row in rows if row.pose_detected is True),
            "false": sum(1 for row in rows if row.pose_detected is False),
            "null": sum(1 for row in rows if row.pose_detected is None),
        },
        "rows_with_usable_pose": sum(1 for row in rows if row.usable_pose),
        "rows_with_problems": sum(1 for row in rows if row.problems),
        "by_mask_source": source_rows,
        "regions": regions,
    }


def _row_payload(row: RowResult) -> dict:
    regions = {}
    for name, _group in REGION_JOINT_SETS:
        region = row.regions.get(name)
        regions[name] = (
            None
            if region is None
            else {
                "coverage": round(region.coverage, 4),
                "joint_center_coverage": round(region.joint_coverage, 4),
                "width_px": region.width_px,
                "height_px": region.height_px,
                "visible_joints": region.visible_joints,
            }
        )
    return {
        "index": row.index,
        "frame_id": row.frame_id,
        "status": row.status,
        "error": row.error,
        "mask_source": row.mask_source,
        "fallback_reason": row.fallback_reason,
        "mask_area_fraction": row.mask_area_fraction,
        "mask_present": row.mask_present,
        "pose_joint_source": row.pose_joint_source,
        "exact_mask_joints": row.exact_mask_joints,
        "usable_pose": row.usable_pose,
        "pose_detected": row.pose_detected,
        "overlay": row.overlay,
        "problems": row.problems,
        "excluded": row.excluded,
        "regions": regions,
    }


def _render_html(report: dict) -> str:
    escape = html.escape
    summary = report["summary"]
    lines = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<title>Replay mask inspection</title>",
        "<style>body{font-family:system-ui,sans-serif;background:#14171a;color:#e6e6e6;margin:24px}"
        "table{border-collapse:collapse;margin:12px 0}td,th{border:1px solid #39424a;padding:4px 8px;"
        "font-size:13px;text-align:left}th{background:#20262b}img{max-width:360px;border:1px solid #39424a}"
        ".warn{color:#ffb454}.note{color:#9aa7b0;max-width:900px}</style></head><body>",
        "<h1>Replay mask inspection</h1>",
        f"<p class='note'>Replay data: <code>{escape(report['output_dir'])}</code><br>",
        "Joint-region coverage is a diagnostic proxy. A joint-derived box includes background by "
        "design, so coverage cannot establish face or hand accuracy. Review the overlay images.</p>",
        f"<p>Rows: {summary['rows_total']} | visible-pose rows: {summary['rows_with_usable_pose']} | "
        f"rows with problems: {summary['rows_with_problems']} | visibility floor: "
        f"{summary['visibility_floor']} | region pad: {summary['region_pad_px']} px | "
        f"minimum region joints: {summary['min_region_joints']}</p>",
        f"<p>Replay <code>pose_detected</code> values: {escape(json.dumps(summary['pose_detected_values']))}. "
        "A null value is unknown, not False.</p>",
        "<h2>Mask sources</h2>",
        "<table><tr><th>mask_source</th><th>fallback_reason</th><th>rows</th>"
        "<th>area p05</th><th>area p50</th><th>area p95</th><th>person_missing</th>"
        "<th>error</th><th>pose_detected false</th><th>pose_detected null</th></tr>",
    ]
    for entry in summary["by_mask_source"]:
        area = entry["mask_area"]
        lines.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>"
            "<td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(str(entry["mask_source"])), escape(str(entry["fallback_reason"])),
                entry["rows"], area["p05"], area["p50"], area["p95"],
                entry["person_missing_rows"], entry["error_rows"],
                entry["pose_detected_false_rows"], entry["pose_detected_null_rows"],
            )
        )
    lines.append("</table>")
    lines.append("<h2>Joint-region coverage</h2>")
    lines.append(
        "<table><tr><th>region</th><th>joints</th><th>rows</th><th>coverage p05</th>"
        "<th>coverage p50</th><th>coverage p95</th><th>joint-centre p50</th>"
        "<th>excluded</th></tr>"
    )
    for name, region in report["summary"]["regions"].items():
        lines.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>"
            "<td>{}</td><td>{}</td></tr>".format(
                escape(name), escape(str(region["joint_set"])), region["denominator_rows"],
                region["coverage"]["p05"], region["coverage"]["p50"], region["coverage"]["p95"],
                region["joint_center_coverage"]["p50"],
                escape(json.dumps(region["excluded"])),
            )
        )
    lines.append("</table>")
    lines.append("<h2>Rows</h2>")
    lines.append(
        "<table><tr><th>frame</th><th>status</th><th>mask_source</th><th>fallback</th>"
        "<th>area</th><th>joints</th><th>head</th><th>left hand</th><th>right hand</th>"
        "<th>problems</th><th>overlay</th></tr>"
    )
    for row in report["rows"]:
        regions = row["regions"]
        lines.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>"
            "<td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(str(row["frame_id"])),
                escape(str(row["status"])),
                escape(str(row["mask_source"])),
                escape(str(row["fallback_reason"])),
                escape(str(row["mask_area_fraction"])),
                escape(row["pose_joint_source"]),
                _cell(regions["head"]), _cell(regions["left_hand"]), _cell(regions["right_hand"]),
                escape("; ".join(row["problems"])),
                (
                    f"<a href='{escape(row['overlay'])}'><img src='{escape(row['overlay'])}'></a>"
                    if row["overlay"]
                    else ""
                ),
            )
        )
    lines.append("</table></body></html>")
    return "\n".join(lines)


def _cell(region: dict | None) -> str:
    if region is None:
        return ""
    return "{} ({}/{} px)".format(
        region["coverage"], region["width_px"], region["height_px"]
    )


def inspect_replay(
    output_dir: Path,
    *,
    report_dir: Path | None = None,
    region_pad_px: int = 2,
    write_overlays: bool = True,
    log=print,
) -> dict:
    """Read one replay output and write overlays, an HTML page, and report.json."""
    output_dir = Path(output_dir).resolve()
    if region_pad_px < 0:
        raise ValueError("region pad must be non-negative")
    resolved_report = (
        Path(report_dir).resolve()
        if report_dir is not None
        else Path(str(output_dir) + ".mask_report")
    )
    if resolved_report == output_dir or output_dir in resolved_report.parents:
        raise ValueError("report directory must be outside the replay data directory")
    rows = load_rows(output_dir)
    resolved_report.mkdir(parents=True, exist_ok=True)
    results = [
        inspect_row(
            output_dir, row, index,
            report_dir=resolved_report, pad_px=region_pad_px, write_overlay=write_overlays,
        )
        for index, row in enumerate(rows)
    ]
    report = {
        "schema_version": "1",
        "output_dir": str(output_dir),
        "report_dir": str(resolved_report),
        "summary": build_summary(results, pad_px=region_pad_px),
        "rows": [_row_payload(row) for row in results],
    }
    (resolved_report / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (resolved_report / "index.html").write_text(_render_html(report), encoding="utf-8")
    log(f"report written: {resolved_report}")
    return report


def print_report(report: dict, log=print) -> None:
    summary = report["summary"]
    log(f"rows: {summary['rows_total']}  status: {summary['row_status']}")
    log(f"mask present: {summary['mask_presence']}")
    log(f"joint records: {summary['pose_joint_record']}")
    log(f"pose_detected values: {summary['pose_detected_values']}")
    for entry in summary["by_mask_source"]:
        area = entry["mask_area"]
        log(
            "source={} fallback={} rows={} area p05/p50/p95={}/{}/{} "
            "pose_detected false/null={}/{}".format(
                entry["mask_source"], entry["fallback_reason"], entry["rows"],
                area["p05"], area["p50"], area["p95"],
                entry["pose_detected_false_rows"], entry["pose_detected_null_rows"],
            )
        )
    for name, region in summary["regions"].items():
        log(
            "region={} rows={} coverage p05/p50/p95={}/{}/{} excluded={}".format(
                name, region["denominator_rows"],
                region["coverage"]["p05"], region["coverage"]["p50"],
                region["coverage"]["p95"], region["excluded"],
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument(
        "--region-pad-px", type=int, default=2,
        help="pixels added around a joint-derived box; the default is 2",
    )
    parser.add_argument("--no-overlays", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    report = inspect_replay(
        args.output_dir,
        report_dir=args.report_dir,
        region_pad_px=args.region_pad_px,
        write_overlays=not args.no_overlays,
    )
    print_report(report)


if __name__ == "__main__":
    main()
