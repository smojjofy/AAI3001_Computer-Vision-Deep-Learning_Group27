"""Compare the stride-1 and stride-4 mask on the same saved frame.

Example:
    python -m scripts.compare_mask_strides \
        --output-dir data/processed/learning_replay/person01/clip01

This is the same-frame test for the stride-4 gradient sub-sampling decision. For
each replay row that has a saved depth plane and the exact mask-input joints, the
tool runs the live mask selector twice on that one frame: once with the live
default stride 4 and once with stride 1. It compares source-size mask pixels,
fallback reasons, and head and hand region coverage, and it writes a paired RGB
overlay for each compared row. The selector is not patched between the runs.

The result is evidence about the stride estimate only. It is not a mask accuracy
result, and it needs real-person frames. This tool changes no production code and
no replay data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from applications.live_pipeline.adapters import DepthThresholdConfig  # noqa: E402
from applications.live_pipeline.mask_selection import select_pose_gradient_mask  # noqa: E402
from scripts.inspect_replay_masks import (  # noqa: E402
    REGION_JOINT_SETS,
    _read_image,
    _summary,
    _draw_overlay,
    measure_region,
)

# Compare the live default with the full-image stride.
REFERENCE_STRIDE = 4
ALTERNATE_STRIDE = 1
REGION_PAD_PX = 2


def mask_iou(first: np.ndarray, second: np.ndarray) -> float | None:
    """Intersection over union of two binary masks, or None when both are empty."""
    a = np.asarray(first, dtype=bool)
    b = np.asarray(second, dtype=bool)
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return None
    return float(np.logical_and(a, b).sum() / union)


def load_depth(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    with np.load(path) as data:
        if "depth_m" not in data:
            return None
        return np.asarray(data["depth_m"], dtype=np.float32)


def _selection_payload(selection) -> dict:
    mask = np.asarray(selection.info.mask)
    return {
        "source": selection.source,
        "fallback_reason": selection.fallback_reason,
        "area_fraction": round(float(mask.mean()), 4),
        "mask": mask,
    }


def compare_row(output_dir: Path, row: dict, index: int, *, report_dir: Path) -> dict:
    """Run both strides on one saved frame and report the difference."""
    generation = row.get("generation")
    frame_id = row.get("selected_frame_id")
    key = f"{index:04d}_g{generation}_f{int(frame_id):08d}" if frame_id is not None else f"{index:04d}"
    payload: dict = {
        "index": index,
        "frame_id": frame_id,
        "status": row.get("status"),
        "saved_mask_source": row.get("mask_source"),
        "saved_fallback_reason": row.get("fallback_reason"),
        "excluded": None,
        "overlay": None,
        "problems": [],
        "reference": None,
        "alternate": None,
        "mask_iou": None,
        "different_pixel_fraction": None,
        "saved_mask_iou": None,
        "regions": {},
    }
    mask_pose_path = row.get("mask_pose")
    if not mask_pose_path:
        payload["excluded"] = "no_joint_record"
        return payload
    joint_file = output_dir / mask_pose_path
    if not joint_file.is_file():
        payload["excluded"] = "missing_file"
        payload["problems"].append(f"mask_pose file missing: {mask_pose_path}")
        return payload
    joints = json.loads(joint_file.read_text(encoding="utf-8")).get("joints")
    if not joints:
        payload["excluded"] = "no_joint_record"
        return payload
    depth_path = row.get("selected_depth")
    if not depth_path:
        payload["excluded"] = "no_depth"
        return payload
    depth = load_depth(output_dir / depth_path)
    if depth is None or depth.ndim != 2:
        payload["excluded"] = "missing_file"
        payload["problems"].append(f"depth file missing or unreadable: {depth_path}")
        return payload

    config = DepthThresholdConfig()
    reference = _selection_payload(
        select_pose_gradient_mask(
            depth, pose_joints=joints, config=config, gradient_stride=REFERENCE_STRIDE
        )
    )
    alternate = _selection_payload(
        select_pose_gradient_mask(
            depth, pose_joints=joints, config=config, gradient_stride=ALTERNATE_STRIDE
        )
    )
    reference_mask = reference.pop("mask")
    alternate_mask = alternate.pop("mask")
    payload["reference"] = {**reference, "stride": REFERENCE_STRIDE}
    payload["alternate"] = {**alternate, "stride": ALTERNATE_STRIDE}
    payload["mask_iou"] = mask_iou(reference_mask, alternate_mask)
    payload["different_pixel_fraction"] = round(
        float(np.logical_xor(reference_mask.astype(bool), alternate_mask.astype(bool)).mean()), 6
    )

    saved_mask = None
    saved_mask_path = row.get("baseline_mask")
    if saved_mask_path:
        saved_mask = _read_image(output_dir / saved_mask_path, cv2.IMREAD_GRAYSCALE)
        if saved_mask is None:
            payload["problems"].append(f"baseline_mask missing or unreadable: {saved_mask_path}")
        else:
            saved_mask = saved_mask > 0
    if saved_mask is not None and saved_mask.shape == reference_mask.shape:
        payload["saved_mask_iou"] = mask_iou(saved_mask, reference_mask)
    elif saved_mask is not None:
        payload["problems"].append("baseline_mask and recomputed mask size differ")

    for name, group in REGION_JOINT_SETS:
        reference_region = measure_region(joints, group, reference_mask, pad_px=REGION_PAD_PX)
        alternate_region = measure_region(joints, group, alternate_mask, pad_px=REGION_PAD_PX)
        if reference_region is None or alternate_region is None:
            payload["regions"][name] = None
            continue
        payload["regions"][name] = {
            "reference_coverage": round(reference_region.coverage, 4),
            "alternate_coverage": round(alternate_region.coverage, 4),
            "coverage_delta": round(alternate_region.coverage - reference_region.coverage, 4),
            "reference_joint_coverage": round(reference_region.joint_coverage, 4),
            "alternate_joint_coverage": round(alternate_region.joint_coverage, 4),
            "width_px": reference_region.width_px,
            "height_px": reference_region.height_px,
            "visible_joints": reference_region.visible_joints,
        }

    rgb_path = row.get("selected_rgb")
    rgb = _read_image(output_dir / rgb_path, cv2.IMREAD_COLOR) if rgb_path else None
    if rgb is not None and rgb.shape[:2] == reference_mask.shape:
        overlay_dir = report_dir / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        first = _draw_overlay(rgb, reference_mask, {})
        second = _draw_overlay(rgb, alternate_mask, {})
        _label(first, f"stride {REFERENCE_STRIDE}")
        _label(second, f"stride {ALTERNATE_STRIDE}")
        paired = np.hstack([first, second])
        overlay_path = overlay_dir / f"{key}_stride_compare.png"
        if not cv2.imwrite(str(overlay_path), paired):
            payload["problems"].append(f"could not write overlay: {overlay_path}")
        else:
            payload["overlay"] = f"overlays/{overlay_path.name}"
    return payload


def _label(image: np.ndarray, text: str) -> None:
    cv2.putText(
        image, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
    )


def build_summary(rows: list[dict]) -> dict:
    compared = [row for row in rows if row["excluded"] is None]
    exclusions: dict[str, int] = {}
    for row in rows:
        if row["excluded"] is not None:
            exclusions[row["excluded"]] = exclusions.get(row["excluded"], 0) + 1

    fallback = {"both_pose_gradient": 0, "both_fallback": 0, "changed": 0}
    for row in compared:
        first = row["reference"]["source"]
        second = row["alternate"]["source"]
        if first == "pose_gradient" and second == "pose_gradient":
            fallback["both_pose_gradient"] += 1
        elif first == "pose_gradient" or second == "pose_gradient":
            fallback["changed"] += 1
        else:
            fallback["both_fallback"] += 1

    def collect(key: str) -> list[float]:
        return [row[key] for row in compared if row[key] is not None]

    area_delta = [
        abs(row["alternate"]["area_fraction"] - row["reference"]["area_fraction"])
        for row in compared
    ]
    regions: dict[str, dict] = {}
    for name, _group in REGION_JOINT_SETS:
        coverages = [
            row["regions"][name] for row in compared if row["regions"].get(name) is not None
        ]
        regions[name] = {
            "rows": len(coverages),
            "reference_coverage": _summary([item["reference_coverage"] for item in coverages]),
            "alternate_coverage": _summary([item["alternate_coverage"] for item in coverages]),
            "coverage_delta": _summary([item["coverage_delta"] for item in coverages]),
            "changed_rows": sum(1 for item in coverages if item["coverage_delta"] != 0.0),
        }

    saved_iou = collect("saved_mask_iou")
    return {
        "rows_total": len(rows),
        "rows_compared": len(compared),
        "exclusions": dict(sorted(exclusions.items())),
        "fallback_sources": fallback,
        "mask_iou": _summary(collect("mask_iou")),
        "different_pixel_fraction": _summary(collect("different_pixel_fraction")),
        "area_abs_delta": _summary(area_delta),
        "saved_mask_iou_reference": _summary(saved_iou),
        "saved_mask_mismatch_rows": sum(
            1 for value in saved_iou if value < 0.999
        ),
        "regions": regions,
        "problems": sum(1 for row in rows if row["problems"]),
    }


def _render_html(report: dict) -> str:
    from html import escape

    summary = report["summary"]
    lines = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<title>Stride 1 and stride 4 mask comparison</title>",
        "<style>body{font-family:system-ui,sans-serif;background:#14171a;color:#e6e6e6;margin:24px}"
        "table{border-collapse:collapse;margin:12px 0}td,th{border:1px solid #39424a;padding:4px 8px;"
        "font-size:13px;text-align:left}th{background:#20262b}img{max-width:900px;border:1px solid #39424a}"
        ".note{color:#9aa7b0;max-width:900px}</style></head><body>",
        "<h1>Stride 1 and stride 4 mask comparison</h1>",
        f"<p class='note'>Replay data: <code>{escape(report['output_dir'])}</code><br>",
        "Both masks come from the live selector on the same saved depth and the same exact "
        "mask-input joints. The left overlay uses stride 4 (the live default), the right overlay "
        "uses stride 1. This is evidence about the gradient percentile estimate only; it is not a "
        "mask accuracy result.</p>",
        f"<p>rows: {summary['rows_total']} | compared: {summary['rows_compared']} | "
        f"exclusions: {escape(json.dumps(summary['exclusions']))} | fallback sources: "
        f"{escape(json.dumps(summary['fallback_sources']))}</p>",
        "<table><tr><th>metric</th><th>p05</th><th>p50</th><th>p95</th><th>rows</th></tr>",
    ]
    for label, key in (
        ("mask intersection over union", "mask_iou"),
        ("different pixel fraction", "different_pixel_fraction"),
        ("absolute area-fraction delta", "area_abs_delta"),
        ("stride-4 against saved mask union", "saved_mask_iou_reference"),
    ):
        entry = summary[key]
        lines.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(label), entry["p05"], entry["p50"], entry["p95"], entry["rows"],
            )
        )
    lines.append("</table>")
    lines.append(
        "<h2>Joint regions</h2><table><tr><th>region</th><th>rows</th>"
        "<th>stride-4 coverage p50</th><th>stride-1 coverage p50</th>"
        "<th>delta p50</th><th>changed rows</th></tr>"
    )
    for name, region in summary["regions"].items():
        lines.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(name), region["rows"],
                region["reference_coverage"]["p50"],
                region["alternate_coverage"]["p50"],
                region["coverage_delta"]["p50"],
                region["changed_rows"],
            )
        )
    lines.append("</table><h2>Rows</h2>")
    lines.append(
        "<table><tr><th>frame</th><th>status</th><th>stride-4 source</th>"
        "<th>stride-1 source</th><th>union</th><th>different fraction</th>"
        "<th>saved union</th><th>excluded</th><th>overlay</th></tr>"
    )
    for row in report["rows"]:
        reference = row["reference"] or {}
        alternate = row["alternate"] or {}
        lines.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>"
            "<td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(str(row["frame_id"])), escape(str(row["status"])),
                escape(str(reference.get("source"))), escape(str(alternate.get("source"))),
                escape(str(row["mask_iou"])), escape(str(row["different_pixel_fraction"])),
                escape(str(row["saved_mask_iou"])), escape(str(row["excluded"])),
                (
                    f"<a href='{escape(row['overlay'])}'><img src='{escape(row['overlay'])}'></a>"
                    if row["overlay"]
                    else ""
                ),
            )
        )
    lines.append("</table></body></html>")
    return "\n".join(lines)


def compare_replay(
    output_dir: Path,
    *,
    report_dir: Path | None = None,
    log=print,
) -> dict:
    """Compare both strides for every usable row in one replay output."""
    from scripts.inspect_replay_masks import load_rows

    output_dir = Path(output_dir).resolve()
    resolved_report = (
        Path(report_dir).resolve()
        if report_dir is not None
        else Path(str(output_dir) + ".stride_compare")
    )
    if resolved_report == output_dir or output_dir in resolved_report.parents:
        raise ValueError("report directory must be outside the replay data directory")
    rows = load_rows(output_dir)
    resolved_report.mkdir(parents=True, exist_ok=True)
    results = [
        compare_row(output_dir, row, index, report_dir=resolved_report)
        for index, row in enumerate(rows)
    ]
    report = {
        "schema_version": "1",
        "output_dir": str(output_dir),
        "report_dir": str(resolved_report),
        "reference_stride": REFERENCE_STRIDE,
        "alternate_stride": ALTERNATE_STRIDE,
        "region_pad_px": REGION_PAD_PX,
        "summary": build_summary(results),
        "rows": results,
    }
    (resolved_report / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (resolved_report / "index.html").write_text(_render_html(report), encoding="utf-8")
    log(f"report written: {resolved_report}")
    return report


def print_report(report: dict, log=print) -> None:
    summary = report["summary"]
    log(
        "rows: {} compared: {} exclusions: {} fallback: {}".format(
            summary["rows_total"], summary["rows_compared"],
            summary["exclusions"], summary["fallback_sources"],
        )
    )
    for key in ("mask_iou", "different_pixel_fraction", "area_abs_delta", "saved_mask_iou_reference"):
        entry = summary[key]
        log(f"{key}: rows={entry['rows']} p05/p50/p95={entry['p05']}/{entry['p50']}/{entry['p95']}")
    for name, region in summary["regions"].items():
        log(
            "region={} rows={} coverage stride4/stride1={}/{} delta p50={} changed rows={}".format(
                name, region["rows"], region["reference_coverage"]["p50"],
                region["alternate_coverage"]["p50"], region["coverage_delta"]["p50"],
                region["changed_rows"],
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    report = compare_replay(args.output_dir, report_dir=args.report_dir)
    print_report(report)


if __name__ == "__main__":
    main()
