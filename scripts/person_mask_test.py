"""Controlled, repeatable test of one person-mask checkpoint.

This tool has two subcommands.

``offline`` scores a frozen checkpoint against the accepted
``pose_gradient`` mask and the hand-checked masks of one saved replay clip. It
extends the existing evaluator: it adds face-edge coverage, source-size metrics
side by side, paired overlays for wins and losses, bootstrap confidence
intervals, and a written decision rule. It reads replay data, the label store,
and one checkpoint. It writes only to ``--report-dir``.

``live-shadow`` prints the command that starts the live application in shadow
mode, and then it reads the shadow log back. It does not open a camera.

Example, from ``live-splat-pipeline/``:

    python -m scripts.person_mask_test offline ^
        --checkpoint checkpoints/paired_research/person_mask_candidate_v1.pt ^
        --replay-dir data/processed/learning_replay/person_test_01/test_01 ^
        --labels-root data/processed/learning_replay/labels ^
        --dataset test_01 ^
        --training-filter data/processed/learning_replay/person_test_01/test_01.training_filter.json ^
        --split test --report-dir outputs/paired_research/mask_candidate_test

    python -m scripts.person_mask_test live-shadow ^
        --shadow-log outputs/paired_research/live_shadow/mask_shadow_run01.jsonl

``baseline_mask`` is an input, never a label. The checked mask is the label.
"""

from __future__ import annotations

import argparse
from html import escape
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.inference.paired_research.data import WIDTH as RESEARCH_WIDTH, HEIGHT as RESEARCH_HEIGHT  # noqa: E402

SCHEMA_VERSION = "1"
MASK_THRESHOLD = 0.5
FIXED_SEED = 7
DEFAULT_BOOTSTRAP = 10000
DEFAULT_MAX_OVERLAYS = 40
MIN_GAIN = 0.03
LOWER_PERCENTILE = 2.5
UPPER_PERCENTILE = 97.5
MIN_VISIBILITY = 0.15
FACE_JOINTS = tuple(range(0, 11))
FACE_MIN_JOINTS = 3
FACE_PAD_FRACTION = 0.10
FACE_EDGE_PX = 3
SOURCE_PAD_PX = 2
SCORE_DECIMALS = 6
TINT_ALPHA = 0.45
ACCEPTED_COLOUR = (0, 200, 0)
CANDIDATE_COLOUR = (0, 200, 0)
CHECKED_COLOUR = (255, 128, 0)
WIN_TEXT_COLOUR = (80, 220, 80)
LOSS_TEXT_COLOUR = (80, 80, 255)
CHECKPOINT_TASK = "paired_mask_motion_research"
FACE_DEPTH_README = "outputs/face_depth_diagnostic/README.md"
FROZEN_STATUS = "frozen_candidate_for_untouched_test_not_live_promoted"


# --------------------------------------------------------------- utilities


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile of a list, or None for an empty list.

    The rank is ``ceil(p / 100 * n)``, clamped to ``[1, n]``. The result is one
    value of the list. No value is interpolated.
    """
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if not 0.0 <= p <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    rank = math.ceil(p / 100.0 * len(ordered))
    rank = min(max(rank, 1), len(ordered))
    return ordered[rank - 1]


def _rounded(value: float | None) -> float | None:
    """Round a metric for the report. None stays None."""
    if value is None:
        return None
    return round(float(value), SCORE_DECIMALS)


def _mean(values: Iterable[float]) -> float | None:
    """Arithmetic mean, or None when the list is empty."""
    listed = [float(value) for value in values]
    return round(statistics.fmean(listed), SCORE_DECIMALS) if listed else None


def _median(values: Iterable[float]) -> float | None:
    """Median, or None when the list is empty."""
    listed = [float(value) for value in values]
    return round(statistics.median(listed), SCORE_DECIMALS) if listed else None


def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of one file."""
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------- face region


def face_region_from_joints(
    joints: list[dict] | None,
    *,
    width: int,
    height: int,
    min_visibility: float = MIN_VISIBILITY,
    pad_fraction: float = FACE_PAD_FRACTION,
) -> tuple[int, int, int, int] | None:
    """Padded box around the visible face pose points, or None.

    The box uses pose points 0 through 10. A point needs visibility at or above
    ``min_visibility`` and must fall inside the image. At least
    ``FACE_MIN_JOINTS`` such points are necessary. The box is enlarged by
    ``pad_fraction`` of its width and height on each side, by at least one pixel,
    and then clipped to the image. The result is ``(left, right, top, bottom)``
    with inclusive edges.
    """
    if joints is None or width < 1 or height < 1:
        return None
    xs: list[int] = []
    ys: list[int] = []
    for joint in joints:
        index = joint.get("index")
        if index is None or int(index) not in FACE_JOINTS:
            continue
        visibility = joint.get("visibility")
        if visibility is None or float(visibility) < float(min_visibility):
            continue
        x = joint.get("x")
        y = joint.get("y")
        if x is None or y is None:
            continue
        px = int(round(float(x) * (width - 1)))
        py = int(round(float(y) * (height - 1)))
        if 0 <= px < width and 0 <= py < height:
            xs.append(px)
            ys.append(py)
    if len(xs) < FACE_MIN_JOINTS:
        return None
    pad_x = max(1, int(round(float(pad_fraction) * (max(xs) - min(xs)))))
    pad_y = max(1, int(round(float(pad_fraction) * (max(ys) - min(ys)))))
    left = max(0, min(xs) - pad_x)
    right = min(width - 1, max(xs) + pad_x)
    top = max(0, min(ys) - pad_y)
    bottom = min(height - 1, max(ys) + pad_y)
    if right < left or bottom < top:
        return None
    return left, right, top, bottom


def _region_slice(region: tuple[int, int, int, int], shape: tuple[int, ...]) -> tuple[slice, slice]:
    """Array slices of one inclusive box, clipped to the array shape."""
    height, width = shape[:2]
    left, right, top, bottom = region
    return slice(max(0, top), min(height, bottom + 1)), slice(max(0, left), min(width, right + 1))


def face_edge_mask(
    checked: np.ndarray,
    region: tuple[int, int, int, int] | None,
    *,
    edge_px: int = FACE_EDGE_PX,
) -> np.ndarray:
    """Checked-person boundary pixels that lie inside the face region.

    The boundary is a band of ``edge_px`` pixels around the checked mask, on the
    inside. Pixels outside the region are false. A None region gives all false.
    """
    checked_bool = np.asarray(checked, dtype=bool)
    inside = checked_bool.astype(np.uint8)
    if edge_px < 1:
        raise ValueError("edge_px must be a positive integer")
    kernel = np.ones((2 * int(edge_px) + 1, 2 * int(edge_px) + 1), np.uint8)
    dilated = cv2.dilate(inside, kernel) > 0
    eroded = cv2.erode(inside, kernel) > 0
    band = np.logical_and(checked_bool, np.logical_and(dilated, ~eroded))
    if region is None:
        return np.zeros_like(checked_bool)
    window = np.zeros_like(checked_bool)
    rows, columns = _region_slice(region, checked_bool.shape)
    window[rows, columns] = True
    return np.logical_and(band, window)


# ----------------------------------------------------------------- scores


def score_mask(
    predicted: np.ndarray | None,
    checked: np.ndarray,
    *,
    face_region: tuple[int, int, int, int] | None = None,
    face_edge_px: int = FACE_EDGE_PX,
    pad_px: int = SOURCE_PAD_PX,
) -> dict:
    """Score one predicted mask against the checked mask.

    Returns intersection over union, person coverage, background leakage, the
    largest leaking connected component, and face-edge coverage. A metric is None
    when its denominator is zero. Face-edge coverage is a diagnostic proxy: the
    face region comes from the pose points, so it includes background by design,
    and it cannot establish face accuracy.
    """
    checked_bool = np.asarray(checked, dtype=bool)
    result: dict[str, Any] = {
        "iou": None,
        "person_coverage": None,
        "background_leakage": None,
        "largest_leak_px": None,
        "face_edge_coverage": None,
        "face_region_pixels": None,
        "face_edge_pixels": None,
        "checked_pixels": int(checked_bool.sum()),
        "mask_pixels": None,
        "face_region": None if face_region is None else list(face_region),
    }
    edge = face_edge_mask(checked_bool, face_region, edge_px=face_edge_px)
    result["face_region_pixels"] = (
        None if face_region is None else int(_region_full(face_region, checked_bool.shape).sum())
    )
    result["face_edge_pixels"] = int(edge.sum())
    if predicted is None:
        result["error"] = "no mask for this method"
        return result
    predicted_bool = np.asarray(predicted, dtype=bool)
    if predicted_bool.shape != checked_bool.shape:
        result["error"] = "mask and label size differ"
        return result
    result["mask_pixels"] = int(predicted_bool.sum())
    intersection = int(np.logical_and(predicted_bool, checked_bool).sum())
    union = int(np.logical_or(predicted_bool, checked_bool).sum())
    person = int(checked_bool.sum())
    leak = np.logical_and(predicted_bool, ~checked_bool)
    leak_pixels = int(leak.sum())
    result["iou"] = _rounded(intersection / union) if union else None
    result["person_coverage"] = _rounded(intersection / person) if person else None
    result["background_leakage"] = _rounded(leak_pixels / person) if person else None
    if leak_pixels:
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            leak.astype(np.uint8), connectivity=8
        )
        result["largest_leak_px"] = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
    else:
        result["largest_leak_px"] = 0
    edge_total = int(edge.sum())
    if edge_total:
        kept = int(np.logical_and(edge, predicted_bool).sum())
        result["face_edge_coverage"] = _rounded(kept / edge_total)
    return result


def _region_full(region: tuple[int, int, int, int], shape: tuple[int, ...]) -> np.ndarray:
    """Boolean plane of the region, for the region pixel count."""
    plane = np.zeros(shape[:2], dtype=bool)
    rows, columns = _region_slice(region, shape)
    plane[rows, columns] = True
    return plane


# ------------------------------------------------------------- statistics


def pairwise_summary(diffs: list[float], *, bootstrap: int = DEFAULT_BOOTSTRAP, seed: int = FIXED_SEED) -> dict:
    """Mean, median, wins, ties, losses, and a seeded bootstrap interval.

    ``diffs`` holds one value for each frame: the method score minus the
    accepted score. Wins, ties, and losses count the sign of each value. The
    bootstrap draws ``bootstrap`` resamples with replacement, with
    ``random.Random(seed)``, and reports the nearest-rank 2.5th and 97.5th
    percentile of the resampled mean.
    """
    if bootstrap < 1:
        raise ValueError("bootstrap must be a positive integer")
    values = [float(value) for value in diffs]
    wins = sum(1 for value in values if value > 0.0)
    losses = sum(1 for value in values if value < 0.0)
    ties = len(values) - wins - losses
    summary: dict[str, Any] = {
        "frames": len(values),
        "mean": _mean(values),
        "median": _median(values),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "bootstrap": int(bootstrap),
        "seed": int(seed),
        "percentile_rule": "nearest rank, ceil(p/100*n) clamped to [1, n]",
        "lower_percentile": LOWER_PERCENTILE,
        "upper_percentile": UPPER_PERCENTILE,
        "lower_95": None,
        "upper_95": None,
        "mean_gain": _mean(values),
    }
    if not values:
        return summary
    if len(values) == 1 or bootstrap == 1:
        summary["lower_95"] = summary["mean"]
        summary["upper_95"] = summary["mean"]
        return summary
    generator = random.Random(int(seed))
    means = [
        statistics.fmean(generator.choices(values, k=len(values))) for _ in range(int(bootstrap))
    ]
    summary["lower_95"] = _rounded(percentile(means, LOWER_PERCENTILE))
    summary["upper_95"] = _rounded(percentile(means, UPPER_PERCENTILE))
    return summary


def decide(metrics: dict, *, min_gain: float = MIN_GAIN) -> dict:
    """Apply the offline decision rule to one paired summary.

    The result names the state of the OFFLINE gate. It is not a promotion and it
    never enables a live model. ``offline_gate_passed`` means the measured
    numbers met the three written conditions: the lower bootstrap bound is above
    zero, the mean gain is at least ``min_gain``, and the loss count is not
    greater than the win count. Otherwise the outcome is
    ``offline_gate_failed`` and the ``reason`` list names every condition that
    failed.

    Passing this gate is a precondition for review. It is not approval, because
    the gate cannot see the size of each loss, the concentration of the gain, or
    whether the frames were chosen for difficulty. A promotion decision needs a
    rule frozen before a new untouched session, and that decision belongs to the
    user.
    """
    lower = metrics.get("lower_95", metrics.get("mean"))
    mean = metrics.get("mean", metrics.get("mean_gain"))
    wins = int(metrics.get("wins") or 0)
    losses = int(metrics.get("losses") or 0)
    reasons: list[str] = []
    if lower is None or float(lower) <= 0.0:
        reasons.append("lower_bootstrap_bound_not_above_zero")
    if mean is None or float(mean) < float(min_gain):
        reasons.append("mean_gain_below_minimum")
    if losses > wins:
        reasons.append("losses_exceed_wins")
    return {
        "outcome": "offline_gate_passed" if not reasons else "offline_gate_failed",
        "offline_gate_passed": not reasons,
        "live_promoted": False,
        "reason": reasons,
        "min_gain": float(min_gain),
        "lower_95": lower,
        "mean_gain": mean,
        "wins": wins,
        "losses": losses,
        "rule": (
            "offline_gate_passed needs lower_95 > 0, mean_gain >= min_gain, and "
            "losses <= wins. Passing this gate is a precondition for review. It "
            "is not a promotion, and live_promoted is always False here."
        ),
        "limits": [
            "The bootstrap resamples temporally related frames, so the interval is not a session-level confidence result.",
            "The gate does not measure the size of a loss. A large loss and a small loss count the same.",
            "The gate does not separate frames chosen for difficulty from frames chosen at random.",
            "Passing this gate does not enable a live model.",
        ],
    }


# --------------------------------------------------------------- overlays


def _tint(image: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int]) -> np.ndarray:
    """Tint the mask pixels at alpha 0.45 and draw the outline."""
    output = image.copy()
    inside = np.asarray(mask, dtype=bool)
    if inside.any():
        tint = np.array(colour, dtype=np.float32)
        output[inside] = (
            output[inside].astype(np.float32) * (1.0 - TINT_ALPHA) + tint * TINT_ALPHA
        ).astype(np.uint8)
        contours, _hierarchy = cv2.findContours(
            inside.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(output, contours, -1, colour, 2)
    return output


def _draw_text(image: np.ndarray, lines: list[str], *, colour: tuple[int, int, int] = (230, 230, 230)) -> None:
    """Draw short text lines at the top left of the image."""
    for index, text in enumerate(lines):
        y = 16 + 14 * index
        cv2.putText(
            image, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
            (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            image, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
            colour, 1, cv2.LINE_AA,
        )


def _panel_text(name: str, entry: dict | None) -> list[str]:
    """Method name and per-method numbers for one overlay panel."""
    if entry is None:
        return [name, "no score"]
    parts = [name]
    parts.append(f"iou {_format(entry.get('iou'))}")
    parts.append(f"cov {_format(entry.get('person_coverage'))} leak {_format(entry.get('background_leakage'))}")
    parts.append(f"face {_format(entry.get('face_edge_coverage'))} big-leak {entry.get('largest_leak_px')}")
    return parts


def _format(value: Any) -> str:
    """Short text for one number."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def paired_overlay(
    rgb: np.ndarray,
    *,
    accepted: np.ndarray,
    candidate: np.ndarray,
    checked: np.ndarray,
    scores: dict[str, dict | None],
    face_region: tuple[int, int, int, int] | None,
    label: str,
) -> np.ndarray:
    """RGB, accepted, candidate, and checked overlays in one row.

    Accepted and candidate use the green tint. The checked mask uses blue. The
    panel text names the colour and the method for each panel.
    """
    panels = [rgb.copy()]
    _draw_text(panels[0], ["rgb", label])
    for name, colour in (
        ("accepted (green)", ACCEPTED_COLOUR),
        ("candidate (green)", CANDIDATE_COLOUR),
        ("checked label (blue)", CHECKED_COLOUR),
    ):
        index = len(panels)
        mask = (accepted, candidate, checked)[index - 1]
        blur = cv2.GaussianBlur(rgb, (0, 0), 5.0)
        base = cv2.addWeighted(rgb, 0.35, blur, 0.65, 0.0)
        panel = _tint(base, mask, colour)
        if face_region is not None:
            left, right, top, bottom = face_region
            cv2.rectangle(panel, (left, top), (right, bottom), (0, 255, 255), 1)
        _draw_text(panel, _panel_text(name, scores.get(name.split(" ")[0])))
        panels.append(panel)
    return np.hstack(panels)


# ------------------------------------------------------------------ report


def _write_json(path: Path, payload: Any) -> None:
    """Write one JSON file with indentation."""
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _render_html(report: dict) -> str:
    """Render the dark-theme summary page."""
    lines = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<title>Person-mask checkpoint test</title>",
        "<style>body{font-family:system-ui,sans-serif;background:#14171a;color:#e6e6e6;margin:24px}"
        "table{border-collapse:collapse;margin:12px 0}td,th{border:1px solid #39424a;padding:4px 8px;"
        "font-size:13px;text-align:left}th{background:#20262b}img{max-width:1200px;border:1px solid #39424a}"
        ".note{color:#9aa7b0;max-width:1100px}.warn{color:#ffb454}code{color:#9ad}", "</style></head><body>",
        "<h1>Person-mask checkpoint test</h1>",
        f"<p class='note'>Replay data: <code>{escape(report['replay_dir'])}</code><br>",
        f"Label store: <code>{escape(str(report['labels_root']))}</code> dataset "
        f"<code>{escape(str(report['dataset']))}</code><br>",
        f"Checkpoint: <code>{escape(report['checkpoint'])}</code> SHA-256 "
        f"<code>{escape(report['checkpoint_sha256'])}</code><br>",
        f"Split <code>{escape(report['split'])}</code>, clip "
        f"<code>{escape(str(report['clip_id']))}</code>, session "
        f"<code>{escape(str(report['session_id']))}</code>, frames {report['frames_scored']}.</p>",
        "<p class='note'>The accepted mask is the saved <code>baseline_mask</code> of the replay. "
        "It is an input, not a label. The checked mask is the label. Accepted and candidate use a "
        "green tint. The checked mask uses blue. Face-edge coverage is a diagnostic proxy.</p>",
    ]
    warnings = report.get("data_integrity", {}).get("warnings", [])
    if warnings:
        lines.append("<p class='warn'>Data-integrity warnings: " + escape("; ".join(warnings)) + "</p>")
    lines.append(
        "<h2>Scope</h2><p class='note'>" + escape(" ".join(report.get("scope_notes", []))) + "</p>"
    )
    lines.append(
        "<h2>Methods</h2><table><tr><th>method</th><th>size</th><th>mean IoU</th>"
        "<th>mean gain</th><th>median gain</th><th>W/T/L</th><th>95% interval</th>"
        "<th>decision</th><th>reason</th></tr>"
    )
    for entry in report["methods"]:
        paired = entry["paired"]
        decision = entry["decision"]
        lines.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}/{}/{}</td>"
            "<td>[{}, {}]</td><td>{}</td><td>{}</td></tr>".format(
                escape(entry["method"]), escape(entry["size"]),
                escape(_format(entry["iou"]["mean"])), escape(_format(paired["mean"])),
                escape(_format(paired["median"])),
                paired["wins"], paired["ties"], paired["losses"],
                escape(_format(paired["lower_95"])), escape(_format(paired["upper_95"])),
                escape(decision["outcome"]), escape(", ".join(decision["reason"]) or "none"),
            )
        )
    lines.append("</table>")
    lines.append(
        "<h2>Metrics by method</h2><table><tr><th>method</th><th>size</th><th>coverage mean</th>"
        "<th>leakage mean</th><th>face-edge mean</th><th>largest leak mean px</th><th>frames</th></tr>"
    )
    for entry in report["methods"]:
        lines.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(entry["method"]), escape(entry["size"]),
                escape(_format(entry["person_coverage"]["mean"])),
                escape(_format(entry["background_leakage"]["mean"])),
                escape(_format(entry["face_edge_coverage"]["mean"])),
                escape(_format(entry["largest_leak_px"]["mean"])),
                entry["iou"]["frames"],
            )
        )
    lines.append("</table>")
    lines.append(
        "<h2>Frames</h2><table><tr><th>frame</th><th>time s</th><th>accepted IoU</th>"
        "<th>candidate IoU</th><th>candidate vs accepted</th><th>candidate restricted vs accepted</th>"
        "<th>restricted IoU</th><th>face edge accepted/candidate</th><th>result</th>"
        "<th>overlay</th></tr>"
    )
    for record in report["records"]:
        scores = record["scores"]["source"]
        gains = record["gains"]["candidate"]
        restrictions = record["gains"]["candidate_restricted"]
        accepted_score = (scores.get("accepted") or {})
        candidate_score = (scores.get("candidate") or {})
        restricted_score = (scores.get("candidate_restricted") or {})
        lines.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>"
            "<td>{}/{}</td><td>{}</td><td>{}</td></tr>".format(
                escape(record["frame_key"]), escape(str(record.get("timestamp_s"))),
                escape(_format(accepted_score.get("iou"))),
                escape(_format(candidate_score.get("iou"))),
                escape(_format(gains)),
                escape(_format(restrictions)),
                escape(_format(restricted_score.get("iou"))),
                escape(_format(accepted_score.get("face_edge_coverage"))),
                escape(_format(candidate_score.get("face_edge_coverage"))),
                escape(record.get("result") or record.get("error") or ""),
                (
                    f"<a href='{escape(record['overlay'])}'><img src='{escape(record['overlay'])}'></a>"
                    if record.get("overlay")
                    else ""
                ),
            )
        )
    lines.append("</table>")
    lines.append(
        "<h2>Rules</h2><table><tr><th>item</th><th>value</th></tr>"
        + "".join(
            f"<tr><td>{escape(str(key))}</td><td>{escape(json.dumps(value))}</td></tr>"
            for key, value in report.get("rules", {}).items()
        )
        + "</table>"
    )
    lines.append("<h2>Limits</h2><ul>")
    for limit in report.get("limits", []):
        lines.append(f"<li>{escape(limit)}</li>")
    lines.append("</ul></body></html>")
    return "\n".join(lines)


# ---------------------------------------------------------- offline scoring


def _read_image(path: Path, flags: int) -> np.ndarray | None:
    """Read one image file, or None when it is absent or unreadable."""
    if not path.is_file():
        return None
    return cv2.imread(str(path), flags)


def _read_binary_mask(path: Path) -> np.ndarray | None:
    """Read a mask PNG as a boolean plane, or None."""
    image = _read_image(path, cv2.IMREAD_GRAYSCALE)
    return None if image is None else (image >= 128)


def _load_depth(path: Path) -> np.ndarray | None:
    """Read the metric depth plane of one replay row, or None."""
    if not path.is_file():
        return None
    with np.load(path) as archive:
        if "depth_m" not in archive:
            return None
        return np.asarray(archive["depth_m"], dtype=np.float32)


def _load_joints(path: Path) -> list[dict] | None:
    """Read the exact mask-input joints of one replay row, or None."""
    if not path.is_file():
        return None
    joints = json.loads(path.read_text(encoding="utf-8")).get("joints")
    return joints if isinstance(joints, list) and joints else None


def _selection_mask(
    joints: list[dict], depth: np.ndarray, config: Any, stride: int
) -> np.ndarray:
    """One pose-gradient selection, reduced to the boolean mask."""
    from applications.live_pipeline.mask_selection import select_pose_gradient_mask

    selection = select_pose_gradient_mask(
        depth, pose_joints=joints, config=config, gradient_stride=stride
    )
    return np.asarray(selection.info.mask) > 0


def _recompute_accepted(
    row: dict, replay_dir: Path, saved: np.ndarray, *, config: Any
) -> tuple[np.ndarray, dict]:
    """Recompute the accepted mask from the saved depth and exact joints.

    The replay stores one mask per row. The live path chooses the pose-gradient
    mask with a depth-threshold fallback. This function tries stride 4 first and
    stride 1 second, then reports the agreement of the stored mask with both.
    """
    depth = _load_depth(replay_dir / row.get("selected_depth", "")) if row.get("selected_depth") else None
    joints = _load_joints(replay_dir / row.get("mask_pose", "")) if row.get("mask_pose") else None
    info: dict[str, Any] = {
        "depth_present": depth is not None,
        "joints_present": joints is not None,
        "agreement_stride4": None,
        "agreement_stride1": None,
        "agrees": False,
        "error": None,
    }
    if depth is None or joints is None:
        info["error"] = "missing depth or exact joints"
        return saved, info
    try:
        first = _selection_mask(joints, depth, config, 4)
        info["agreement_stride4"] = _rounded(float(np.mean(first == saved)))
        best = first
        if info["agreement_stride4"] == 1.0:
            info["agrees"] = True
            return first, info
        second = _selection_mask(joints, depth, config, 1)
        info["agreement_stride1"] = _rounded(float(np.mean(second == saved)))
        if info["agreement_stride1"] == 1.0:
            info["agrees"] = True
            return second, info
        if (info["agreement_stride1"] or 0.0) > (info["agreement_stride4"] or 0.0):
            best = second
    except Exception as error:  # noqa: BLE001 - recorded, never swallowed silently
        info["error"] = f"{type(error).__name__}: {error}"
        return saved, info
    info["agrees"] = False
    return best, info


def _gate_radius(world_radius_px: int, width: int, height: int) -> tuple[int, int]:
    """Convert the 320x180 gate radius to source pixels for each axis."""
    radius_x = max(1, int(round(world_radius_px * width / RESEARCH_WIDTH)))
    radius_y = max(1, int(round(world_radius_px * height / RESEARCH_HEIGHT)))
    return radius_x, radius_y


def candidate_restricted_mask(accepted: np.ndarray, candidate: np.ndarray, *, radius_px: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Candidate pixels that lie inside the accepted mask or near it.

    The source-equivalent radii are ``round(radius * width / 320)`` and
    ``round(radius * height / 180)``. The kernel is one ellipse that spans those
    radii. This is a diagnostic restriction. It is not part of the checkpoint.
    """
    height, width = accepted.shape[:2]
    radius_x, radius_y = _gate_radius(radius_px, width, height)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_x + 1, 2 * radius_y + 1))
    allowed = cv2.dilate(np.asarray(accepted, dtype=np.uint8), kernel) > 0
    return np.logical_and(np.asarray(candidate, dtype=bool), allowed), (radius_x, radius_y)


def _summary_of(values: list[float]) -> dict:
    """Mean, median, and percentile summary of a list of floats."""
    if not values:
        return {"frames": 0, "mean": None, "median": None, "p05": None, "p50": None, "p95": None}
    return {
        "frames": len(values),
        "mean": _mean(values),
        "median": _median(values),
        "p05": _rounded(percentile(values, 5.0)),
        "p50": _rounded(percentile(values, 50.0)),
        "p95": _rounded(percentile(values, 95.0)),
    }


def _method_entry(
    name: str,
    size: str,
    records: list[dict],
    reference: str,
    *,
    bootstrap: int,
    seed: int,
    min_gain: float,
) -> dict:
    """Aggregate one scored method against the accepted mask."""
    valid = [record for record in records if record.get("error") is None]
    method_scores = [record["scores"][size][name] for record in valid if record["scores"][size].get(name)]
    accepted_scores = [
        record["scores"][size][reference] for record in valid if record["scores"][size].get(reference)
    ]
    pairs = [
        (method["iou"], accepted["iou"])
        for method, accepted in zip(method_scores, accepted_scores)
        if method.get("iou") is not None and accepted.get("iou") is not None
    ]
    diffs = [method - accepted for method, accepted in pairs]
    paired = pairwise_summary(diffs, bootstrap=bootstrap, seed=seed)
    entry = {
        "method": name,
        "size": size,
        "reference": reference,
        "frames": len(pairs),
        "iou": _summary_of([method for method, _accepted in pairs]),
        "accepted_iou": _summary_of([accepted for _method, accepted in pairs]),
        "paired": paired,
        "decision": decide(paired, min_gain=min_gain),
    }
    for metric in ("person_coverage", "background_leakage", "face_edge_coverage", "largest_leak_px"):
        values = [
            score[metric] for score in method_scores
            if score.get(metric) is not None and score.get("error") is None
        ]
        entry[metric] = _summary_of([float(value) for value in values])
    return entry


def _overlay_selection(records: list[dict], cap: int) -> list[dict]:
    """Pick the frames to draw: all losses first, then wins by decreasing gain."""
    losses = sorted(
        (record for record in records if record.get("result") == "loss"),
        key=lambda record: (record["gains"]["candidate"] or 0.0),
    )
    wins = sorted(
        (record for record in records if record.get("result") == "win"),
        key=lambda record: -(record["gains"]["candidate"] or 0.0),
    )
    ordered = losses + wins
    return ordered[: max(0, int(cap))]


def run_offline(
    checkpoint: Path,
    replay_dir: Path,
    labels_root: Path | None,
    filter_path: Path,
    *,
    dataset: str | None = None,
    expected_split: str = "test",
    split_override: str | None = None,
    report_dir: Path | None = None,
    frames: list[str] | None = None,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = FIXED_SEED,
    device: str = "cpu",
    expect_sha256: str | None = None,
    max_overlays: int = DEFAULT_MAX_OVERLAYS,
    log=print,
) -> dict:
    """Score one checkpoint and write report.json, index.html, and overlays."""
    import torch
    from torch.nn import functional as F

    from applications.live_pipeline.adapters import DepthThresholdConfig
    from modules.inference.checkpoint import load_checkpoint
    from modules.inference.paired_research.data import PairedReplayDataset
    from modules.inference.paired_research.model import PairedMaskMotionNet

    checkpoint = Path(checkpoint)
    replay_dir = Path(replay_dir).resolve()
    if expected_split not in ("validation", "test"):
        raise ValueError("offline scoring needs a validation or test session")
    dataset_split = str(split_override or expected_split)
    if dataset_split not in ("train", "validation", "test"):
        raise ValueError("the dataset split must be train, validation, or test")
    if expect_sha256 is not None:
        actual = sha256_file(checkpoint)
        if actual.lower() != str(expect_sha256).strip().lower():
            raise ValueError(
                f"checkpoint SHA-256 {actual} does not match the expected value {expect_sha256}"
            )
    payload = load_checkpoint(checkpoint, expected_task=CHECKPOINT_TASK, map_location="cpu")
    config = payload["model_config"]
    trained_heads = list(config.get("trained_heads", []))
    if "mask" not in trained_heads:
        raise ValueError("checkpoint mask head is not trained")
    if "in_channels" in config and int(config["in_channels"]) != 15:
        raise ValueError("checkpoint does not read the 15-channel research input")
    quality_dataset = PairedReplayDataset(
        replay_dir, labels_root, filter_path, expected_split=dataset_split
    )
    manifest = quality_dataset.manifest
    if (manifest.get("session_id") == config.get("session_id")
            or manifest.get("clip_sha256") == config.get("clip_sha256")):
        raise ValueError("scored session or clip matches the training data")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")

    resolved_report = (
        Path(report_dir).resolve() if report_dir is not None else Path(str(replay_dir) + ".person_mask_test")
    )
    if resolved_report == replay_dir or replay_dir in resolved_report.parents:
        raise ValueError("report directory must be outside the replay data directory")
    resolved_report.mkdir(parents=True, exist_ok=True)
    overlay_dir = resolved_report / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    model = PairedMaskMotionNet(
        in_channels=int(config["in_channels"]),
        base_channels=int(config["base_channels"]),
        mask_mode=str(config.get("mask_mode", "absolute")),
        gate_radius_px=int(config.get("gate_radius_px", 12)),
        residual_scale=float(config.get("residual_scale", 4.0)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    radius_px = int(config.get("gate_radius_px", 12))
    config_threshold = DepthThresholdConfig()
    checkpoint_sha = sha256_file(checkpoint)

    records: list[dict] = []
    warnings_list: list[str] = []
    mismatched: list[dict] = []
    with torch.inference_mode():
        for key, (inputs, _target), row, label_path in zip(
            quality_dataset.frame_keys, quality_dataset.samples,
            quality_dataset.rows, quality_dataset.label_paths,
        ):
            record: dict[str, Any] = {"frame_key": key, "overlay": None}
            rgb = _read_image(replay_dir / row["selected_rgb"], cv2.IMREAD_COLOR) if row.get("selected_rgb") else None
            checked = _read_binary_mask(Path(label_path))
            saved_accepted = _read_image(replay_dir / row["baseline_mask"], cv2.IMREAD_GRAYSCALE) if row.get("baseline_mask") else None
            joints = _load_joints(replay_dir / row["mask_pose"]) if row.get("mask_pose") else None
            if checked is None or saved_accepted is None or rgb is None:
                record.update({"error": "missing checked mask, accepted mask, or RGB"})
                records.append(record)
                continue
            saved_accepted = saved_accepted >= 128
            accepted, integrity = _recompute_accepted(row, replay_dir, saved_accepted, config=config_threshold)
            record["accepted_mask_check"] = integrity
            if not integrity["agrees"]:
                mismatched.append({
                    "frame_key": key,
                    "agreement_stride4": integrity["agreement_stride4"],
                    "agreement_stride1": integrity["agreement_stride1"],
                    "error": integrity["error"],
                })
            if accepted.shape != checked.shape:
                record.update({"error": "source mask dimensions do not match"})
                records.append(record)
                continue
            if checked.sum() == 0:
                record.update({"error": "checked mask is empty"})
                records.append(record)
                continue
            mask_logits, _motion = model(inputs.unsqueeze(0).to(device))
            probability = torch.sigmoid(mask_logits)[0, 0].cpu().numpy()
            research_candidate = probability >= MASK_THRESHOLD
            source_logits = F.interpolate(
                mask_logits, size=checked.shape, mode="bilinear", align_corners=False
            )
            candidate = source_logits[0, 0].cpu().numpy() >= math.log(
                MASK_THRESHOLD / (1.0 - MASK_THRESHOLD)
            )
            restricted, radii = candidate_restricted_mask(
                accepted, candidate, radius_px=radius_px
            )
            research_checked = _read_resized_mask(
                Path(label_path), RESEARCH_WIDTH, RESEARCH_HEIGHT
            )
            research_accepted = inputs[14].numpy() >= 0.5
            region = face_region_from_joints(joints, width=checked.shape[1], height=checked.shape[0])
            record.update({
                "frame_id": row.get("selected_frame_id"),
                "timestamp_s": round(float(row.get("selected_timestamp_ns", 0)) / 1e9, 3),
                "rgb": row.get("selected_rgb"),
                "source_size": [int(checked.shape[1]), int(checked.shape[0])],
                "mask_source": row.get("mask_source"),
                "fallback_reason": row.get("fallback_reason"),
                "face_region": None if region is None else list(region),
                "restriction_radius_px": list(radii),
                "scores": {
                    "source": {
                        "accepted": score_mask(accepted, checked, face_region=region),
                        "candidate": score_mask(candidate, checked, face_region=region),
                        "candidate_restricted": score_mask(restricted, checked, face_region=region),
                        "checked": score_mask(checked, checked, face_region=region),
                    },
                    "research": {
                        "accepted": score_mask(research_accepted, research_checked),
                        "candidate": score_mask(research_candidate, research_checked),
                    },
                },
            })
            source_scores = record["scores"]["source"]
            gains = {}
            for name in ("candidate", "candidate_restricted"):
                method_iou = source_scores[name]["iou"]
                accepted_iou = source_scores["accepted"]["iou"]
                gains[name] = (
                    None if method_iou is None or accepted_iou is None
                    else _rounded(method_iou - accepted_iou)
                )
            record["gains"] = gains
            record["result"] = _result_label(gains["candidate"])
            research_gains = None
            if (record["scores"]["research"]["candidate"]["iou"] is not None
                    and record["scores"]["research"]["accepted"]["iou"] is not None):
                research_gains = _rounded(
                    record["scores"]["research"]["candidate"]["iou"]
                    - record["scores"]["research"]["accepted"]["iou"]
                )
            record["research_gain"] = research_gains
            record["_masks"] = {
                "accepted": accepted, "candidate": candidate, "checked": checked,
                "candidate_restricted": restricted,
            }
            records.append(record)

    if frames:
        wanted = [str(name).strip() for name in frames if str(name).strip()]
        known = {record["frame_key"] for record in records}
        unknown = [name for name in wanted if name not in known]
        if unknown:
            raise ValueError(f"requested frames are not in the scored set: {unknown}")
        order = {name: index for index, name in enumerate(wanted)}
        records = sorted(
            (record for record in records if record["frame_key"] in order),
            key=lambda record: order[record["frame_key"]],
        )
    if not records:
        raise ValueError("no frames were scored")

    cap = max(0, int(max_overlays))
    for record in _overlay_selection(records, cap):
        rgb = _read_image(replay_dir / record["rgb"], cv2.IMREAD_COLOR) if record.get("rgb") else None
        overlay_path = None if rgb is None else _write_overlay(overlay_dir, record, rgb=rgb)
        if overlay_path is not None:
            record["overlay"] = overlay_path
    for record in records:
        record.pop("_masks", None)

    scored = [record for record in records if record.get("error") is None]
    methods = []
    for name in ("candidate", "candidate_restricted"):
        methods.append(_method_entry(
            name, "source", scored, "accepted",
            bootstrap=bootstrap, seed=seed, min_gain=MIN_GAIN,
        ))
    for name in ("candidate",):
        methods.append(_method_entry(
            name, "research", scored, "accepted",
            bootstrap=bootstrap, seed=seed, min_gain=MIN_GAIN,
        ))
    for name in ("accepted", "checked"):
        methods.append(_reference_entry(name, "source", scored, bootstrap=bootstrap, seed=seed))

    if mismatched:
        warnings_list.append(
            "accepted mask recompute does not agree with the saved baseline_mask for "
            f"{len(mismatched)} of {len(scored)} scored frames"
        )
    if any(record.get("scores", {}).get("source", {}).get("accepted", {}).get("iou") is None for record in scored):
        warnings_list.append("some accepted masks have no comparable intersection over union")

    manifest_status = _manifest_status()
    report = {
        "schema_version": SCHEMA_VERSION,
        "tool": "person_mask_test offline",
        "replay_dir": str(replay_dir),
        "labels_root": None if labels_root is None else str(Path(labels_root).resolve()),
        "dataset": dataset,
        "training_filter": str(Path(filter_path).resolve()),
        "split": dataset_split,
        "expected_split": expected_split,
        "session_id": manifest.get("session_id"),
        "clip_id": manifest.get("clip_id"),
        "clip_sha256": manifest.get("clip_sha256"),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_train_session": config.get("session_id"),
        "checkpoint_train_clip_sha256": config.get("clip_sha256"),
        "checkpoint_manifest_status": manifest_status,
        "mask_mode": config.get("mask_mode", "absolute"),
        "gate_radius_px_at_320x180": radius_px,
        "trained_heads": trained_heads,
        "device": device,
        "frames_scored": len(scored),
        "frames_requested": len(records),
        "frames_total_in_dataset": len(quality_dataset),
        "max_overlays": cap,
        "overlays_written": sum(1 for record in records if record.get("overlay")),
        "overlay_order": "all losses first (lowest gain first), then wins by decreasing gain",
        "data_integrity": {
            "checked": "every scored frame",
            "mismatch_frames": len(mismatched),
            "mismatches": mismatched,
            "warnings": warnings_list,
        },
        "rules": {
            "threshold": MASK_THRESHOLD,
            "research_size": [RESEARCH_WIDTH, RESEARCH_HEIGHT],
            "source_rule": "bilinear interpolation of mask logits, align_corners=False, threshold logit(0.5) = 0",
            "accepted_source": "pose_gradient mask recomputed with select_pose_gradient_mask(depth, mask_pose joints, DepthThresholdConfig(), gradient_stride=4); agrees with the saved baseline_mask when the live selector also used stride 4",
            "accepted_input": "baseline_mask of the replay row is an input, not a label",
            "restriction": (
                "candidate & dilate(accepted, ellipse kernel with radius_x = round(radius * width / 320) "
                "and radius_y = round(radius * height / 180)); a diagnostic restriction, not part of the checkpoint"
            ),
            "restriction_radius_px_at_320x180": radius_px,
            "face_region": (
                "bounding box of mask_pose pose points 0 through 10 with visibility >= 0.15, "
                "enlarged by 10 percent of its width and height on each side (at least 1 px), clipped to the image"
            ),
            "face_edge": (
                "checked-person pixels in a 3 px band inside the checked boundary, restricted to the face region"
            ),
            "bootstrap": {
                "resamples": int(bootstrap),
                "seed": int(seed),
                "generator": "random.Random(seed), resample frames with replacement",
                "percentile_rule": "nearest rank, ceil(p/100*n) clamped to [1, n]",
                "interval": [LOWER_PERCENTILE, UPPER_PERCENTILE],
            },
            "decision_rule": {
                "min_gain": MIN_GAIN,
                "conditions": [
                    "lower bootstrap bound above 0.0",
                    "mean IoU difference at least min_gain",
                    "loss count not greater than win count",
                ],
                "outcomes": ["offline_gate_passed", "offline_gate_failed"],
                "note": (
                    "These are offline gate states, not promotions. Passing the gate is a "
                    "precondition for review. No state in this report enables a live model."
                ),
            },
        },
        "scope_notes": _scope_notes(manifest_status),
        "methods": methods,
        "records": records,
        "limits": _limits(mismatched),
    }
    _write_json(resolved_report / "report.json", report)
    (resolved_report / "index.html").write_text(_render_html(report), encoding="utf-8")
    log(f"report written: {resolved_report}")
    print_summary(report, log=log)
    return report


def _read_resized_mask(path: Path, width: int, height: int) -> np.ndarray:
    """Read a checked mask and resize it to the research size with nearest neighbours."""
    image = _read_image(Path(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot read checked mask: {path}")
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    return resized >= 128


def _result_label(gain: float | None) -> str:
    """Win, tie, or loss for one signed IoU difference."""
    if gain is None:
        return "unknown"
    if gain > 0.0:
        return "win"
    if gain < 0.0:
        return "loss"
    return "tie"


def _write_overlay(overlay_dir: Path, record: dict, *, rgb: np.ndarray | None) -> str | None:
    """Draw and save the paired overlay of one frame."""
    scores = record["scores"]["source"]
    image = paired_overlay(
        rgb,
        accepted=record["_masks"]["accepted"],
        candidate=record["_masks"]["candidate"],
        checked=record["_masks"]["checked"],
        scores=scores,
        face_region=record.get("face_region"),
        label=f"{record['frame_key']} {record['result']} gain {_format(record['gains']['candidate'])}",
    )
    path = overlay_dir / f"{record['frame_key']}_{record['result']}.png"
    if not cv2.imwrite(str(path), image):
        return None
    return f"overlays/{path.name}"


def _reference_entry(name: str, size: str, records: list[dict], *, bootstrap: int, seed: int) -> dict:
    """Summary of one method that is scored without a pair."""
    scores = [
        record["scores"][size][name]
        for record in records
        if record["scores"][size].get(name) and record["scores"][size][name].get("error") is None
    ]
    entry = {
        "method": name,
        "size": size,
        "reference": None,
        "frames": sum(1 for score in scores if score.get("iou") is not None),
        "iou": _summary_of([score["iou"] for score in scores if score.get("iou") is not None]),
        "paired": pairwise_summary([], bootstrap=bootstrap, seed=seed),
        "decision": {"outcome": "not_applicable", "live_promoted": False, "reason": ["reference method"]},
    }
    for metric in ("person_coverage", "background_leakage", "face_edge_coverage", "largest_leak_px"):
        entry[metric] = _summary_of([
            float(score[metric]) for score in scores if score.get(metric) is not None
        ])
    return entry


def _manifest_status(expected_status: str = FROZEN_STATUS) -> dict:
    """Read the checkpoint manifest status when the file exists.

    A status other than ``frozen_candidate_for_untouched_test_not_live_promoted``
    is kept as it is, so the report never states a status that the file lacks.
    """
    path = ROOT / "outputs" / "paired_research" / "candidate_v1_manifest.json"
    if not path.is_file():
        return {"status": None, "expected_status": expected_status, "path": None, "note": "manifest not found"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "status": payload.get("status"),
        "expected_status": expected_status,
        "matches_expected": payload.get("status") == expected_status,
        "path": str(path),
    }


def _scope_notes(manifest_status: dict) -> list[str]:
    """State what this run can and cannot claim."""
    notes = [
        f"{FACE_DEPTH_README} records that the test clip was already inspected for the face-depth "
        "question. This run is therefore not an independent quality claim for a tuned depth rule.",
        "This run is legitimate for a mask candidate whose checkpoint was frozen before the test "
        "session was recorded, because the checkpoint saw no frame of this session.",
        f"The checkpoint manifest status is {manifest_status.get('status')!r}. The brief expects "
        f"{FROZEN_STATUS!r}. No live promotion follows from one number.",
    ]
    return notes


def _limits(mismatched: list[dict]) -> list[str]:
    """List the known limits of this report."""
    values = [
        "One session, 18 checked frames from the same clip, so the frames are temporally related.",
        "Face-edge coverage is a diagnostic proxy: the face region comes from pose points and includes background.",
        "The checked mask is a person silhouette, not a face label.",
        "The candidate_restricted mask is a diagnostic restriction, not part of the checkpoint.",
        "The motion head of the model is not measured.",
        "No live capture-to-display result is included.",
        "The checked-mask boundary is approximate, so a boundary error of under about 2 px is inside the measurement.",
    ]
    if mismatched:
        values.append(
            f"The accepted mask recompute disagrees with the saved baseline_mask on {len(mismatched)} frame(s)."
        )
    return values


def print_summary(report: dict, log=print) -> None:
    """Print one line for each method and the decision."""
    for entry in report["methods"]:
        if entry.get("reference") is not None:
            log(
                "method={} size={} frames={} mean_iou={} accepted_mean_iou={} mean_gain={} "
                "median_gain={} W/T/L={}/{}/{} interval=[{}, {}]".format(
                    entry["method"], entry["size"], entry["frames"],
                    entry["iou"]["mean"], (entry.get("accepted_iou") or {}).get("mean"),
                    entry["paired"]["mean"], entry["paired"]["median"],
                    entry["paired"]["wins"], entry["paired"]["ties"], entry["paired"]["losses"],
                    entry["paired"]["lower_95"], entry["paired"]["upper_95"],
                )
            )
        else:
            log(
                "method={} size={} frames={} mean_iou={} face_edge_mean={}".format(
                    entry["method"], entry["size"], entry["frames"],
                    entry["iou"]["mean"], entry["face_edge_coverage"]["mean"],
                )
            )
    decisions = []
    for entry in report["methods"]:
        if entry.get("reference") is not None:
            decisions.append(
                f"{entry['method']}@{entry['size']}={entry['decision']['outcome']}"
                f"({', '.join(entry['decision']['reason']) or 'all conditions met'})"
            )
    log("decision: " + "; ".join(decisions))


# ------------------------------------------------------------ live shadow


SHADOW_FLAGS = (
    "--shadow-mask-checkpoint",
    "--shadow-mask-log",
    "--shadow-mask-sha256",
    "--shadow-mask-device",
)
DEFAULT_LIVE_CHECKPOINT = "checkpoints/paired_research/person_mask_candidate_v1.pt"
DEFAULT_LIVE_SHA256 = "d821ffcc66557047b50625c4c7345634aacf72b55b0181d0d34d7e487d927606"


def live_shadow_command(
    *,
    shadow_log: Path,
    checkpoint: str = DEFAULT_LIVE_CHECKPOINT,
    sha256: str = DEFAULT_LIVE_SHA256,
    device: str = "cuda",
    duration_s: float = 60.0,
    camera_device: int = 0,
) -> list[str]:
    """Build the live command line that the operator must run."""
    return [
        sys.executable, "-m", "scripts.run_integration_app",
        "--source", "webcam",
        "--camera-backend", "dshow",
        "--device", str(camera_device),
        "--shadow-mask-checkpoint", checkpoint,
        "--shadow-mask-sha256", sha256,
        "--shadow-mask-device", device,
        "--shadow-mask-log", str(shadow_log),
        "--duration-s", str(duration_s),
    ]


def _quote_command(command: list[str]) -> str:
    """Render a command line for a PowerShell console."""
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def _quantiles(values: list[float]) -> dict:
    """p50 and p95 for each named list."""
    return {
        "frames": len(values),
        "p50": _rounded(percentile(values, 50.0)),
        "p95": _rounded(percentile(values, 95.0)),
    }


def summarize_live_shadow(records: list[dict]) -> dict:
    """Counts and quantiles over the shadow records.

    The record format belongs to ``modules.inference.paired_research.shadow``.
    This function reads the record through that module, so the log reader and
    the log writer cannot drift apart. When the module is not importable, the
    function falls back to the flat field names of an older log.
    """
    from collections import Counter

    try:
        from modules.inference.paired_research.shadow import summarize_shadow_records
    except ImportError:
        summarize_shadow_records = None
    if summarize_shadow_records is not None:
        summary = summarize_shadow_records(records)
        return {
            "records": summary["records"],
            "outcomes": dict(sorted(summary["status_counts"].items())),
            "skip_reasons": dict(sorted(summary["skip_reasons"].items())),
            "error_types": dict(sorted(summary["error_types"].items())),
            "shadow_total_ms": summary["timings_ms_total"],
            "iou": summary["compare_iou"],
            "removed_fraction_of_accepted": summary["compare_removed_fraction_of_accepted"],
            "net_area_change_fraction": summary["compare_net_area_change_fraction"],
            "shrink_ratio": summary["compare_shrink_ratio"],
            "candidate_area_fraction": summary["candidate_area_fraction"],
            "accepted_area_fraction": summary["accepted_area_fraction"],
        }
    # Fallback for a log written by an older tool.
    outcomes = Counter(str(record.get("outcome", record.get("status", "unknown"))) for record in records)
    skips = Counter(str(record.get("skip_reason")) for record in records if record.get("skip_reason"))
    errors = Counter(str(record.get("error_type")) for record in records if record.get("error_type"))
    usable = [record for record in records if record.get("outcome", record.get("status")) == "ok"]

    def collect(key: str, candidate_keys: tuple[str, ...]) -> list[float]:
        values: list[float] = []
        for record in usable:
            for name in candidate_keys:
                value = record.get(name)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    values.append(float(value))
                    break
        return values

    return {
        "records": len(records),
        "outcomes": dict(sorted(outcomes.items())),
        "skip_reasons": dict(sorted(skips.items())),
        "error_types": dict(sorted(errors.items())),
        "shadow_total_ms": _quantiles(collect("shadow_total_ms", ("shadow_total_ms", "total_ms"))),
        "iou": _quantiles(collect("iou", ("iou",))),
        "removed_fraction_of_accepted": _quantiles(
            collect("removed_fraction_of_accepted", ("removed_fraction_of_accepted",))
        ),
    }


def run_live_shadow(
    shadow_log: Path,
    *,
    report_dir: Path | None = None,
    replay_dir: Path | None = None,
    checkpoint: str = DEFAULT_LIVE_CHECKPOINT,
    sha256: str = DEFAULT_LIVE_SHA256,
    device: str = "cuda",
    duration_s: float = 60.0,
    camera_device: int = 0,
    log=print,
) -> int:
    """Print the live command, then summarise the shadow log when it exists."""
    shadow_log = Path(shadow_log)
    command = live_shadow_command(
        shadow_log=shadow_log.resolve(), checkpoint=checkpoint, sha256=sha256,
        device=device, duration_s=duration_s, camera_device=camera_device,
    )
    log("Run this command to start the live application in shadow mode:")
    log("  " + _quote_command(command))
    log(
        "The live application must honour the flags "
        + ", ".join(SHADOW_FLAGS)
        + ". If it does not, the shadow log stays empty."
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": "person_mask_test live-shadow",
        "shadow_log": str(shadow_log),
        "command": command,
        "command_text": _quote_command(command),
        "shadow_flags": list(SHADOW_FLAGS),
        "checkpoint": checkpoint,
        "checkpoint_sha256_guard": sha256,
        "log_present": shadow_log.is_file(),
        "summary": None,
    }
    if report_dir is not None:
        resolved = Path(report_dir).resolve()
        if replay_dir is not None:
            replay = Path(replay_dir).resolve()
            if resolved == replay or replay in resolved.parents:
                raise ValueError("report directory must be outside the replay data directory")
        resolved.mkdir(parents=True, exist_ok=True)
    if not shadow_log.is_file():
        log(f"shadow log does not exist yet: {shadow_log}")
        log("Run the command above, then run this subcommand again.")
        if report_dir is not None:
            _write_json(Path(report_dir).resolve() / "live_shadow_report.json", payload)
        return 0
    try:
        import modules.inference.paired_research.shadow as shadow_module
    except ImportError as error:
        log(f"cannot import modules.inference.paired_research.shadow: {error}")
        log("The shadow module is not available, so the log cannot be summarised.")
        return 2
    for name in ("read_shadow_records", "summarize_shadow_records"):
        if not hasattr(shadow_module, name):
            log(f"modules.inference.paired_research.shadow has no {name}")
            return 2
    records = list(shadow_module.read_shadow_records(shadow_log))
    # summarize_live_shadow normalizes the module's record format, so the log
    # reader and the log writer cannot drift apart.
    summary = summarize_live_shadow([record if isinstance(record, dict) else dict(record) for record in records])
    payload["summary"] = summary
    if report_dir is not None:
        _write_json(Path(report_dir).resolve() / "live_shadow_report.json", payload)
    log(f"shadow records: {payload['summary'].get('records')}")
    log(f"outcomes: {payload['summary'].get('outcomes')}")
    log(f"skip reasons: {payload['summary'].get('skip_reasons')}")
    log(f"error types: {payload['summary'].get('error_types')}")
    for key in (
        "shadow_total_ms",
        "iou",
        "removed_fraction_of_accepted",
        "net_area_change_fraction",
        "shrink_ratio",
    ):
        log(f"{key} p50/p95: {payload['summary'].get(key)}")
    return 0


# --------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with the offline and live-shadow subcommands."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    offline = sub.add_parser(
        "offline", help="score one mask checkpoint against the accepted mask and checked masks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    offline.add_argument("--checkpoint", type=Path, required=True)
    offline.add_argument("--replay-dir", type=Path, required=True)
    offline.add_argument("--labels-root", type=Path, default=None)
    offline.add_argument("--dataset", default=None)
    offline.add_argument("--training-filter", type=Path, required=True)
    offline.add_argument("--split", choices=("validation", "test"), required=True)
    offline.add_argument("--report-dir", type=Path, required=True)
    offline.add_argument("--frames", default=None, help="comma-separated frame keys to score")
    offline.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    offline.add_argument("--seed", type=int, default=FIXED_SEED)
    offline.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu",
        help="torch device for inference; the default is cpu for a repeatable result",
    )
    offline.add_argument("--expect-sha256", default=None)
    offline.add_argument("--max-overlays", type=int, default=DEFAULT_MAX_OVERLAYS)

    shadow = sub.add_parser(
        "live-shadow", help="print the live shadow command and read back its log",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    shadow.add_argument("--shadow-log", type=Path, required=True)
    shadow.add_argument("--report-dir", type=Path, default=None)
    shadow.add_argument("--replay-dir", type=Path, default=None)
    shadow.add_argument("--checkpoint", default=DEFAULT_LIVE_CHECKPOINT)
    shadow.add_argument("--sha256", default=DEFAULT_LIVE_SHA256)
    shadow.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    shadow.add_argument("--duration-s", type=float, default=60.0)
    shadow.add_argument("--camera-device", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "offline":
        frames = None if args.frames is None else [part for part in args.frames.split(",")]
        if args.dataset is None:
            parser.error("offline needs --dataset")
        labels_root = args.labels_root
        if labels_root is None:
            labels_root = args.replay_dir.resolve().parent.parent / "labels"
        run_offline(
            args.checkpoint, args.replay_dir, labels_root, args.training_filter,
            dataset=args.dataset, expected_split=args.split, report_dir=args.report_dir,
            frames=frames, bootstrap=args.bootstrap, seed=args.seed, device=args.device,
            expect_sha256=args.expect_sha256, max_overlays=args.max_overlays,
        )
        return 0
    return run_live_shadow(
        args.shadow_log, report_dir=args.report_dir, replay_dir=args.replay_dir,
        checkpoint=args.checkpoint, sha256=args.sha256, device=args.device,
        duration_s=args.duration_s, camera_device=args.camera_device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
