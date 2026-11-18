"""Tests for the optional per-camera pinhole intrinsics file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from applications.live_pipeline.adapters import webcam_intrinsics
from applications.live_pipeline.camera_config import load_camera_config


def write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "camera.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_explicit_pinhole_intrinsics_load(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        {
            "model": "pinhole",
            "width": 640,
            "height": 480,
            "fx": 700.0,
            "fy": 710.0,
            "cx": 320.0,
            "cy": 240.0,
        },
    )
    config = load_camera_config(path)
    assert config.intrinsics == (700.0, 710.0, 320.0, 240.0)
    assert config.matches(640, 480)
    assert not config.matches(1280, 720)


def test_field_of_view_variant_matches_the_estimate(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        {"model": "pinhole", "width": 640, "height": 480, "horizontal_fov_deg": 60.0},
    )
    config = load_camera_config(path)
    assert config.intrinsics == webcam_intrinsics(640, 480, 60.0)


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"model": "fisheye", "width": 640, "height": 480}, "only 'pinhole'"),
        ({"model": "pinhole", "width": 0, "height": 480}, "width"),
        ({"model": "pinhole", "width": 640, "height": 480, "fx": -1, "fy": 1, "cx": 1, "cy": 1}, "fx"),
        ({"model": "pinhole", "width": 640, "height": 480, "horizontal_fov_deg": 180}, "less than 180"),
        ({"model": "pinhole", "width": 640, "height": 480, "fx": 1, "fy": 1, "cx": 900, "cy": 240}, "principal"),
        ({"model": "pinhole", "width": 640.0, "height": 480}, "width"),
    ],
)
def test_invalid_configs_are_rejected(tmp_path: Path, payload: dict, message: str) -> None:
    path = write_config(tmp_path, payload)
    with pytest.raises(ValueError, match=message):
        load_camera_config(path)


def test_malformed_json_and_missing_file_are_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_camera_config(bad)
    with pytest.raises(FileNotFoundError):
        load_camera_config(tmp_path / "missing.json")
