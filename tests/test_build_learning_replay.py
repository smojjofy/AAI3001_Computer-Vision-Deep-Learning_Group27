"""Output-publication rules for the learning replay extractor.

Boundaries: a completed staged run, an interrupted staged run, an existing
completed target, a conflicting session split, a stale staging directory, a
manifest write failure before publication, the default non-staged path, and the
uniqueness of the staging name. Every model call is replaced, so no model file is
needed and no clip is decoded.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts import build_learning_replay as module

COUNTS = {"windows": 4, "refresh_ok": 2, "refresh_failed": 1, "reuse": 1}


class FakePipeline:
    instances: list["FakePipeline"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.closed = False
        self.yolo_imgsz = 640
        self.depth_confidence = False
        FakePipeline.instances.append(self)

    def close(self) -> None:
        self.closed = True


def install_fakes(monkeypatch, *, run_error: BaseException | None = None, manifest_error=None) -> None:
    FakePipeline.instances = []

    def fake_run_replay(packets, *, selector, processor, output_dir, settings, models):
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "rows.jsonl").write_text('{"status": "ok"}\n', encoding="utf-8")
        if run_error is not None:
            raise run_error
        return dict(COUNTS)

    monkeypatch.setattr(module, "run_replay", fake_run_replay)
    monkeypatch.setattr(module, "RefreshPipeline", FakePipeline)
    monkeypatch.setattr(module, "LiveReplayProcessor", lambda pipeline: ("processor", pipeline))
    if manifest_error is not None:
        def fail_manifest(directory, manifest):
            raise manifest_error

        monkeypatch.setattr(module, "_write_manifest", fail_manifest)


@pytest.fixture()
def clip(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"clip placeholder")
    return path


def settings_for(clip: Path, *, clip_id: str = "clip1", split: str = "train"):
    return module.ReplaySettings(session_id="session1", clip_id=clip_id, split=split)


def staging_dirs(session_dir: Path) -> list[Path]:
    return sorted(session_dir.glob("clip1.staging-*"))


def test_completed_staged_run_publishes_the_final_output(tmp_path: Path, clip: Path, monkeypatch) -> None:
    install_fakes(monkeypatch)
    messages: list[str] = []
    output, counts = module.build_replay_clip(
        clip=clip, settings=settings_for(clip), output_root=tmp_path / "out",
        staged_output=True, log=messages.append,
    )
    session_dir = tmp_path / "out" / "session1"
    assert output == session_dir / "clip1"
    assert counts == COUNTS
    assert (output / "rows.jsonl").is_file()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["staged_output"] is True
    assert manifest["counts"] == COUNTS
    assert staging_dirs(session_dir) == []
    # The session split was reserved before extraction and survives the publish.
    assert json.loads((session_dir / "split.json").read_text(encoding="utf-8")) == {
        "session_id": "session1", "split": "train",
    }
    assert FakePipeline.instances and FakePipeline.instances[0].closed is True


def test_interrupted_staged_run_leaves_only_a_staging_directory(tmp_path: Path, clip: Path, monkeypatch) -> None:
    install_fakes(monkeypatch, run_error=ValueError("decode failed"))
    messages: list[str] = []
    with pytest.raises(ValueError, match="decode failed"):
        module.build_replay_clip(
            clip=clip, settings=settings_for(clip), output_root=tmp_path / "out",
            staged_output=True, log=messages.append,
        )
    session_dir = tmp_path / "out" / "session1"
    assert not (session_dir / "clip1").exists()
    staged = staging_dirs(session_dir)
    assert len(staged) == 1
    assert (staged[0] / "rows.jsonl").is_file()
    assert not (staged[0] / "manifest.json").exists()
    # The split is reserved before extraction, so it remains after a failure.
    assert (session_dir / "split.json").is_file()
    assert any("staging directory kept" in message for message in messages)


def test_staged_run_does_not_replace_a_completed_output(tmp_path: Path, clip: Path, monkeypatch) -> None:
    install_fakes(monkeypatch)
    first, _ = module.build_replay_clip(
        clip=clip, settings=settings_for(clip), output_root=tmp_path / "out",
        staged_output=True, log=lambda _message: None,
    )
    before = (first / "rows.jsonl").read_text(encoding="utf-8")
    session_dir = tmp_path / "out" / "session1"
    with pytest.raises(FileExistsError, match="already exists"):
        module.build_replay_clip(
            clip=clip, settings=settings_for(clip), output_root=tmp_path / "out",
            staged_output=True, log=lambda _message: None,
        )
    assert (first / "rows.jsonl").read_text(encoding="utf-8") == before
    assert staging_dirs(session_dir) == []


def test_conflicting_split_is_refused_before_any_staging(tmp_path: Path, clip: Path, monkeypatch) -> None:
    install_fakes(monkeypatch)
    module.build_replay_clip(
        clip=clip, settings=settings_for(clip), output_root=tmp_path / "out",
        staged_output=True, log=lambda _message: None,
    )
    session_dir = tmp_path / "out" / "session1"
    with pytest.raises(ValueError, match="different splits"):
        module.build_replay_clip(
            clip=clip, settings=settings_for(clip, clip_id="clip2", split="test"),
            output_root=tmp_path / "out", staged_output=True, log=lambda _message: None,
        )
    assert not (session_dir / "clip2").exists()
    assert staging_dirs(session_dir) == []
    assert json.loads((session_dir / "split.json").read_text(encoding="utf-8"))["split"] == "train"


def test_stale_staging_directories_are_reported_and_kept(tmp_path: Path, clip: Path, monkeypatch) -> None:
    install_fakes(monkeypatch)
    session_dir = tmp_path / "out" / "session1"
    stale = session_dir / "clip1.staging-999-abcdef123456"
    stale.mkdir(parents=True)
    (stale / "rows.jsonl").write_text("{}\n", encoding="utf-8")
    messages: list[str] = []
    output, _ = module.build_replay_clip(
        clip=clip, settings=settings_for(clip), output_root=tmp_path / "out",
        staged_output=True, log=messages.append,
    )
    assert stale.is_dir()
    assert (stale / "rows.jsonl").is_file()
    assert any(str(stale) in message for message in messages)
    assert output.is_dir()


def test_manifest_failure_before_publication_keeps_the_staging_run(
    tmp_path: Path, clip: Path, monkeypatch
) -> None:
    install_fakes(monkeypatch, manifest_error=OSError("disk full"))
    messages: list[str] = []
    with pytest.raises(OSError, match="disk full"):
        module.build_replay_clip(
            clip=clip, settings=settings_for(clip), output_root=tmp_path / "out",
            staged_output=True, log=messages.append,
        )
    session_dir = tmp_path / "out" / "session1"
    assert not (session_dir / "clip1").exists()
    staged = staging_dirs(session_dir)
    assert len(staged) == 1
    assert (staged[0] / "rows.jsonl").is_file()
    assert any("staging directory kept" in message for message in messages)


def test_default_path_still_writes_the_clip_directory_directly(
    tmp_path: Path, clip: Path, monkeypatch
) -> None:
    install_fakes(monkeypatch)
    output, _ = module.build_replay_clip(
        clip=clip, settings=settings_for(clip), output_root=tmp_path / "out",
    )
    session_dir = tmp_path / "out" / "session1"
    assert output == session_dir / "clip1"
    assert (output / "manifest.json").is_file()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["staged_output"] is False
    assert staging_dirs(session_dir) == []


def test_staging_name_is_unique_beyond_the_process_id(tmp_path: Path) -> None:
    session_dir = tmp_path / "out" / "session1"
    first = module.staging_path(session_dir, "clip1")
    second = module.staging_path(session_dir, "clip1")
    assert first != second
    assert first.parent == session_dir
    assert str(os.getpid()) in first.name
    assert first.name.startswith("clip1.staging-")


def test_publish_refuses_an_existing_target_and_keeps_the_staging_directory(tmp_path: Path) -> None:
    session_dir = tmp_path / "out" / "session1"
    staging = session_dir / "clip1.staging-1-aaaaaaaaaaaa"
    staging.mkdir(parents=True)
    (staging / "rows.jsonl").write_text("{}\n", encoding="utf-8")
    target = session_dir / "clip1"
    target.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="already exists"):
        module.publish_output(staging, target)
    assert staging.is_dir()
    assert (staging / "rows.jsonl").is_file()


def test_publish_protects_against_a_concurrent_writer(tmp_path: Path, monkeypatch) -> None:
    session_dir = tmp_path / "out" / "session1"
    staging = session_dir / "clip1.staging-1-bbbbbbbbbbbb"
    staging.mkdir(parents=True)
    (staging / "rows.jsonl").write_text("{}\n", encoding="utf-8")
    target = session_dir / "clip1"

    def racing_rename(source, destination):
        # A second writer publishes between the pre-check and the rename.
        Path(destination).mkdir()
        raise FileExistsError("target exists")

    monkeypatch.setattr(module.os, "rename", racing_rename)
    with pytest.raises(FileExistsError, match="staging directory is kept"):
        module.publish_output(staging, target)
    assert staging.is_dir()
    assert (staging / "rows.jsonl").is_file()
    assert target.is_dir()


def test_reserve_split_creates_once_and_accepts_the_same_split(tmp_path: Path) -> None:
    session_dir = tmp_path / "out" / "session1"
    session_dir.mkdir(parents=True)
    first = module.reserve_split(session_dir, "session1", "train")
    assert json.loads(first.read_text(encoding="utf-8")) == {
        "session_id": "session1", "split": "train",
    }
    assert module.reserve_split(session_dir, "session1", "train") == first
    with pytest.raises(ValueError, match="different splits"):
        module.reserve_split(session_dir, "session1", "validation")


def test_staged_output_help_states_the_repair_limit() -> None:
    parser = module.build_parser()
    action = next(action for action in parser._actions if "--staged-output" in action.option_strings)
    assert action.default is False
    assert "cannot repair" in action.help
