from dataclasses import dataclass


@dataclass(frozen=True)
class PoseLandmark:
    x: float
    y: float
    z: float
    visibility: float | None
    presence: float | None


@dataclass(frozen=True)
class PoseEstimation:
    detected: bool
    landmarks: tuple[PoseLandmark, ...]


@dataclass(frozen=True)
class PoseFrameResult:
    """Pose result tied to its original source frame."""

    frame_id: int
    source_id: str
    capture_timestamp_ns: int
    pose: PoseEstimation

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError(
                "frame_id must be non-negative"
            )

        if not self.source_id:
            raise ValueError(
                "source_id must not be empty"
            )

        if self.capture_timestamp_ns < 0:
            raise ValueError(
                "capture_timestamp_ns must be non-negative"
            )