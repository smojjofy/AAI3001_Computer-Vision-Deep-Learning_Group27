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