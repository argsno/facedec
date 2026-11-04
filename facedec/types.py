from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

BBox = tuple[float, float, float, float]


@dataclass
class DetectedFace:
    """One detected and aligned face, in original-image coordinates."""

    bbox: BBox
    kps: np.ndarray | None
    det_score: float
    embedding: np.ndarray
    size_px: float
    blur: float
    yaw: float
    quality: float
    face_id: str = ""
    source: str = "poster"

    def to_dict(self) -> dict[str, Any]:
        return {
            "face_id": self.face_id,
            "bbox": [round(float(v), 2) for v in self.bbox],
            "det_score": round(float(self.det_score), 4),
            "quality": {
                "score": round(float(self.quality), 4),
                "size_px": round(float(self.size_px), 1),
                "blur": round(float(self.blur), 2),
                "yaw": round(float(self.yaw), 2),
            },
            "source": self.source,
        }


@dataclass
class GalleryMember:
    cast_id: str
    name: str
    role: str | None = None
    embeddings: list[np.ndarray] = field(default_factory=list)
    quality: float = 0.0
    status: str = "ok"
    warnings: list[str] = field(default_factory=list)
    portrait_count: int = 0


@dataclass
class Candidate:
    cast_id: str
    name: str
    role: str | None
    score: float


@dataclass
class FaceMatch:
    face: DetectedFace
    assigned: Candidate | None
    candidates: list[Candidate]
    margin: float | None
    status: str
    reason: str | None = None
