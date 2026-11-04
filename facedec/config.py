from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """Thresholds and preprocessing knobs.

    The match thresholds are deliberately conservative placeholders. They are
    not calibrated probabilities; tune them on a labeled poster set.
    """

    model_name: str = "buffalo_l"
    det_size: int = 640
    det_thresh: float = 0.5

    tile: bool = True
    tile_size: int = 1024
    tile_overlap: int = 128
    tile_trigger: int = 1600
    max_tiles: int = 24

    min_face_px: float = 48.0
    max_faces: int = 64

    accept_threshold: float = 0.45
    review_threshold: float = 0.30
    margin_threshold: float = 0.08
    min_quality: float = 0.10
    unknown_score: float | None = None

    def __post_init__(self) -> None:
        if self.det_size <= 0:
            raise ValueError("det_size must be positive")
        if self.tile_size <= 0 or self.tile_overlap < 0:
            raise ValueError("tile_size must be positive and tile_overlap non-negative")
        if self.tile_overlap >= self.tile_size:
            raise ValueError("tile_overlap must be smaller than tile_size")
        if not 0.0 <= self.review_threshold <= self.accept_threshold <= 1.0:
            raise ValueError("expected 0 <= review_threshold <= accept_threshold <= 1")
        if not 0.0 <= self.min_quality <= 1.0:
            raise ValueError("min_quality must be in [0, 1]")
        if self.min_face_px <= 0:
            raise ValueError("min_face_px must be positive")
        if self.max_faces <= 0:
            raise ValueError("max_faces must be positive")
