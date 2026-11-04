from __future__ import annotations

import inspect
import math
import warnings

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align

from .config import PipelineConfig
from .types import DetectedFace


class FaceEngine:
    """Thin wrapper around InsightFace's detector + ArcFace recognizer.

    The CPU provider is hard-coded for the feasibility harness. The engine is
    stateless apart from the loaded models and ``last_warnings``.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        # Only load detection + recognition for CPU feasibility. The 3D
        # landmark/genderage addons cost extra per face and are not needed;
        # yaw falls back to a 5-point landmark estimate.
        self.app = FaceAnalysis(
            name=config.model_name,
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        prepare_kwargs: dict = {
            "ctx_id": -1,
            "det_size": (config.det_size, config.det_size),
        }
        if "det_thresh" in inspect.signature(self.app.prepare).parameters:
            prepare_kwargs["det_thresh"] = config.det_thresh
        self.app.prepare(**prepare_kwargs)
        self.last_warnings: list[str] = []

    def analyze(self, image: np.ndarray, source: str = "poster") -> list[DetectedFace]:
        """Detect and embed all faces in an image (no tiling)."""

        with warnings.catch_warnings():
            # InsightFace 2.x still calls a deprecated skimage API internally.
            warnings.simplefilter("ignore", FutureWarning)
            raw_faces = self.app.get(image)
        faces: list[DetectedFace] = []

        for raw in raw_faces:
            bbox = np.asarray(raw.bbox, dtype=np.float32)
            kps = (
                np.asarray(raw.kps, dtype=np.float32)
                if getattr(raw, "kps", None) is not None
                else None
            )

            embedding = getattr(raw, "normed_embedding", None)
            if embedding is None:
                embedding = getattr(raw, "embedding", None)
            if embedding is None:
                continue
            embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(embedding))
            if norm <= 1e-9:
                continue
            embedding = embedding / norm

            size_px = float(min(bbox[2] - bbox[0], bbox[3] - bbox[1]))
            blur = self._blur(image, kps)
            yaw = self._yaw(raw, kps)
            det_score = float(getattr(raw, "det_score", 0.0))
            quality = self._quality(size_px, blur, yaw, det_score)

            faces.append(
                DetectedFace(
                    bbox=(
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    ),
                    kps=kps,
                    det_score=det_score,
                    embedding=embedding,
                    size_px=size_px,
                    blur=blur,
                    yaw=yaw,
                    quality=quality,
                    source=source,
                )
            )

        return faces

    def analyze_poster(self, image: np.ndarray) -> list[DetectedFace]:
        """Detect faces with a global pass plus overlapping tiles when needed."""

        self.last_warnings = []
        height, width = image.shape[:2]

        if not self.config.tile or max(height, width) <= self.config.tile_trigger:
            faces = self.analyze(image, source="global")
        else:
            faces = self._analyze_global(image)
            faces.extend(self._analyze_tiles(image))
            faces = self._merge_duplicates(faces)

        if len(faces) > self.config.max_faces:
            faces.sort(
                key=lambda f: (f.det_score * f.size_px, f.det_score),
                reverse=True,
            )
            faces = faces[: self.config.max_faces]
            self.last_warnings.append(
                f"MAX_FACES_TRUNCATED: kept {self.config.max_faces} faces"
            )

        faces.sort(key=lambda f: (f.bbox[0], f.bbox[1]))
        return faces

    def _analyze_global(self, image: np.ndarray) -> list[DetectedFace]:
        height, width = image.shape[:2]
        scale = min(1.0, 1280.0 / max(height, width))
        if scale < 1.0:
            resized = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
        else:
            resized = image

        faces = self.analyze(resized, source="global")
        if scale < 1.0:
            for face in faces:
                x1, y1, x2, y2 = face.bbox
                face.bbox = (x1 / scale, y1 / scale, x2 / scale, y2 / scale)
                if face.kps is not None:
                    face.kps = face.kps / scale
                face.size_px /= scale
                # Recompute quality on the original pixels; the detector ran
                # on a downscaled copy, but the aligned crop can be sharper.
                face.blur = self._blur(image, face.kps)
                face.quality = self._quality(
                    face.size_px, face.blur, face.yaw, face.det_score
                )
        return faces

    def _analyze_tiles(self, image: np.ndarray) -> list[DetectedFace]:
        height, width = image.shape[:2]
        tile_size = min(self.config.tile_size, max(height, width))
        overlap = min(self.config.tile_overlap, tile_size - 1)

        xs, ys = self._tile_positions(width, height, tile_size, overlap)
        if len(xs) * len(ys) > self.config.max_tiles:
            original = tile_size
            while (
                len(xs) * len(ys) > self.config.max_tiles
                and tile_size < max(height, width)
            ):
                tile_size = min(max(height, width), int(math.ceil(tile_size * 1.25)))
                overlap = min(overlap, tile_size - 1)
                xs, ys = self._tile_positions(width, height, tile_size, overlap)
            self.last_warnings.append(
                "TILE_COUNT_CAPPED: "
                f"tile_size {original}->{tile_size}, tiles={len(xs) * len(ys)}"
            )

        faces: list[DetectedFace] = []
        for y in ys:
            for x in xs:
                tile = image[y : y + tile_size, x : x + tile_size]
                if tile.size == 0:
                    continue
                for face in self.analyze(tile, source="tile"):
                    x1, y1, x2, y2 = face.bbox
                    face.bbox = (x1 + x, y1 + y, x2 + x, y2 + y)
                    if face.kps is not None:
                        face.kps = face.kps + np.array([x, y], dtype=np.float32)
                    face.blur = self._blur(image, face.kps)
                    face.quality = self._quality(
                        face.size_px, face.blur, face.yaw, face.det_score
                    )
                    faces.append(face)
        return faces

    @staticmethod
    def _tile_positions(
        width: int, height: int, tile_size: int, overlap: int
    ) -> tuple[list[int], list[int]]:
        def positions(length: int) -> list[int]:
            if length <= tile_size:
                return [0]
            step = max(1, tile_size - overlap)
            values = list(range(0, length - tile_size + 1, step))
            if values[-1] != length - tile_size:
                values.append(length - tile_size)
            return values

        return positions(width), positions(height)

    @staticmethod
    def _center(bbox: tuple[float, float, float, float]) -> np.ndarray:
        return np.array(
            [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0],
            dtype=np.float32,
        )

    @classmethod
    def _same_face(cls, a: DetectedFace, b: DetectedFace) -> bool:
        ax1, ay1, ax2, ay2 = a.bbox
        bx1, by1, bx2, by2 = b.bbox
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        iou = inter / union if union > 0 else 0.0

        diag_a = math.hypot(ax2 - ax1, ay2 - ay1)
        diag_b = math.hypot(bx2 - bx1, by2 - by1)
        center_dist = float(np.linalg.norm(cls._center(a.bbox) - cls._center(b.bbox)))
        similar_size = 0.5 < (diag_a / max(diag_b, 1e-6)) < 2.0
        return iou > 0.35 or (
            similar_size and center_dist < 0.25 * min(diag_a, diag_b)
        )

    @classmethod
    def _merge_duplicates(cls, faces: list[DetectedFace]) -> list[DetectedFace]:
        ordered = sorted(
            faces,
            key=lambda f: (f.det_score * f.size_px, f.det_score),
            reverse=True,
        )
        kept: list[DetectedFace] = []
        for face in ordered:
            if not any(cls._same_face(face, other) for other in kept):
                kept.append(face)
        return kept

    @staticmethod
    def _blur(image: np.ndarray, kps: np.ndarray | None) -> float:
        if kps is None:
            return 0.0
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                aligned = face_align.norm_crop(
                    image, landmark=kps, image_size=112
                )
        except Exception:
            return 0.0
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _yaw(raw: object, kps: np.ndarray | None) -> float:
        pose = getattr(raw, "pose", None)
        if pose is not None:
            values = np.asarray(pose, dtype=np.float32).reshape(-1)
            if values.size >= 2 and np.isfinite(values[1]):
                return float(values[1])
        if kps is not None and len(kps) >= 3:
            eye_center = (kps[0] + kps[1]) / 2.0
            eye_dist = float(np.linalg.norm(kps[0] - kps[1])) + 1e-6
            return float(
                np.degrees(np.arctan2(kps[2][0] - eye_center[0], eye_dist))
            )
        return 0.0

    @staticmethod
    def _quality(size_px: float, blur: float, yaw: float, det_score: float) -> float:
        size_score = min(1.0, size_px / 96.0)
        blur_score = min(1.0, blur / 120.0)
        pose_score = max(0.15, 1.0 - abs(yaw) / 60.0)
        score = size_score * blur_score * pose_score * max(0.1, det_score)
        return float(max(0.0, min(1.0, score)))
