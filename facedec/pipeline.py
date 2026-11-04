from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .engine import FaceEngine
from .gallery import build_gallery
from .imageio import draw_debug, load_image, save_face_crops
from .matching import assign, cosine_scores, decide
from .types import FaceMatch, GalleryMember


def load_cast(path: str | Path) -> dict:
    cast_path = Path(path)
    with cast_path.open("r", encoding="utf-8") as handle:
        cast = json.load(handle)
    if not isinstance(cast, dict):
        raise ValueError("cast JSON must be an object")
    resolve_cast_paths(cast, cast_path.parent)
    return cast


def resolve_cast_paths(cast: dict, base_dir: str | Path) -> None:
    """Resolve relative portrait paths next to the cast JSON file."""

    base_dir = Path(base_dir)
    members = cast.get("members")
    if not isinstance(members, list):
        return
    for member in members:
        paths = member.get("portrait_paths") or []
        if isinstance(paths, str):
            paths = [paths]
        resolved: list[str] = []
        for path in paths:
            candidate = Path(path).expanduser()
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            resolved.append(str(candidate))
        member["portrait_paths"] = resolved


def _face_payload(match: FaceMatch) -> dict[str, Any]:
    payload = match.face.to_dict()
    if match.assigned is None or match.status == "unknown":
        payload["match"] = None
    else:
        payload["match"] = {
            "cast_id": match.assigned.cast_id,
            "name": match.assigned.name,
            "role": match.assigned.role,
            "score": round(float(match.assigned.score), 4),
            "margin": None
            if match.margin is None
            else round(float(match.margin), 4),
            "status": match.status,
            "reason": match.reason,
            "candidates": [
                {
                    "cast_id": candidate.cast_id,
                    "name": candidate.name,
                    "role": candidate.role,
                    "score": round(float(candidate.score), 4),
                }
                for candidate in match.candidates
            ],
        }
    payload["confidence"] = {
        "accepted": "high",
        "review": "medium",
        "unknown": "low",
    }.get(match.status, "low")
    return payload


def _member_payload(member: GalleryMember) -> dict[str, Any]:
    return {
        "cast_id": member.cast_id,
        "name": member.name,
        "role": member.role,
        "status": member.status,
        "usable_images": len(member.embeddings),
        "portrait_count": member.portrait_count,
        "quality": round(float(member.quality), 4),
        "warnings": list(member.warnings),
    }


def identify(
    poster_path: str | Path,
    cast: dict,
    engine: FaceEngine,
    config: PipelineConfig,
    debug_path: str | Path | None = None,
    faces_dir: str | Path | None = None,
    members: list[GalleryMember] | None = None,
    gallery_warnings: list[dict] | None = None,
) -> dict[str, Any]:
    """Run the full offline pipeline once. No queue and no calibration.

    ``members`` lets a batch caller reuse a prebuilt gallery; when omitted the
    gallery is built from ``cast`` (the CLI/API path).
    """

    timings: dict[str, float] = {}
    started = time.perf_counter()

    poster = load_image(poster_path)
    poster_height, poster_width = poster.shape[:2]
    timings["load_poster_ms"] = (time.perf_counter() - started) * 1000.0

    gallery_started = time.perf_counter()
    if members is None:
        members, warnings = build_gallery(engine, cast, config)
        gallery_cached = False
    else:
        warnings = list(gallery_warnings or [])
        gallery_cached = True
    timings["gallery_ms"] = (time.perf_counter() - gallery_started) * 1000.0

    detect_started = time.perf_counter()
    faces = engine.analyze_poster(poster)
    for index, face in enumerate(faces, start=1):
        face.face_id = f"f_{index}"
    timings["detect_embed_ms"] = (time.perf_counter() - detect_started) * 1000.0

    match_started = time.perf_counter()
    scores = cosine_scores(faces, members)
    face_to_cast, cast_to_face = assign(scores, config)
    matches = decide(faces, members, scores, face_to_cast, config)
    timings["match_ms"] = (time.perf_counter() - match_started) * 1000.0

    warnings.extend(
        {"code": "ENGINE_WARNING", "message": message}
        for message in engine.last_warnings
    )
    if not faces:
        warnings.append(
            {
                "code": "NO_FACE_DETECTED",
                "message": "no faces passed the detector threshold",
            }
        )

    resolved_cast_ids = {
        match.assigned.cast_id
        for match in matches
        if match.assigned is not None and match.status != "unknown"
    }
    cast_not_found: list[dict[str, Any]] = []
    for member in members:
        if member.cast_id in resolved_cast_ids:
            continue
        reason = "gallery_unusable" if not member.embeddings else "not_matched"
        cast_not_found.append(
            {
                "cast_id": member.cast_id,
                "name": member.name,
                "role": member.role,
                "reason": reason,
                "status": member.status,
            }
        )

    unknown_faces = [
        {
            "face_id": match.face.face_id,
            "bbox": [round(float(v), 2) for v in match.face.bbox],
            "reason": match.reason,
            "candidates": [
                {
                    "cast_id": candidate.cast_id,
                    "name": candidate.name,
                    "score": round(float(candidate.score), 4),
                }
                for candidate in match.candidates
            ],
        }
        for match in matches
        if match.status == "unknown"
    ]

    timings["total_ms"] = (time.perf_counter() - started) * 1000.0

    result: dict[str, Any] = {
        "poster": {
            "path": str(poster_path),
            "width": int(poster_width),
            "height": int(poster_height),
        },
        "model": {
            "name": config.model_name,
            "det_size": config.det_size,
            "providers": ["CPUExecutionProvider"],
        },
        "config": asdict(config),
        "gallery_cached": gallery_cached,
        "cast": [_member_payload(member) for member in members],
        "faces": [_face_payload(match) for match in matches],
        "unknown_faces": unknown_faces,
        "cast_not_found": cast_not_found,
        "warnings": warnings,
        "timings_ms": {
            key: round(float(value), 1) for key, value in timings.items()
        },
    }

    if debug_path is not None:
        draw_debug(poster, matches, debug_path)
        result["debug_image"] = str(debug_path)

    if faces_dir is not None:
        save_face_crops(poster, matches, faces_dir)
        result["faces_dir"] = str(faces_dir)

    return result
