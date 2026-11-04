from __future__ import annotations

import itertools

import numpy as np

from .config import PipelineConfig
from .engine import FaceEngine
from .imageio import load_image
from .types import GalleryMember


def build_gallery(
    engine: FaceEngine,
    cast: dict,
    config: PipelineConfig,
) -> tuple[list[GalleryMember], list[dict]]:
    """Turn caller-provided portraits into a deduplicated embedding gallery.

    Precision-first rules:
    - skip a portrait when its largest face is too small;
    - skip a group portrait when a second face is comparable in size;
    - flag an identity whose own images disagree strongly.
    """

    items = cast.get("members")
    if not isinstance(items, list) or not items:
        raise ValueError("cast.members must be a non-empty list")

    members: list[GalleryMember] = []
    warnings: list[dict] = []
    seen_ids: set[str] = set()

    for item in items:
        cast_id = str(item.get("cast_id", "")).strip()
        name = str(item.get("name", "")).strip()
        role = item.get("role")
        if not cast_id or not name:
            raise ValueError("every cast member needs cast_id and name")
        if cast_id in seen_ids:
            raise ValueError(f"duplicate cast_id: {cast_id}")
        seen_ids.add(cast_id)

        portrait_paths = item.get("portrait_paths") or []
        if isinstance(portrait_paths, str):
            portrait_paths = [portrait_paths]

        member = GalleryMember(cast_id=cast_id, name=name, role=role)
        qualities: list[float] = []

        if not portrait_paths:
            member.warnings.append("no_portrait_paths")

        for path in portrait_paths:
            member.portrait_count += 1
            try:
                image = load_image(path)
            except Exception as exc:  # noqa: BLE001 - report data errors, don't crash
                member.warnings.append(f"portrait_load_failed:{path}:{exc}")
                continue

            faces = engine.analyze(image, source="gallery")
            if not faces:
                member.warnings.append(f"no_face:{path}")
                continue

            faces.sort(key=lambda f: (f.size_px, f.det_score), reverse=True)
            best = faces[0]

            if best.size_px < config.min_face_px:
                member.warnings.append(
                    f"face_too_small:{path}:{best.size_px:.0f}px"
                )
                continue

            if len(faces) > 1 and faces[1].size_px >= 0.6 * best.size_px:
                member.warnings.append(f"ambiguous_group_portrait:{path}")
                continue

            member.embeddings.append(best.embedding)
            qualities.append(best.quality)

        if not member.embeddings:
            member.status = "unusable"
        else:
            member.quality = float(np.mean(qualities))
            if len(member.embeddings) >= 2:
                similarities = [
                    float(np.dot(member.embeddings[i], member.embeddings[j]))
                    for i, j in itertools.combinations(range(len(member.embeddings)), 2)
                ]
                mean_similarity = float(np.mean(similarities))
                if mean_similarity < 0.30:
                    member.status = "inconsistent"
                    member.warnings.append(
                        f"gallery_inconsistent:mean_cosine={mean_similarity:.2f}"
                    )
            if member.quality < 0.15 and member.status == "ok":
                member.status = "low_quality"
                member.warnings.append(
                    f"gallery_low_quality:quality={member.quality:.2f}"
                )

        members.append(member)
        for message in member.warnings:
            warnings.append(
                {
                    "code": "GALLERY_WARNING",
                    "cast_id": cast_id,
                    "message": message,
                }
            )

    return members, warnings
