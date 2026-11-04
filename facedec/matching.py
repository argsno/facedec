from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import PipelineConfig
from .types import Candidate, DetectedFace, FaceMatch, GalleryMember

NEGATIVE = -1e6


def cosine_scores(
    faces: list[DetectedFace],
    members: list[GalleryMember],
) -> np.ndarray:
    """Return an F x C matrix; each entry is the best portrait similarity."""

    scores = np.full((len(faces), len(members)), -1.0, dtype=np.float32)
    if not faces or not members:
        return scores

    face_matrix = np.vstack([face.embedding for face in faces])
    for column, member in enumerate(members):
        if not member.embeddings:
            continue
        gallery_matrix = np.vstack(member.embeddings)
        scores[:, column] = (face_matrix @ gallery_matrix.T).max(axis=1)
    return scores


def _candidates(
    row: np.ndarray,
    members: list[GalleryMember],
    k: int = 3,
) -> list[Candidate]:
    order = np.argsort(-row)
    result: list[Candidate] = []
    for index in order[:k]:
        index = int(index)
        score = float(row[index])
        if score <= -0.5:
            continue
        member = members[index]
        result.append(
            Candidate(
                cast_id=member.cast_id,
                name=member.name,
                role=member.role,
                score=score,
            )
        )
    return result


def assign(
    scores: np.ndarray,
    config: PipelineConfig,
) -> tuple[dict[int, int | None], dict[int, int]]:
    """Hungarian assignment with explicit unknown / not-found dummy nodes.

    A face can be assigned to at most one cast member; a cast member can be
    assigned to at most one face. The dummy score is the review threshold, so
    a weak pair cannot win just because a cast slot is empty.
    """

    face_count, cast_count = scores.shape
    face_to_cast: dict[int, int | None] = {}
    cast_to_face: dict[int, int] = {}

    if face_count == 0:
        return face_to_cast, cast_to_face
    if cast_count == 0:
        return {face: None for face in range(face_count)}, cast_to_face

    unknown = (
        config.unknown_score
        if config.unknown_score is not None
        else config.review_threshold
    )

    size = face_count + cast_count
    matrix = np.full((size, size), NEGATIVE, dtype=np.float64)
    matrix[:face_count, :cast_count] = scores

    # Every face gets its own "unknown" column.
    for face in range(face_count):
        matrix[face, cast_count + face] = unknown
    # Every cast member gets its own "not found" row. Dummy rows can pair
    # with dummy columns at the same neutral score, so a strong real match
    # still beats "face unknown + cast not found".
    for cast in range(cast_count):
        matrix[face_count + cast, cast] = unknown
    matrix[face_count:, cast_count:] = unknown

    row_indices, column_indices = linear_sum_assignment(-matrix)
    column_to_row = {
        int(column): int(row)
        for row, column in zip(row_indices, column_indices)
    }

    for face in range(face_count):
        column = int(column_indices[face])
        face_to_cast[face] = column if column < cast_count else None

    for cast in range(cast_count):
        row = column_to_row.get(cast)
        if row is not None and row < face_count:
            cast_to_face[cast] = row

    return face_to_cast, cast_to_face


def decide(
    faces: list[DetectedFace],
    members: list[GalleryMember],
    scores: np.ndarray,
    face_to_cast: dict[int, int | None],
    config: PipelineConfig,
) -> list[FaceMatch]:
    matches: list[FaceMatch] = []

    for face_index, face in enumerate(faces):
        row = scores[face_index]
        candidates = _candidates(row, members, k=3)
        cast_index = face_to_cast.get(face_index)

        assigned: Candidate | None = None
        margin: float | None = None
        if cast_index is not None:
            member = members[cast_index]
            assigned = Candidate(
                cast_id=member.cast_id,
                name=member.name,
                role=member.role,
                score=float(row[cast_index]),
            )
            other_scores = [
                float(row[index])
                for index, other in enumerate(members)
                if index != cast_index and other.embeddings
            ]
            if other_scores:
                margin = assigned.score - max(other_scores)

        status = "unknown"
        reason: str | None = None

        if assigned is None:
            reason = "no_candidate_above_threshold"
        elif face.size_px < config.min_face_px:
            reason = "face_too_small"
        elif face.quality < config.min_quality:
            if assigned.score >= config.review_threshold:
                status = "review"
                reason = "low_quality"
            else:
                reason = "low_quality_below_review_threshold"
        elif assigned.score >= config.accept_threshold and (
            margin is None or margin >= config.margin_threshold
        ):
            status = "accepted"
        elif assigned.score >= config.review_threshold:
            status = "review"
            reason = (
                "low_margin"
                if margin is not None and margin < config.margin_threshold
                else "below_accept_threshold"
            )
        else:
            reason = "below_review_threshold"

        matches.append(
            FaceMatch(
                face=face,
                assigned=assigned,
                candidates=candidates,
                margin=margin,
                status=status,
                reason=reason,
            )
        )

    return matches
