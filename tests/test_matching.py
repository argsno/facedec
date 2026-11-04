from __future__ import annotations

import numpy as np
import pytest

from facedec.config import PipelineConfig
from facedec.matching import assign, cosine_scores, decide
from facedec.types import DetectedFace, GalleryMember


def make_face(embedding: np.ndarray, size_px: float = 120.0, quality: float = 0.8) -> DetectedFace:
    vector = np.asarray(embedding, dtype=np.float32)
    vector = vector / np.linalg.norm(vector)
    return DetectedFace(
        bbox=(0.0, 0.0, size_px, size_px),
        kps=None,
        det_score=0.95,
        embedding=vector,
        size_px=size_px,
        blur=200.0,
        yaw=0.0,
        quality=quality,
    )


def make_member(cast_id: str, embedding: np.ndarray) -> GalleryMember:
    vector = np.asarray(embedding, dtype=np.float32)
    vector = vector / np.linalg.norm(vector)
    return GalleryMember(cast_id=cast_id, name=cast_id, embeddings=[vector], quality=0.8)


def test_cosine_scores_uses_best_portrait() -> None:
    a1 = np.array([1.0, 0.0, 0.0])
    a2 = np.array([0.9, 0.1, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    members = [
        GalleryMember(cast_id="a", name="A", embeddings=[a1, a2]),
        make_member("b", b),
    ]
    face = make_face([0.95, 0.05, 0.0])

    scores = cosine_scores([face], members)

    assert scores.shape == (1, 2)
    assert scores[0, 0] > 0.99
    assert scores[0, 1] < 0.1


def test_assignment_does_not_assign_one_actor_to_two_faces() -> None:
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    members = [make_member("a", a), make_member("b", b)]
    faces = [make_face(a), make_face(a)]

    scores = cosine_scores(faces, members)
    face_to_cast, cast_to_face = assign(scores, PipelineConfig())

    assert list(face_to_cast.values()).count(0) == 1
    assert 1 not in face_to_cast.values()
    assigned_face = next(
        face for face, cast in face_to_cast.items() if cast == 0
    )
    assert cast_to_face == {0: assigned_face}


def test_assignment_leaves_weak_face_unknown() -> None:
    a = np.array([1.0, 0.0, 0.0])
    unrelated = np.array([0.0, 0.0, 1.0])
    members = [make_member("a", a)]
    face = make_face(unrelated)

    scores = cosine_scores([face], members)
    face_to_cast, cast_to_face = assign(scores, PipelineConfig())

    assert face_to_cast == {0: None}
    assert cast_to_face == {}


def test_decision_accepts_a_clear_match() -> None:
    a = np.array([1.0, 0.0, 0.0])
    members = [make_member("a", a)]
    face = make_face(a)
    scores = cosine_scores([face], members)
    face_to_cast, _ = assign(scores, PipelineConfig())

    matches = decide([face], members, scores, face_to_cast, PipelineConfig())

    assert matches[0].status == "accepted"
    assert matches[0].assigned is not None
    assert matches[0].assigned.cast_id == "a"


def test_decision_rejects_too_small_face() -> None:
    a = np.array([1.0, 0.0, 0.0])
    members = [make_member("a", a)]
    face = make_face(a, size_px=20.0)
    scores = cosine_scores([face], members)
    face_to_cast, _ = assign(scores, PipelineConfig())

    matches = decide([face], members, scores, face_to_cast, PipelineConfig())

    assert matches[0].status == "unknown"
    assert matches[0].reason == "face_too_small"
