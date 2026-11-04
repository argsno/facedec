from __future__ import annotations

import json

import numpy as np

from facedec.imageio import save_face_crops
from facedec.types import Candidate, DetectedFace, FaceMatch


def test_save_face_crops_writes_crop_and_index(tmp_path) -> None:
    image = np.full((200, 200, 3), 128, dtype=np.uint8)
    face = DetectedFace(
        bbox=(50.0, 50.0, 150.0, 150.0),
        kps=None,
        det_score=0.9,
        embedding=np.ones(3, dtype=np.float32),
        size_px=100.0,
        blur=100.0,
        yaw=0.0,
        quality=0.8,
        face_id="f_1",
    )
    match = FaceMatch(
        face=face,
        assigned=Candidate("c1", "Actor A", "Role A", 0.5),
        candidates=[Candidate("c1", "Actor A", "Role A", 0.5)],
        margin=0.1,
        status="review",
    )

    save_face_crops(image, [match], tmp_path)

    assert (tmp_path / "f_1.jpg").exists()
    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert index["faces"][0]["face_id"] == "f_1"
    assert index["faces"][0]["match"]["cast_id"] == "c1"
