from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("skimage")

from PIL import Image  # noqa: E402
from skimage import data  # noqa: E402

from facedec.config import PipelineConfig  # noqa: E402
from facedec.engine import FaceEngine  # noqa: E402
from facedec.pipeline import identify  # noqa: E402


@pytest.mark.slow
def test_same_face_matches_itself(tmp_path) -> None:
    if not os.environ.get("RUN_SLOW_TESTS"):
        pytest.skip("set RUN_SLOW_TESTS=1 to load the face models")

    rgb = data.astronaut()
    poster_path = tmp_path / "poster.jpg"
    portrait_path = tmp_path / "portrait.jpg"
    Image.fromarray(rgb).save(poster_path, quality=95)
    Image.fromarray(rgb).save(portrait_path, quality=95)

    cast = {
        "members": [
            {
                "cast_id": "c1",
                "name": "Test Actor",
                "portrait_paths": [str(portrait_path)],
            }
        ]
    }
    config = PipelineConfig(tile=False)
    engine = FaceEngine(config)
    result = identify(str(poster_path), cast, engine, config)

    assert len(result["faces"]) == 1
    face = result["faces"][0]
    assert face["match"] is not None
    assert face["match"]["cast_id"] == "c1"
    assert face["match"]["status"] == "accepted"


@pytest.mark.slow
def test_tiling_keeps_coordinates_and_deduplicates(tmp_path) -> None:
    if not os.environ.get("RUN_SLOW_TESTS"):
        pytest.skip("set RUN_SLOW_TESTS=1 to load the face models")

    rgb = data.astronaut()
    large = np.asarray(
        Image.fromarray(rgb).resize((2048, 2048), Image.Resampling.BICUBIC)
    )
    poster_path = tmp_path / "poster_large.jpg"
    portrait_path = tmp_path / "portrait.jpg"
    Image.fromarray(large).save(poster_path, quality=95)
    Image.fromarray(rgb).save(portrait_path, quality=95)

    cast = {
        "members": [
            {
                "cast_id": "c1",
                "name": "Test Actor",
                "portrait_paths": [str(portrait_path)],
            }
        ]
    }
    config = PipelineConfig(tile=True, tile_size=1024, tile_overlap=128)
    engine = FaceEngine(config)
    result = identify(str(poster_path), cast, engine, config)

    assert len(result["faces"]) == 1
    bbox = result["faces"][0]["bbox"]
    assert 0 <= bbox[0] < bbox[2] <= 2048
    assert 0 <= bbox[1] < bbox[3] <= 2048
    assert result["faces"][0]["match"]["status"] == "accepted"
