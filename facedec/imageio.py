from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from .types import FaceMatch

# Reject decompression bombs before PIL allocates the full image.
Image.MAX_IMAGE_PIXELS = 60_000_000


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as BGR, honoring EXIF orientation and alpha."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in {"RGBA", "LA"} or (
            im.mode == "P" and "transparency" in im.info
        ):
            im = im.convert("RGBA")
            background = Image.new("RGB", im.size, (128, 128, 128))
            background.paste(im, mask=im.split()[-1])
            im = background
        else:
            im = im.convert("RGB")
        rgb = np.asarray(im, dtype=np.uint8)

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def save_image(path: str | Path, image_bgr: np.ndarray) -> None:
    """Write an image without relying on cv2's Unicode path handling."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".jpg"
    ok, buffer = cv2.imencode(suffix, image_bgr)
    if not ok:
        raise ValueError(f"could not encode image: {path}")
    buffer.tofile(str(path))


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_face_crops(
    image_bgr: np.ndarray,
    matches: list[FaceMatch],
    out_dir: str | Path,
) -> None:
    """Save one padded crop per detected face plus an index.json mapping."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    height, width = image_bgr.shape[:2]
    items: list[dict[str, Any]] = []

    for match in matches:
        face = match.face
        x1, y1, x2, y2 = (int(round(v)) for v in face.bbox)
        side = max(x2 - x1, y2 - y1)
        pad = int(0.55 * side)
        left = max(0, x1 - pad)
        top = max(0, y1 - pad)
        right = min(width, x2 + pad)
        bottom = min(height, y2 + pad)
        crop = image_bgr[top:bottom, left:right]
        if crop.size == 0:
            continue

        crop_h, crop_w = crop.shape[:2]
        square_side = max(crop_h, crop_w)
        square = np.full((square_side, square_side, 3), 255, dtype=np.uint8)
        offset_y = (square_side - crop_h) // 2
        offset_x = (square_side - crop_w) // 2
        square[offset_y : offset_y + crop_h, offset_x : offset_x + crop_w] = crop

        crop_path = out_dir / f"{face.face_id}.jpg"
        save_image(crop_path, square)

        if match.assigned is None or match.status == "unknown":
            match_payload = None
        else:
            match_payload = {
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

        items.append(
            {
                "face_id": face.face_id,
                "bbox": [round(float(v), 2) for v in face.bbox],
                "crop": str(crop_path),
                "match": match_payload,
            }
        )

    save_json(out_dir / "index.json", {"faces": items})


def draw_debug(
    image_bgr: np.ndarray,
    matches: list[FaceMatch],
    path: str | Path,
) -> None:
    """Draw boxes and short ASCII labels. Chinese names are intentionally omitted."""

    canvas = image_bgr.copy()
    colors = {
        "accepted": (0, 180, 0),
        "review": (0, 165, 255),
        "unknown": (0, 0, 220),
    }
    for match in matches:
        face = match.face
        x1, y1, x2, y2 = (int(round(v)) for v in face.bbox)
        color = colors.get(match.status, (200, 200, 200))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        if match.assigned is not None:
            label = f"{face.face_id} {match.assigned.cast_id} {match.assigned.score:.2f}"
        else:
            label = f"{face.face_id} unknown"

        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        top = max(0, y1 - th - baseline - 4)
        cv2.rectangle(
            canvas,
            (x1, top),
            (x1 + tw + 6, top + th + baseline + 4),
            color,
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (x1 + 3, top + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    save_image(path, canvas)
