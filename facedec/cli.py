from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import PipelineConfig
from .engine import FaceEngine
from .imageio import save_json
from .pipeline import identify, load_cast


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="facedec",
        description="CPU-only poster cast identification (feasibility harness)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--poster", required=True, help="poster image path")
    parser.add_argument("--cast", required=True, help="cast JSON path")
    parser.add_argument("--out", default="result.json", help="result JSON path")
    parser.add_argument("--debug", default=None, help="optional debug overlay path")
    parser.add_argument(
        "--faces-dir",
        default=None,
        help="optional directory for per-face crops and index.json",
    )

    parser.add_argument("--model", default="buffalo_l", help="InsightFace model pack")
    parser.add_argument("--det-size", type=int, default=640, help="detector input size")
    parser.add_argument(
        "--tile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable overlapping tiles for large posters",
    )
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=int, default=128)
    parser.add_argument("--tile-trigger", type=int, default=1600)
    parser.add_argument("--max-tiles", type=int, default=24)
    parser.add_argument("--min-face-px", type=float, default=48.0)
    parser.add_argument("--max-faces", type=int, default=64)

    parser.add_argument(
        "--accept-threshold",
        type=float,
        default=0.45,
        help="raw cosine threshold for auto-accept (uncalibrated placeholder)",
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=0.30,
        help="raw cosine threshold for human review",
    )
    parser.add_argument("--margin-threshold", type=float, default=0.08)
    parser.add_argument("--min-quality", type=float, default=0.10)
    parser.add_argument(
        "--unknown-score",
        type=float,
        default=None,
        help="Hungarian dummy score; defaults to review_threshold",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = PipelineConfig(
            model_name=args.model,
            det_size=args.det_size,
            tile=args.tile,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            tile_trigger=args.tile_trigger,
            max_tiles=args.max_tiles,
            min_face_px=args.min_face_px,
            max_faces=args.max_faces,
            accept_threshold=args.accept_threshold,
            review_threshold=args.review_threshold,
            margin_threshold=args.margin_threshold,
            min_quality=args.min_quality,
            unknown_score=args.unknown_score,
        )
        cast = load_cast(args.cast)
        if not Path(args.poster).exists():
            raise FileNotFoundError(f"poster not found: {args.poster}")

        print(f"Loading {config.model_name} on CPU...", file=sys.stderr)
        engine = FaceEngine(config)
        result = identify(
            args.poster,
            cast,
            engine,
            config,
            debug_path=args.debug,
            faces_dir=args.faces_dir,
        )

        if args.out == "-":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            save_json(args.out, result)
            print(f"result: {args.out}", file=sys.stderr)
        if args.debug:
            print(f"debug:  {args.debug}", file=sys.stderr)
        if args.faces_dir:
            print(f"faces:  {args.faces_dir}", file=sys.stderr)

        accepted = sum(1 for face in result["faces"] if face["match"] and face["match"]["status"] == "accepted")
        review = sum(1 for face in result["faces"] if face["match"] and face["match"]["status"] == "review")
        unknown = len(result["unknown_faces"])
        print(
            f"faces={len(result['faces'])} accepted={accepted} "
            f"review={review} unknown={unknown} "
            f"cast_not_found={len(result['cast_not_found'])} "
            f"time={result['timings_ms']['total_ms']:.0f}ms",
            file=sys.stderr,
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
