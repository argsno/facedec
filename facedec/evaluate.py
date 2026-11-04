from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

import numpy as np

from .config import PipelineConfig
from .engine import FaceEngine
from .gallery import build_gallery
from .imageio import save_json
from .pipeline import identify, load_cast


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def greedy_match(
    ground_truth: list[dict],
    detections: list[dict],
    threshold: float = 0.5,
) -> dict[int, int]:
    """Match GT boxes to detections by IoU, one-to-one, highest first."""

    pairs: list[tuple[float, int, int]] = []
    for gt_index, gt in enumerate(ground_truth):
        for det_index, detection in enumerate(detections):
            score = iou(gt["bbox"], detection["bbox"])
            if score >= threshold:
                pairs.append((score, gt_index, det_index))
    pairs.sort(reverse=True)

    used_gt: set[int] = set()
    used_det: set[int] = set()
    matches: dict[int, int] = {}
    for _, gt_index, det_index in pairs:
        if gt_index in used_gt or det_index in used_det:
            continue
        used_gt.add(gt_index)
        used_det.add(det_index)
        matches[gt_index] = det_index
    return matches


def _aggregate(poster_reports: list[dict]) -> dict:
    def total(key: str) -> int:
        return sum(int(report[key]) for report in poster_reports)

    gt_faces = total("gt_faces")
    matched_gt = total("matched_gt")
    labeled = total("labeled_faces")
    detected_labeled = total("detected_labeled")
    accepted = total("accepted")
    accepted_correct = total("accepted_correct")
    review = total("review")
    unknown = total("unknown")
    top1_correct = total("top1_correct")
    unknown_gt = total("unknown_gt_faces")
    unknown_gt_false_matches = total("unknown_gt_false_matches")
    latencies = [report["latency_ms"] for report in poster_reports]

    def ratio(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 4) if denominator else None

    return {
        "posters": len(poster_reports),
        "gt_faces": gt_faces,
        "detected_faces": total("detected_faces"),
        "matched_gt": matched_gt,
        "detection_recall": ratio(matched_gt, gt_faces),
        "labeled_faces": labeled,
        "detected_labeled": detected_labeled,
        "labeled_detection_recall": ratio(detected_labeled, labeled),
        "top1_correct": top1_correct,
        "top1_accuracy": ratio(top1_correct, detected_labeled),
        "accepted": accepted,
        "accepted_correct": accepted_correct,
        "auto_accept_precision": ratio(accepted_correct, accepted),
        "coverage": ratio(accepted, labeled),
        "review": review,
        "review_rate": ratio(review, labeled),
        "unknown": unknown,
        "unknown_rate": ratio(unknown, labeled),
        "unknown_gt_faces": unknown_gt,
        "unknown_gt_false_matches": unknown_gt_false_matches,
        "unknown_gt_false_match_rate": ratio(unknown_gt_false_matches, unknown_gt),
        "latency_ms_mean": round(mean(latencies), 1) if latencies else None,
        "latency_ms_p95": round(float(np.percentile(latencies, 95)), 1)
        if latencies
        else None,
    }


def evaluate(
    manifest_path: str | Path,
    engine: FaceEngine,
    config: PipelineConfig,
    iou_threshold: float = 0.5,
) -> dict:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    posters = manifest.get("posters") if isinstance(manifest, dict) else None
    if not isinstance(posters, list) or not posters:
        raise ValueError("manifest.posters must be a non-empty list")

    started = time.perf_counter()
    gallery_build_ms = 0.0
    reports: list[dict] = []
    gallery_cache: dict[str, tuple[list, list[dict]]] = {}

    for entry in posters:
        poster_path = Path(entry["poster"]).expanduser()
        if not poster_path.is_absolute():
            poster_path = manifest_path.parent / poster_path

        cast_source = entry["cast"]
        if isinstance(cast_source, str):
            cast_path = Path(cast_source).expanduser()
            if not cast_path.is_absolute():
                cast_path = manifest_path.parent / cast_path
            cast = load_cast(cast_path)
            cast_key = f"path:{cast_path.resolve()}"
        elif isinstance(cast_source, dict):
            cast = cast_source
            cast_key = "inline:" + json.dumps(
                cast_source, sort_keys=True, ensure_ascii=False
            )
        else:
            raise ValueError("each poster needs cast as a path or object")

        if cast_key not in gallery_cache:
            gallery_started = time.perf_counter()
            members, gallery_warnings = build_gallery(engine, cast, config)
            gallery_build_ms += (time.perf_counter() - gallery_started) * 1000.0
            gallery_cache[cast_key] = (members, gallery_warnings)
        members, gallery_warnings = gallery_cache[cast_key]

        ground_truth = entry.get("faces", [])
        if not isinstance(ground_truth, list):
            raise ValueError("faces must be a list")

        result = identify(
            str(poster_path),
            cast,
            engine,
            config,
            members=members,
            gallery_warnings=gallery_warnings,
        )
        detections = result["faces"]
        matches = greedy_match(ground_truth, detections, threshold=iou_threshold)

        report = {
            "poster": str(poster_path),
            "gt_faces": len(ground_truth),
            "detected_faces": len(detections),
            "matched_gt": len(matches),
            "labeled_faces": 0,
            "detected_labeled": 0,
            "top1_correct": 0,
            "accepted": 0,
            "accepted_correct": 0,
            "review": 0,
            "unknown": 0,
            "unknown_gt_faces": 0,
            "unknown_gt_false_matches": 0,
            "latency_ms": result["timings_ms"]["total_ms"],
        }

        for gt_index, gt in enumerate(ground_truth):
            det_index = matches.get(gt_index)
            detection = detections[det_index] if det_index is not None else None
            match = detection.get("match") if detection else None
            status = match["status"] if match else "unknown"
            expected = gt.get("cast_id")

            if expected is None:
                report["unknown_gt_faces"] += 1
                if match is not None and status in {"accepted", "review"}:
                    report["unknown_gt_false_matches"] += 1
                continue

            report["labeled_faces"] += 1
            if detection is None:
                continue
            report["detected_labeled"] += 1

            predicted = match["cast_id"] if match else None
            correct = predicted == expected
            if correct:
                report["top1_correct"] += 1

            if status == "accepted":
                report["accepted"] += 1
                if correct:
                    report["accepted_correct"] += 1
            elif status == "review":
                report["review"] += 1
            else:
                report["unknown"] += 1

        reports.append(report)

    aggregate = _aggregate(reports)
    aggregate["gallery_build_ms"] = round(gallery_build_ms, 1)
    aggregate["wall_time_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    aggregate["latency_scope"] = "poster only; excludes cached gallery build"

    return {
        "manifest": str(manifest_path),
        "iou_threshold": iou_threshold,
        "config": {
            "accept_threshold": config.accept_threshold,
            "review_threshold": config.review_threshold,
            "margin_threshold": config.margin_threshold,
            "min_face_px": config.min_face_px,
        },
        "aggregate": aggregate,
        "per_poster": reports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="facedec-eval",
        description="Evaluate the feasibility harness on a labeled poster manifest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="eval.json")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--model", default="buffalo_l")
    parser.add_argument("--no-tile", action="store_true")
    parser.add_argument("--accept-threshold", type=float, default=0.45)
    parser.add_argument("--review-threshold", type=float, default=0.30)
    parser.add_argument("--margin-threshold", type=float, default=0.08)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = PipelineConfig(
            model_name=args.model,
            tile=not args.no_tile,
            accept_threshold=args.accept_threshold,
            review_threshold=args.review_threshold,
            margin_threshold=args.margin_threshold,
        )
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")
        print(f"Loading {config.model_name} on CPU...", file=sys.stderr)
        engine = FaceEngine(config)
        result = evaluate(manifest_path, engine, config, iou_threshold=args.iou)
        save_json(args.out, result)
        print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
        print(f"result: {args.out}", file=sys.stderr)
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
