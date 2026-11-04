from __future__ import annotations

import pytest

from facedec.evaluate import _aggregate, greedy_match, iou


def test_iou() -> None:
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [10, 10, 20, 20]) == 0.0
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == 1 / 3


def test_greedy_match_is_one_to_one() -> None:
    ground_truth = [{"bbox": [0, 0, 10, 10]}, {"bbox": [20, 20, 30, 30]}]
    detections = [
        {"bbox": [0, 0, 10, 10]},
        {"bbox": [1, 1, 11, 11]},
        {"bbox": [21, 21, 31, 31]},
    ]

    matches = greedy_match(ground_truth, detections, threshold=0.5)

    assert matches[0] == 0
    assert matches[1] == 2
    assert len(set(matches.values())) == 2


def test_aggregate_metrics() -> None:
    reports = [
        {
            "gt_faces": 2,
            "detected_faces": 2,
            "matched_gt": 2,
            "labeled_faces": 2,
            "detected_labeled": 2,
            "top1_correct": 1,
            "accepted": 1,
            "accepted_correct": 1,
            "review": 1,
            "unknown": 0,
            "unknown_gt_faces": 0,
            "unknown_gt_false_matches": 0,
            "latency_ms": 100.0,
        },
        {
            "gt_faces": 1,
            "detected_faces": 1,
            "matched_gt": 1,
            "labeled_faces": 1,
            "detected_labeled": 1,
            "top1_correct": 0,
            "accepted": 0,
            "accepted_correct": 0,
            "review": 0,
            "unknown": 1,
            "unknown_gt_faces": 1,
            "unknown_gt_false_matches": 0,
            "latency_ms": 200.0,
        },
    ]

    aggregate = _aggregate(reports)

    assert aggregate["detection_recall"] == 1.0
    assert aggregate["top1_accuracy"] == pytest.approx(1 / 3, abs=1e-4)
    assert aggregate["auto_accept_precision"] == 1.0
    assert aggregate["coverage"] == pytest.approx(1 / 3, abs=1e-4)
    assert aggregate["review_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert aggregate["unknown_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert aggregate["unknown_gt_false_match_rate"] == 0.0
    assert aggregate["latency_ms_mean"] == 150.0
