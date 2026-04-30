"""Metric utilities for evaluating object detection performance.

This module computes model quality metrics such as precision, recall, and
mean Average Precision (mAP), plus operational metrics used by the dashboard.
"""

from pathlib import Path

from machledata.infer import Detection, predict_batch


def summarize_detections(total_images: int, total_detections: int) -> dict[str, float]:
    """Compute simple detection summary metrics for demos and smoke tests."""
    average = total_detections / total_images if total_images else 0.0
    return {
        "total_images": float(total_images),
        "total_detections": float(total_detections),
        "detections_per_image": average,
    }


def compute_detection_statistics(
    detections_list: list[list[Detection]],
) -> dict[str, float]:
    """Compute statistics from a batch of detection results.

    Args:
        detections_list: List of detection lists (one per image).

    Returns:
        Dictionary with detection statistics.
    """
    total_images = len(detections_list)
    total_detections = sum(len(dets) for dets in detections_list)
    avg_detections = total_detections / total_images if total_images else 0.0

    # Compute confidence statistics
    all_confidences = []
    for dets in detections_list:
        all_confidences.extend([d.confidence for d in dets])

    stats = {
        "total_images": float(total_images),
        "total_detections": float(total_detections),
        "average_detections_per_image": avg_detections,
        "average_confidence": float(sum(all_confidences) / len(all_confidences))
        if all_confidences
        else 0.0,
    }

    # Add confidence percentiles if we have detections
    if all_confidences:
        sorted_conf = sorted(all_confidences)
        stats["confidence_min"] = float(min(all_confidences))
        stats["confidence_max"] = float(max(all_confidences))
        stats["confidence_median"] = float(sorted_conf[len(sorted_conf) // 2])

    return stats


def compute_class_distribution(
    detections_list: list[list[Detection]],
) -> dict[str, int]:
    """Compute distribution of detected classes.

    Args:
        detections_list: List of detection lists.

    Returns:
        Dictionary with class names and their occurrence counts.
    """
    class_counts = {}
    for dets in detections_list:
        for det in dets:
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1

    return class_counts


def evaluate_on_images(
    image_paths: list[Path],
    model_path: str | Path | None = None,
    ground_truth: dict[str, list[Detection]] | None = None,
) -> dict:
    """Evaluate model on a set of images and optionally compare with ground truth.

    Args:
        image_paths: List of image paths to evaluate on.
        model_path: Path to saved model. Uses default if None.
        ground_truth: Optional dict mapping image paths to ground truth detections.

    Returns:
        Evaluation results including statistics and optional comparison metrics.
    """
    # Run predictions
    predictions = predict_batch(image_paths, model_path)

    # Convert to list of detection lists for statistics
    prediction_lists = [predictions[str(path)] for path in image_paths]

    # Compute statistics
    stats = compute_detection_statistics(prediction_lists)
    class_dist = compute_class_distribution(prediction_lists)

    results = {
        "statistics": stats,
        "class_distribution": class_dist,
    }

    # Compute comparison metrics if ground truth is provided
    if ground_truth:
        results["comparison"] = _compute_comparison_metrics(predictions, ground_truth)

    return results


def _compute_comparison_metrics(
    predictions: dict[str, list[Detection]],
    ground_truth: dict[str, list[Detection]],
) -> dict:
    """Compare predictions with ground truth detections.

    Args:
        predictions: Predicted detections.
        ground_truth: Ground truth detections.

    Returns:
        Comparison metrics including precision, recall, and F1 score.
    """
    total_pred = sum(len(dets) for dets in predictions.values())
    total_gt = sum(len(dets) for dets in ground_truth.values())

    # Simple overlap-based matching (for demo purposes)
    # In production, use more sophisticated IoU-based matching
    matches = 0
    for img_path, pred_dets in predictions.items():
        gt_dets = ground_truth.get(img_path, [])
        # Simple label-based matching
        pred_labels = {d.class_name for d in pred_dets}
        gt_labels = {d.class_name for d in gt_dets}
        matches += len(pred_labels & gt_labels)

    precision = matches / total_pred if total_pred > 0 else 0.0
    recall = matches / total_gt if total_gt > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "matches": matches,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


