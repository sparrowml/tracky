import numpy as np
from scipy.optimize import linear_sum_assignment


class MODA:
    def __init__(
        self, false_negatives: int, false_positives: int, n_truth: int
    ) -> None:
        self.false_negatives = false_negatives
        self.false_positives = false_positives
        self.n_truth = n_truth

    def __add__(self, other: "MODA") -> "MODA":
        return MODA(
            false_negatives=self.false_negatives + other.false_negatives,
            false_positives=self.false_positives + other.false_positives,
            n_truth=self.n_truth + other.n_truth,
        )

    @property
    def value(self) -> float:
        n_errors = abs(self.false_negatives) + abs(self.false_positives)
        if self.n_truth == 0:
            return 0
        return 1 - n_errors / self.n_truth


def compute_moda(
    predicted_boxes: np.ndarray,
    ground_truth_boxes: np.ndarray,
    iou_threshold: float = 0.5,
) -> MODA:
    if len(predicted_boxes) == 0:
        n = len(ground_truth_boxes)
        return MODA(false_negatives=n, false_positives=0, n_truth=n)
    elif len(ground_truth_boxes) == 0:
        return MODA(false_negatives=0, false_positives=len(predicted_boxes), n_truth=0)
    x1 = np.maximum(predicted_boxes[:, None, 0], ground_truth_boxes[None, :, 0])
    y1 = np.maximum(predicted_boxes[:, None, 1], ground_truth_boxes[None, :, 1])
    x2 = np.minimum(predicted_boxes[:, None, 2], ground_truth_boxes[None, :, 2])
    y2 = np.minimum(predicted_boxes[:, None, 3], ground_truth_boxes[None, :, 3])
    inner_box = np.stack([x1, y1, x2, y2], -1)
    intersection = np.maximum(inner_box[..., 2] - inner_box[..., 0], 0) * np.maximum(
        inner_box[..., 3] - inner_box[..., 1], 0
    )

    a_area = (predicted_boxes[:, 2] - predicted_boxes[:, 0]) * (
        predicted_boxes[:, 3] - predicted_boxes[:, 1]
    )
    b_area = (ground_truth_boxes[:, 2] - ground_truth_boxes[:, 0]) * (
        ground_truth_boxes[:, 3] - ground_truth_boxes[:, 1]
    )
    total_area = a_area[:, None] + b_area[None, :]
    union = total_area - intersection
    cost = 1 - intersection / union
    pred_indices, gt_indices = linear_sum_assignment(cost)

    false_positives = set(np.arange(len(predicted_boxes))) - set(pred_indices)
    false_negatives = set(np.arange(len(ground_truth_boxes))) - set(gt_indices)

    unmatched = cost[pred_indices, gt_indices] > iou_threshold
    false_positives |= set(pred_indices[unmatched])
    false_negatives |= set(gt_indices[unmatched])

    return MODA(
        false_negatives=len(false_negatives),
        false_positives=len(false_positives),
        n_truth=len(ground_truth_boxes),
    )
