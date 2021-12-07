from typing import Any, Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment

from .tracklet import Tracklet


class Tracker:
    def __init__(self, distance_threshold: float = 0.05) -> None:
        """
        An object to maintain and update tracklets

        distance_threshold
            A cost score beyond which potential pairs are eliminated
        """
        self.active_tracklets: List[Tracklet] = []
        self.finished_tracklets: List[Tracklet] = []
        self.previous_boxes: np.ndarray = np.array([])
        self.distance_threshold: float = distance_threshold
        self.frame_index: int = 0

    def track(self, boxes: np.ndarray) -> None:
        """
        Update tracklets with boxes from a new frame

        boxes
            A `(n_boxes, 4)` array of bounding boxes
        """
        prev_indices = boxes_indices = []
        if len(boxes) > 0 and len(self.previous_boxes) > 0:
            # Pairwise cost: euclidean distance between boxes
            cost = np.linalg.norm(self.previous_boxes[:, None] - boxes[None], axis=-1)
            # Object matching
            prev_indices, boxes_indices = linear_sum_assignment(cost)
            mask = cost[prev_indices, boxes_indices] < self.distance_threshold
            prev_indices = prev_indices[mask]
            boxes_indices = boxes_indices[mask]
        # Add matches to active tracklets
        for prev_idx, box_idx in zip(prev_indices, boxes_indices):
            self.active_tracklets[prev_idx].add_box(boxes[box_idx])
        # Finalize lost tracklets
        lost_indices = set(range(len(self.active_tracklets))) - set(prev_indices)
        for lost_idx in sorted(lost_indices, reverse=True):
            self.finished_tracklets.append(self.active_tracklets.pop(lost_idx))
        # Activate new tracklets
        new_indices = set(range(len(boxes))) - set(boxes_indices)
        for new_idx in new_indices:
            self.active_tracklets.append(Tracklet(self.frame_index, boxes[new_idx]))
        # "Predict" next frame for comparison
        self.previous_boxes = np.stack(
            [tracklet.previous_box for tracklet in self.active_tracklets]
        )
        self.frame_index += 1

    @property
    def tracklets(self) -> List[Tracklet]:
        return sorted(
            self.finished_tracklets + self.active_tracklets, key=lambda t: t.start_index
        )

    def to_dict(self, fps: int, n_decimals: int = 3) -> Dict[str, Any]:
        return dict(
            fps=fps,
            tracklets=[t.to_dict(n_decimals=n_decimals) for t in self.tracklets],
        )
