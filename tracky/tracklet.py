from typing import Any, Dict

import numpy as np


class Tracklet:
    def __init__(self, start_index: int, box: np.ndarray) -> None:
        """
        An object to maintain a single tracklet

        start_index
            The frame index that starts the tracklet
        box
            A NumPy array with shape (4,)
        """
        self.start_index = start_index
        self.boxes = box.reshape(1, 4)

    def add_box(self, box: np.ndarray) -> None:
        """Append a box to the end of the array"""
        self.boxes = np.concatenate(self.boxes, box.reshape(1, 4))

    @property
    def previous_box(self) -> np.ndarray:
        return self.boxes[-1]

    def to_dict(self, n_decimals: int = 3) -> Dict[str, Any]:
        """Return a dictionary of the tracklet that can be serialized"""
        return dict(
            start=self.start_index,
            boxes=np.round(self.boxes, n_decimals).tolist(),
        )
