import json

import numpy as np

from .tracker import Tracker


def test_inf_threshold_always_matches():
    tracker = Tracker(distance_threshold=np.inf)
    tracker.track(np.zeros((1, 4)))
    tracker.track(np.ones((1, 4)))
    assert len(tracker.tracklets) == 1, "Boxes should get matched"


def test_zero_threshold_always_matches():
    tracker = Tracker(distance_threshold=0)
    tracker.track(np.zeros((1, 4)))
    tracker.track(np.ones((1, 4)))
    assert len(tracker.tracklets) == 2, "Boxes shouldn't get matched"


def test_to_dict_can_be_json_serialized():
    tracker = Tracker()
    tracker.track(np.random.randn(10, 4))
    tracker.track(np.random.randn(4, 4))
    json.dumps(tracker.to_dict(fps=25, n_decimals=1))
