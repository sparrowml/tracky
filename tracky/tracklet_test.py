import json

import numpy as np

from .tracklet import Tracklet


def test_tracklet_accepts_1d_array():
    tracklet = Tracklet(0, np.random.randn(4))
    assert tracklet.boxes.shape == (1, 4)


def test_tracklet_accepts_2d_array():
    tracklet = Tracklet(0, np.random.randn(1, 4))
    assert tracklet.boxes.shape == (1, 4)


def test_tracklet_add_box_concatenates():
    tracklet = Tracklet(0, np.random.randn(4))
    tracklet.add_box(np.random.randn(4))
    assert tracklet.boxes.shape == (2, 4)


def test_tracklet_previous_box_is_last_box_added():
    tracklet = Tracklet(0, np.random.randn(4))
    tracklet.add_box(np.zeros(4))
    tracklet.add_box(np.ones(4))
    np.testing.assert_equal(tracklet.previous_box, np.ones(4))


def test_to_dict_can_be_json_serialized():
    tracklet = Tracklet(0, np.random.randn(4))
    json.dumps(tracklet.to_dict(n_decimals=1))
