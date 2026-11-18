import numpy as np
from r6.common import make_windows, BASE_FEATURES

def test_window_count():
    x = np.zeros((100, len(BASE_FEATURES)), dtype=np.float32)
    w = make_windows(x, window=75, stride=15)
    assert w.shape == (2, 75, len(BASE_FEATURES))

def test_window_short():
    x = np.zeros((20, len(BASE_FEATURES)), dtype=np.float32)
    assert make_windows(x, window=75, stride=15).shape[0] == 0
