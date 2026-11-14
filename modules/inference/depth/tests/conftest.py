import sys
from pathlib import Path
 
import cv2
import numpy as np
import pytest
 
DEPTH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPTH_ROOT))                  # so "from src import ..." works
 
 
@pytest.fixture
def rgb_image():
    """A small synthetic indoor-ish RGB frame (floor, wall, a box), 240 x 320."""
    h, w = 240, 320
    img = np.zeros((h, w, 3), np.uint8)
    img[: h // 2] = (200, 205, 215)                                    # wall
    rows = np.linspace(90, 170, h - h // 2, dtype=np.uint8)[:, None]
    img[h // 2:] = np.stack([rows, rows * 0 + 120, rows // 2 + 40], axis=-1).repeat(w, axis=1)  # floor
    cv2.rectangle(img, (120, 110), (210, 200), (150, 60, 40), -1)       # a box on the floor
    cv2.rectangle(img, (120, 110), (210, 125), (190, 90, 60), -1)       # its lit top
    return np.ascontiguousarray(img)