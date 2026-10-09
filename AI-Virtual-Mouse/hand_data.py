"""hand_data.py - plain data container for one detected hand (no MediaPipe import)."""
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

# MediaPipe hand landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20


@dataclass
class HandData:
    """21 landmarks in PIXEL coordinates (x, y) plus z scaled to pixels."""
    points: np.ndarray            # shape (21, 3)
    frame_w: int
    frame_h: int
    raw: Optional[Any] = None     # original MediaPipe landmark list (for drawing)

    def xy(self, idx: int) -> Tuple[float, float]:
        return float(self.points[idx, 0]), float(self.points[idx, 1])

    def dist2d(self, a: int, b: int) -> float:
        return float(np.hypot(*(self.points[a, :2] - self.points[b, :2])))

    @property
    def palm_size(self) -> float:
        """Wrist -> middle MCP distance: a scale reference that grows/shrinks with the hand."""
        return self.dist2d(WRIST, MIDDLE_MCP)

    @property
    def palm_center_y(self) -> float:
        return float(self.points[[WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP], 1].mean())
