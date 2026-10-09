"""drawing.py - virtual pen: a persistent canvas blended over the webcam frame.

Smoothness comes from two stages:
  1. an adaptive EMA on the fingertip (strong when the finger is almost still = no tremor,
     weak when it moves fast = no lag)
  2. every stroke is drawn as quadratic curves through the midpoints of consecutive points,
     so corners are rounded instead of looking like a jagged polyline.
"""
import math
from typing import Optional, Tuple

import cv2
import numpy as np

import config
from utils import AdaptiveEMA2D

SHIFT = 3                      # sub-pixel precision for cv2.polylines (coords * 2**SHIFT)
_SCALE = 1 << SHIFT


class DrawingCanvas:
    def __init__(self):
        self.canvas: Optional[np.ndarray] = None
        self.color_index = config.DRAW_COLOR_INDEX
        self._smoother = AdaptiveEMA2D(config.DRAW_SMOOTH_SLOW, config.DRAW_SMOOTH_FAST,
                                       config.DRAW_SMOOTH_SLOW_DIST, config.DRAW_SMOOTH_FAST_DIST)
        self._last: Optional[np.ndarray] = None       # last smoothed point
        self._mid: Optional[np.ndarray] = None        # where the ink currently ends

    @property
    def color(self):
        return config.DRAW_COLORS[self.color_index]

    def set_color(self, index: int):
        if 0 <= index < len(config.DRAW_COLORS):
            self.color_index = index

    def ensure_size(self, frame: np.ndarray):
        if self.canvas is None or self.canvas.shape != frame.shape:
            self.canvas = np.zeros_like(frame)        # separate layer: drawing persists across frames
            self._last = self._mid = None
            self._smoother.reset()

    def clear(self):
        if self.canvas is not None:
            self.canvas[:] = 0
        self._last = self._mid = None
        self._smoother.reset()

    def pen_up(self):
        """End the stroke: finish the ink up to the last point, then forget it (no stray lines)."""
        if self._last is not None and self._mid is not None and self.canvas is not None:
            self._poly([self._mid, self._last])
        self._last = self._mid = None
        self._smoother.reset()

    # ------------------------------------------------------------ drawing
    def _poly(self, pts):
        arr = (np.array(pts, dtype=np.float64) * _SCALE).round().astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(self.canvas, [arr], False, self.color, config.DRAW_THICKNESS, cv2.LINE_AA, SHIFT)

    def _dot(self, p):
        r = max(config.DRAW_THICKNESS // 2, 1)
        cv2.circle(self.canvas, (int(round(p[0])), int(round(p[1]))), r, self.color, -1, cv2.LINE_AA)

    def update(self, point: Optional[Tuple[float, float]], pen_down: bool):
        if point is None or not pen_down or self.canvas is None:
            self.pen_up()
            return
        cur = np.array(self._smoother.update(*point), dtype=np.float64)
        if self._last is None:                        # stroke starts: just a dot
            self._dot(cur)
            self._last = self._mid = cur
            return
        dist = math.hypot(*(cur - self._last))
        if dist > config.DRAW_MAX_JUMP_PX:            # tracking glitch: start a new stroke
            self.pen_up()
            self._smoother.set(*point)
            self._dot(np.array(point, dtype=np.float64))
            self._last = self._mid = np.array(point, dtype=np.float64)
            return
        if dist < config.DRAW_MIN_MOVE_PX:            # hand tremor: ignore
            return
        mid = (self._last + cur) / 2.0
        # quadratic Bezier: from the previous midpoint, bending towards the previous point, to the new midpoint
        t = np.linspace(0.0, 1.0, config.DRAW_CURVE_STEPS + 1)[:, None]
        curve = (1 - t) ** 2 * self._mid + 2 * (1 - t) * t * self._last + t ** 2 * mid
        self._poly(curve)
        self._dot(mid)                                # round cap
        self._last, self._mid = cur, mid

    def render(self, frame: np.ndarray) -> np.ndarray:
        """frame + canvas via cv2.addWeighted, applied only where ink exists."""
        if self.canvas is None:
            return frame
        ink = self.canvas.any(axis=2)
        if not ink.any():
            return frame
        blended = cv2.addWeighted(frame, 1.0 - config.DRAW_OVERLAY_ALPHA,
                                  self.canvas, config.DRAW_OVERLAY_ALPHA, 0)
        out = frame.copy()
        out[ink] = blended[ink]
        return out
