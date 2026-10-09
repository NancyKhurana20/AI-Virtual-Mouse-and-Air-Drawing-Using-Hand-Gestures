"""
mouse_controller.py - the only module that talks to PyAutoGUI.

* maps the index fingertip (camera pixels) to screen pixels
* smooths it with an adaptive exponential moving average
* sends left / right clicks exactly where the user AIMED (before the fingers closed)
* holds the cursor still while a click-pinch is held, so the click never "slides" off the target
"""
import math
from collections import deque
from typing import Iterable, Optional, Tuple

import pyautogui

import config
from gesture_detector import EventType, GestureEvent
from utils import AdaptiveEMA2D

pyautogui.PAUSE = 0          # PyAutoGUI sleeps 0.1 s after every call by default - far too slow
pyautogui.FAILSAFE = True    # slam the PHYSICAL mouse into the top-left corner to abort


class MouseController:
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()
        scale = self.screen_w / 1920.0
        self._smoother = AdaptiveEMA2D(config.SMOOTHING_FACTOR_SLOW, config.SMOOTHING_FACTOR,
                                       config.SMOOTHING_SLOW_DISTANCE * scale,
                                       config.SMOOTHING_FAST_DISTANCE * scale)
        self._slop = config.PINCH_STICKY_RADIUS * scale
        self._pos: Optional[Tuple[int, int]] = None           # last position sent to the OS
        self._history = deque(maxlen=60)                      # (time, x, y) of the smoothed cursor
        self._sticky: Optional[Tuple[float, float]] = None    # cursor is held here while a click-pinch is held
        self._last_click: Optional[Tuple[float, str, Tuple[float, float]]] = None   # (time, button, pos)

    # ---------------------------------------------------------- cursor
    def map_to_screen(self, x: float, y: float, fw: int, fh: int) -> Tuple[float, float]:
        """Active camera region (frame minus margin) -> whole screen."""
        mx, my = fw * config.ACTIVE_MARGIN, fh * config.ACTIVE_MARGIN
        nx = (x - mx) / max(fw - 2 * mx, 1)
        ny = (y - my) / max(fh - 2 * my, 1)
        e = config.CURSOR_SCREEN_EDGE
        sx = min(max(nx * self.screen_w, e), self.screen_w - 1 - e)
        sy = min(max(ny * self.screen_h, e), self.screen_h - 1 - e)
        return sx, sy

    def move_cursor(self, finger_xy: Tuple[float, float], frame_size: Tuple[int, int], now: float):
        """Call every frame with the index-fingertip."""
        tx, ty = self.map_to_screen(finger_xy[0], finger_xy[1], *frame_size)
        sx, sy = self._smoother.update(tx, ty)

        if self._sticky is not None:
            ax, ay = self._sticky
            if math.hypot(sx - ax, sy - ay) <= self._slop:
                sx, sy = ax, ay                       # tiny drift while pinching: hold still
            else:
                self._sticky = None                   # deliberate movement: follow the finger again
                self._smoother.set(ax, ay)            # glide out of the anchor, no jump
                sx, sy = self._smoother.update(tx, ty)
        self._history.append((now, sx, sy))
        self._apply(sx, sy)

    def reset_tracking(self):
        """Hand lost: next sample jumps straight to the finger."""
        self._smoother.reset()
        self._sticky = None

    def _apply(self, x: float, y: float):
        p = (int(round(x)), int(round(y)))
        if p != self._pos:
            pyautogui.moveTo(p[0], p[1], duration=0)
            self._pos = p

    def _position_at(self, t: float) -> Optional[Tuple[float, float]]:
        best = None
        for ts, x, y in self._history:
            if ts <= t:
                best = (x, y)
            else:
                break
        if best is None and self._history:
            best = self._history[0][1:]
        return best

    # ---------------------------------------------------------- events
    def handle_events(self, events: Iterable[GestureEvent], now: float):
        for e in events:
            if e.type in (EventType.LEFT_CLICK, EventType.RIGHT_CLICK):
                button = "left" if e.type == EventType.LEFT_CLICK else "right"
                anchor = self._position_at(e.anchor_time) or self._pos
                self._click(anchor, button, now)
            elif e.type == EventType.PINCH_END:
                self._release_sticky()

    def _click(self, anchor, button: str, now: float):
        """Click where the user AIMED; keep the cursor parked there until the pinch is released."""
        if anchor is None:
            pyautogui.click(button=button)
            return
        # 2nd click right after the 1st at (almost) the same place -> same exact pixel,
        # so the OS reliably treats two quick left pinches as a double click.
        lc = self._last_click
        if (lc and lc[1] == button and now - lc[0] <= config.DOUBLE_CLICK_SNAP_TIME
                and math.hypot(anchor[0] - lc[2][0], anchor[1] - lc[2][1]) <= self._slop):
            anchor = lc[2]
        self._apply(*anchor)
        pyautogui.click(button=button)
        self._last_click = (now, button, anchor)
        self._sticky = anchor

    def _release_sticky(self):
        if self._sticky is not None:
            self._smoother.set(*self._sticky)         # ease from the click spot to the finger, no jump
            self._sticky = None

    # ---------------------------------------------------------- safety
    def release_all(self):
        """Always safe to call: clears state and never leaves a mouse button pressed."""
        self._sticky = None
        try:
            pyautogui.mouseUp()
        except Exception:
            pass
