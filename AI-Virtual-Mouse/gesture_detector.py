"""
gesture_detector.py - turns hand landmarks into stable gestures and one-shot events.

This module is PURE LOGIC: it never touches the mouse or the camera and does not
import MediaPipe.  Time is passed in (`now`), so it can be unit-tested.

Gesture map (MOUSE mode)
    index fingertip            -> move cursor (always)
    thumb + index pinch        -> LEFT CLICK   (fires instantly, once per pinch)
    thumb + MIDDLE pinch       -> RIGHT CLICK  (fires instantly, once per pinch)
Gesture map (DRAWING mode)
    index up, middle folded    -> pen DOWN  (thumb is ignored)
    anything else              -> pen UP

Every click needs N consecutive frames, has hysteresis on its distance threshold,
a short cooldown, and must be released before it can fire again.
"""
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

import numpy as np

import config
from hand_data import (HandData, WRIST, THUMB_TIP, INDEX_MCP, INDEX_PIP, INDEX_TIP,
                       MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP, RING_MCP, RING_PIP,
                       PINKY_MCP, PINKY_PIP)

RING_TIP, PINKY_TIP = 16, 20


class Mode(Enum):
    MOUSE = "MOUSE"
    DRAWING = "DRAWING"


class Gesture(Enum):
    NONE = "NONE"
    MOVE = "MOVE"
    LEFT_CLICK = "LEFT CLICK"
    RIGHT_CLICK = "RIGHT CLICK"
    DRAWING = "DRAWING"
    PEN_UP = "PEN UP"


class EventType(Enum):
    LEFT_CLICK = auto()
    RIGHT_CLICK = auto()
    PINCH_END = auto()      # the pinch that produced a click was released


@dataclass
class GestureEvent:
    type: EventType
    anchor_time: float = 0.0   # click where the cursor was at this moment (just before the fingers closed)


@dataclass
class GestureResult:
    hand_present: bool = False
    gesture: Gesture = Gesture.NONE
    pinch_distance_px: float = 0.0
    pinch_ratio: float = 0.0
    pinching: bool = False          # a click pinch is currently held (left or right)
    pen_down: bool = False          # drawing mode: pen touching the canvas
    tracking_lost: bool = False     # hand missing longer than the grace time
    fingers: tuple = (False, False, False, False)   # index, middle, ring, pinky extended
    events: List[GestureEvent] = field(default_factory=list)


def _angle(a, b, c) -> float:
    """Angle at b (degrees) between segments b->a and b->c."""
    v1, v2 = a - b, c - b
    n = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if n < 1e-6:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2)) / n))))


def finger_extended(p: np.ndarray, mcp: int, pip: int, tip: int) -> bool:
    """Straight at the PIP joint AND tip farther from the wrist than the PIP joint."""
    straight = _angle(p[mcp], p[pip], p[tip]) > config.FINGER_EXTENDED_ANGLE
    reach = np.linalg.norm(p[tip] - p[WRIST]) > np.linalg.norm(p[pip] - p[WRIST])
    return bool(straight and reach)


# ---------------------------------------------------------------- pinch FSM
IDLE, CONFIRMING, FIRED, BLOCKED = range(4)


class _PinchFSM:
    """
    One click-pinch.  IDLE -> CONFIRMING (N frames) -> FIRED (click sent, waits for release) -> IDLE.
    A pinch that starts while a cooldown / the other pinch is active goes to BLOCKED and is
    ignored until it is released, so a click can never fire "late" or twice.
    update() returns "fire", "end" or None.
    """

    def __init__(self, confirm_frames: int, release_frames: int, cooldown: float):
        self.confirm, self.release, self.cooldown = confirm_frames, release_frames, cooldown
        self.reset()

    def reset(self):
        self.state = IDLE
        self.on = 0
        self.off = 0
        self.last_fire = -1e9

    @property
    def active(self) -> bool:
        return self.state in (CONFIRMING, FIRED)

    def update(self, raw: bool, allowed: bool, now: float) -> Optional[str]:
        if self.state == IDLE:
            if not raw:
                return None
            if not allowed or now - self.last_fire < self.cooldown:
                self.state, self.off = BLOCKED, 0
                return None
            self.state, self.on = CONFIRMING, 0
        if self.state == CONFIRMING:
            if not raw:
                self.state, self.on = IDLE, 0           # glitch: pinch vanished before confirming
                return None
            self.on += 1
            if self.on >= self.confirm:
                self.state, self.off, self.last_fire = FIRED, 0, now
                return "fire"
            return None
        if self.state == FIRED:
            self.off = 0 if raw else self.off + 1
            if self.off >= self.release:
                self.state = IDLE
                return "end"
            return None
        # BLOCKED
        self.off = 0 if raw else self.off + 1
        if self.off >= self.release:
            self.state = IDLE
        return None


class GestureDetector:
    def __init__(self):
        self.mode = Mode.MOUSE
        self._full_reset()

    # ------------------------------------------------------------- state
    def _full_reset(self):
        self._left = _PinchFSM(config.PINCH_CONFIRM_FRAMES, config.PINCH_RELEASE_FRAMES, config.CLICK_COOLDOWN)
        self._right = _PinchFSM(config.RIGHT_CLICK_CONFIRM_FRAMES, config.PINCH_RELEASE_FRAMES,
                                config.RIGHT_CLICK_COOLDOWN)
        self._left_raw = False
        self._right_raw = False
        self._ti_open = -1e9        # last time thumb & index were clearly apart
        self._tm_open = -1e9        # last time thumb & middle were clearly apart

        self._pen_down = False
        self._pen_on = 0
        self._pen_off = 0

        self._last_seen: Optional[float] = None
        self._flash = Gesture.NONE
        self._flash_time = -1e9
        self._last_result = GestureResult()

    def reset(self) -> List[GestureEvent]:
        """Abort everything (hand lost / mode switch). Clicks are instant, so nothing is left pending."""
        ev = [GestureEvent(EventType.PINCH_END)] if (self._left.active or self._right.active) else []
        self._full_reset()
        return ev

    def set_mode(self, mode: Mode) -> List[GestureEvent]:
        ev = self.reset()
        self.mode = mode
        return ev

    # ------------------------------------------------------------ update
    def update(self, hand: Optional[HandData], now: float) -> GestureResult:
        if hand is None or hand.palm_size < config.MIN_PALM_SIZE_PX:
            return self._no_hand(now)
        self._last_seen = now

        p = hand.points
        palm = hand.palm_size
        d_ti = hand.dist2d(THUMB_TIP, INDEX_TIP) / palm
        d_tm = hand.dist2d(THUMB_TIP, MIDDLE_TIP) / palm
        idx = finger_extended(p, INDEX_MCP, INDEX_PIP, INDEX_TIP)
        mid = finger_extended(p, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP)
        ring = finger_extended(p, RING_MCP, RING_PIP, RING_TIP)
        pnk = finger_extended(p, PINKY_MCP, PINKY_PIP, PINKY_TIP)
        reach = float(np.linalg.norm(p[INDEX_TIP] - p[WRIST])) / palm
        fist = (not mid) and (not ring) and (not pnk) and reach < config.FIST_REACH_RATIO

        res = GestureResult(hand_present=True, pinch_distance_px=d_ti * palm,
                            pinch_ratio=d_ti, fingers=(idx, mid, ring, pnk))

        if self.mode == Mode.DRAWING:
            self._update_drawing(res, idx, mid)
        else:
            self._update_mouse(res, now, d_ti, d_tm, fist)

        res.gesture = self._label(now)
        self._last_result = res
        return res

    def _no_hand(self, now: float) -> GestureResult:
        grace_over = self._last_seen is None or (now - self._last_seen) > config.HAND_LOST_GRACE_TIME
        res = GestureResult(hand_present=False, tracking_lost=grace_over)
        if not grace_over:                       # short tracking drop-out: keep state, do nothing
            last = self._last_result
            res.gesture, res.pinching, res.pen_down = last.gesture, last.pinching, last.pen_down
            return res
        busy = self._left.state != IDLE or self._right.state != IDLE or self._pen_down
        if busy:
            res.events = self.reset()            # hand left: clear every state
        self._last_seen = None
        res.gesture = Gesture.NONE
        self._last_result = res
        return res

    # ----------------------------------------------------------- drawing
    def _update_drawing(self, res: GestureResult, idx: bool, mid: bool):
        """Pen down = index finger up + middle finger folded.  Thumb, ring, pinky are ignored."""
        if idx and not mid:
            self._pen_on += 1
            self._pen_off = 0
            if self._pen_on >= config.PEN_CONFIRM_FRAMES:
                self._pen_down = True
        else:
            self._pen_off += 1
            self._pen_on = 0
            if self._pen_off >= config.PEN_RELEASE_FRAMES:
                self._pen_down = False
        res.pen_down = self._pen_down

    # ------------------------------------------------------------- mouse
    def _update_mouse(self, res: GestureResult, now: float, d_ti: float, d_tm: float, fist: bool):
        ev = res.events

        # remember when each finger pair was last clearly open -> click anchor time
        if d_ti >= config.CLICK_APPROACH_RATIO:
            self._ti_open = now
        if d_tm >= config.CLICK_APPROACH_RATIO:
            self._tm_open = now

        # raw pinches with hysteresis. The two pinches exclude each other:
        # left = thumb is closer to the index, right = thumb is clearly closer to the middle finger.
        lthr = config.PINCH_RELEASE_THRESHOLD if self._left_raw else config.PINCH_THRESHOLD
        self._left_raw = bool(d_ti < lthr and d_ti <= d_tm and (self._left_raw or not fist))
        rthr = config.PINCH_RELEASE_THRESHOLD if self._right_raw else config.PINCH_THRESHOLD
        margin = 0.0 if self._right_raw else config.MIDDLE_PINCH_MARGIN
        self._right_raw = bool(d_tm < rthr and d_tm + margin < d_ti and (self._right_raw or not fist))

        out = self._left.update(self._left_raw, not self._right.active, now)
        if out == "fire":
            ev.append(GestureEvent(EventType.LEFT_CLICK, self._anchor_time(self._ti_open, now)))
            self._flash, self._flash_time = Gesture.LEFT_CLICK, now
        elif out == "end":
            ev.append(GestureEvent(EventType.PINCH_END))

        out = self._right.update(self._right_raw, not self._left.active, now)
        if out == "fire":
            ev.append(GestureEvent(EventType.RIGHT_CLICK, self._anchor_time(self._tm_open, now)))
            self._flash, self._flash_time = Gesture.RIGHT_CLICK, now
        elif out == "end":
            ev.append(GestureEvent(EventType.PINCH_END))

        res.pinching = self._left.active or self._right.active

    @staticmethod
    def _anchor_time(open_time: float, now: float) -> float:
        """Aim point = where the cursor was when the fingers were last open, within sane limits."""
        return min(max(open_time, now - config.CLICK_ANCHOR_MAX_LOOKBACK),
                   now - config.CLICK_ANCHOR_MIN_LOOKBACK)

    # ------------------------------------------------------------- label
    def _label(self, now: float) -> Gesture:
        if self.mode == Mode.DRAWING:
            return Gesture.DRAWING if self._pen_down else Gesture.PEN_UP
        if now - self._flash_time < config.ACTION_LABEL_SECONDS:
            return self._flash
        return Gesture.MOVE
