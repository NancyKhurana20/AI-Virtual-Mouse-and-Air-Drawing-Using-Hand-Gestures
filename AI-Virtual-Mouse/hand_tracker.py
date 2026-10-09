"""hand_tracker.py - webcam capture + MediaPipe Hands (mp.solutions.hands, v0.10.21)."""
import sys
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

import config
from hand_data import HandData


class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened or stops delivering frames."""


class Camera:
    def __init__(self):
        self._cap = None
        self._failures = 0

    def open(self):
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]
        for backend in backends:
            cap = cv2.VideoCapture(config.CAMERA_INDEX, backend)
            if cap is not None and cap.isOpened():
                self._cap = cap
                break
            if cap is not None:
                cap.release()
        if self._cap is None:
            raise CameraError("Unable to open webcam.\nPlease check your camera connection.")
        if config.CAMERA_FOURCC:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.CAMERA_FOURCC))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep latency low (ignored by some drivers)
        return self

    def read(self) -> np.ndarray:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._failures += 1
            if self._failures >= config.CAMERA_MAX_READ_FAILURES:
                raise CameraError("Lost connection to the webcam.\nPlease check your camera connection.")
            return None
        self._failures = 0
        return cv2.flip(frame, 1) if config.MIRROR_FRAME else frame

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self._hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MAX_NUM_HANDS,
            model_complexity=config.MODEL_COMPLEXITY,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )

    def process(self, frame_bgr: np.ndarray) -> Optional[HandData]:
        """BGR frame -> RGB -> MediaPipe -> HandData in pixel coordinates (or None)."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)   # MediaPipe expects RGB, OpenCV gives BGR
        rgb.flags.writeable = False                        # lets MediaPipe avoid a copy
        results = self._hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None
        lm = results.multi_hand_landmarks[0]
        pts = np.array([[p.x * w, p.y * h, p.z * w] for p in lm.landmark], dtype=np.float32)
        return HandData(points=pts, frame_w=w, frame_h=h, raw=lm)

    def draw(self, frame: np.ndarray, hand: HandData):
        self.mp_draw.draw_landmarks(
            frame, hand.raw, self.mp_hands.HAND_CONNECTIONS,
            self.mp_styles.get_default_hand_landmarks_style(),
            self.mp_styles.get_default_hand_connections_style())

    def close(self):
        self._hands.close()
