"""
main.py - AI Virtual Mouse (Image & Video Processing project)

Pipeline per frame:
    Camera -> flip -> BGR->RGB -> MediaPipe hand landmarks -> gesture detection
           -> mouse action  OR  drawing -> HUD -> display

Gestures:  index finger = move | thumb+index pinch = left click | thumb+middle pinch = right click
Drawing (press D):  index finger up = draw | index+middle up = lift pen
Keys (camera window must have focus):  D drawing mode | C clear | Q quit | H help | 1-5 pen colour
"""
import sys
import time

import cv2
import numpy as np

import config
from gesture_detector import Gesture, GestureDetector, Mode
from hand_data import INDEX_TIP, MIDDLE_TIP, THUMB_TIP
from hand_tracker import Camera, CameraError, HandTracker
from drawing import DrawingCanvas
from utils import FPSCounter

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREEN, RED, YELLOW, ORANGE = (255, 255, 255), (0, 255, 0), (0, 0, 255), (0, 255, 255), (0, 165, 255)

HELP_LINES = [
    "AI VIRTUAL MOUSE", "",
    "Index Finger  -> Move Cursor",
    "Thumb + Index Pinch  -> Left Click",
    "Thumb + Middle Pinch  -> Right Click", "",
    "Press D for Drawing Mode:",
    "  Index finger up  -> Draw",
    "  Index + Middle up  -> Lift pen", "",
    "D -> Draw/Mouse   C -> Clear Drawing",
    "Q -> Quit         H -> Hide this help",
]


class VirtualMouseApp:
    def __init__(self, tracker, mouse=None):
        self.tracker = tracker
        self.mouse = mouse                       # None = display only (e.g. tests)
        self.detector = GestureDetector()
        self.drawing = DrawingCanvas()
        self.fps = FPSCounter(config.FPS_AVERAGE_FRAMES)
        self.mode = Mode.MOUSE
        self.start = time.perf_counter()
        self.show_help = True
        self.fps_value = 0.0

    # ------------------------------------------------------------ modes
    def toggle_drawing(self):
        new = Mode.DRAWING if self.mode == Mode.MOUSE else Mode.MOUSE
        self._switch_mode(new)

    def _switch_mode(self, new: Mode):
        if self.mouse:
            self.mouse.release_all()             # never carry a pressed button across modes
            self.mouse.reset_tracking()
        self.detector.set_mode(new)
        self.drawing.pen_up()
        self.mode = new

    # ------------------------------------------------------------ frame
    def process_frame(self, frame: np.ndarray, now: float) -> np.ndarray:
        self.fps_value = self.fps.tick()
        self.drawing.ensure_size(frame)
        hand = self.tracker.process(frame)
        result = self.detector.update(hand, now)

        if self.mode == Mode.MOUSE and self.mouse:
            if hand is not None:
                self.mouse.move_cursor(hand.xy(INDEX_TIP), (hand.frame_w, hand.frame_h), now)
            if result.tracking_lost:
                self.mouse.reset_tracking()
            self.mouse.handle_events(result.events, now)
        elif self.mode == Mode.DRAWING:
            if hand is not None:
                self.drawing.update(hand.xy(INDEX_TIP), result.pen_down)
            elif not result.pen_down:            # a 1-2 frame tracking drop-out keeps the stroke alive
                self.drawing.pen_up()

        display = self.drawing.render(frame)
        self._draw_overlay(display, hand, result, now)
        return display

    # -------------------------------------------------------------- HUD
    def _draw_overlay(self, img, hand, result, now):
        h, w = img.shape[:2]
        if self.mode == Mode.MOUSE:
            m = config.ACTIVE_MARGIN
            cv2.rectangle(img, (int(w * m), int(h * m)), (int(w * (1 - m)), int(h * (1 - m))), (255, 0, 255), 2)

        if hand is not None:
            self.tracker.draw(img, hand)
            ix, iy = map(int, hand.xy(INDEX_TIP))
            tx, ty = map(int, hand.xy(THUMB_TIP))
            mx, my = map(int, hand.xy(MIDDLE_TIP))
            if self.mode == Mode.DRAWING:
                # pen indicator: filled in the pen colour while drawing, hollow ring while the pen is lifted
                if result.pen_down:
                    cv2.circle(img, (ix, iy), 9, self.drawing.color, -1)
                else:
                    cv2.circle(img, (ix, iy), 9, WHITE, 2)
            else:
                left = result.gesture == Gesture.LEFT_CLICK
                right = result.gesture == Gesture.RIGHT_CLICK
                cv2.line(img, (tx, ty), (ix, iy), GREEN if left else WHITE, 2)
                cv2.line(img, (tx, ty), (mx, my), ORANGE if right else (120, 120, 120), 2 if right else 1)
                cv2.circle(img, (ix, iy), 8, YELLOW, -1)
                cv2.circle(img, (tx, ty), 7, (255, 0, 0), -1)

        # info panel (semi-transparent)
        lines = [("AI VIRTUAL MOUSE", WHITE),
                 (f"FPS: {self.fps_value:.0f}", WHITE),
                 ("Hand: Detected" if result.hand_present else "Hand: Not detected",
                  GREEN if result.hand_present else RED),
                 (f"Mode: {self.mode.value}", YELLOW),
                 (f"Gesture: {result.gesture.value}", WHITE)]
        if result.hand_present and self.mode == Mode.MOUSE:
            lines.append((f"Pinch Distance: {int(result.pinch_distance_px)} px", WHITE))
        self._panel(img, 8, 8, 290, 12 + 24 * len(lines))
        for i, (text, col) in enumerate(lines):
            cv2.putText(img, text, (16, 30 + 24 * i), FONT, 0.58, col, 2 if i == 0 else 1, cv2.LINE_AA)

        footer = ("Index up: Draw | Index+Middle up: Lift pen | C: Clear | D: Mouse | Q: Quit"
                  if self.mode == Mode.DRAWING else "D: Draw  C: Clear  H: Help  Q: Quit")
        cv2.putText(img, footer, (10, h - 12), FONT, 0.5, WHITE, 1, cv2.LINE_AA)

        if self.show_help and now - self.start < config.INSTRUCTION_SECONDS:
            bh = 24 * len(HELP_LINES) + 20
            x0, y0 = (w - 400) // 2, max((h - bh) // 2, 0)
            self._panel(img, x0, y0, 400, bh, 0.75)
            for i, t in enumerate(HELP_LINES):
                cv2.putText(img, t, (x0 + 15, y0 + 30 + 24 * i), FONT, 0.55, YELLOW if i == 0 else WHITE, 1, cv2.LINE_AA)

    @staticmethod
    def _panel(img, x, y, w, h, alpha=0.55):
        roi = img[y:y + h, x:x + w]
        if roi.size:
            roi[:] = cv2.addWeighted(roi, 1 - alpha, np.zeros_like(roi), alpha, 0)

    # -------------------------------------------------------------- keys
    def handle_key(self, key: int) -> bool:
        """Returns False when the app should quit."""
        if key in (ord("q"), ord("Q"), 27):
            return False
        if key in (ord("d"), ord("D")):
            self.toggle_drawing()
        elif key in (ord("c"), ord("C")):
            self.drawing.clear()
        elif key in (ord("h"), ord("H")):
            self.show_help = not self.show_help
            self.start = time.perf_counter()
        elif ord("1") <= key <= ord("9"):
            self.drawing.set_color(key - ord("1"))
        return True


def run() -> int:
    camera = Camera()
    tracker = None
    mouse = None
    try:
        camera.open()
        tracker = HandTracker()
        import mouse_controller                  # imported late: PyAutoGUI needs a display
        mouse = mouse_controller.MouseController()
        app = VirtualMouseApp(tracker, mouse)

        cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        if config.WINDOW_TOPMOST:
            try:
                cv2.setWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
            except cv2.error:
                pass
        print("AI Virtual Mouse running. Press Q in the camera window to quit.")

        while True:
            frame = camera.read()
            if frame is None:
                cv2.waitKey(1)
                continue
            display = app.process_frame(frame, time.perf_counter())
            if config.DISPLAY_WIDTH and display.shape[1] != config.DISPLAY_WIDTH:
                scale = config.DISPLAY_WIDTH / display.shape[1]
                display = cv2.resize(display, (config.DISPLAY_WIDTH, int(display.shape[0] * scale)),
                                     interpolation=cv2.INTER_LINEAR)
            cv2.imshow(config.WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if not app.handle_key(key):
                break
            if cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break                            # window closed with the X button
        return 0

    except CameraError as exc:
        print(f"\n{exc}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 0
    except Exception as exc:                     # incl. pyautogui.FailSafeException
        print(f"\nStopped: {type(exc).__name__}: {exc}")
        return 1
    finally:                                     # ALWAYS: no stuck mouse button, free resources
        if mouse is not None:
            mouse.release_all()
        if tracker is not None:
            tracker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(run())
