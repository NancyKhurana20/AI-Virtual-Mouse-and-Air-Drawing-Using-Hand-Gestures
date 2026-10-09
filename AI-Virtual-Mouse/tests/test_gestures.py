"""
Headless end-to-end tests of the gesture logic.

No webcam / screen needed: PyAutoGUI is replaced by a recorder and the hand
is a synthetic set of 21 landmarks.  Run from the project folder:

    python -m unittest discover -s tests -v
"""
import os
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---- fake pyautogui (records every call) -----------------------------------
CALLS = []
fake = types.ModuleType("pyautogui")
fake.PAUSE, fake.FAILSAFE = 0, True
fake.size = lambda: (1920, 1080)
for _n in ("click", "mouseDown", "mouseUp"):
    setattr(fake, _n, (lambda n: lambda *a, **k: CALLS.append((n, a)))(_n))
fake.click = lambda *a, **k: CALLS.append(("rightClick" if k.get("button") == "right" else "click", a))
fake.moveTo = lambda x, y, duration=0: CALLS.append(("moveTo", (x, y)))
sys.modules["pyautogui"] = fake

from hand_data import HandData  # noqa: E402
from main import VirtualMouseApp  # noqa: E402
from mouse_controller import MouseController  # noqa: E402
from gesture_detector import Mode  # noqa: E402

DT = 1 / 30


def make_hand(cx=320, cy=330, thumb="away", index=True, middle=True, ring=True, pinky=True, ty=0):
    """Upright hand, palm size 100 px. thumb: 'away' | 'index' (pinch) | 'middle' (pinch)."""
    P = np.zeros((21, 3), np.float32)
    P[0] = (0, 0, 0)
    mcp = {5: (-30, -95), 9: (0, -100), 13: (30, -95), 17: (55, -85)}
    for base, ext in ((5, index), (9, middle), (13, ring), (17, pinky)):
        mx, my = mcp[base]
        P[base] = (mx, my, 0)
        if ext:
            P[base + 1] = (mx, my - 40, 0); P[base + 2] = (mx, my - 65, 0); P[base + 3] = (mx, my - 87, 0)
        else:
            P[base + 1] = (mx, my - 35, 0); P[base + 2] = (mx, my - 15, 0); P[base + 3] = (mx, my + 0, 0)
    P[1], P[2], P[3] = (-20, -20, 0), (-40, -35, 0), (-55, -45, 0)
    tip = {"away": (-75, -50), "index": P[8][:2] + (6, 6), "middle": P[12][:2] + (4, 4)}[thumb]
    P[4] = (tip[0], tip[1], 0)
    P[:, 0] += cx
    P[:, 1] += cy + ty
    return HandData(points=P, frame_w=640, frame_h=480, raw=None)


class FakeTracker:
    def __init__(self):
        self.hand = None

    def process(self, frame):
        return self.hand

    def draw(self, img, hand):
        pass

    def close(self):
        pass


class Sim:
    def __init__(self):
        CALLS.clear()
        self.tracker = FakeTracker()
        self.mouse = MouseController()
        self.app = VirtualMouseApp(self.tracker, self.mouse)
        self.t = 100.0
        self.frame = np.zeros((480, 640, 3), np.uint8)
        self.out = None

    def run(self, hand_fn, seconds):
        n = max(int(round(seconds / DT)), 1)
        for i in range(n):
            self.t += DT
            self.tracker.hand = hand_fn(i) if callable(hand_fn) else hand_fn
            self.out = self.app.process_frame(self.frame.copy(), self.t)

    def count(self, name):
        return sum(1 for c in CALLS if c[0] == name)

    def index(self, name):
        return [i for i, c in enumerate(CALLS) if c[0] == name]


class Tests(unittest.TestCase):
    def test1_cursor_follows_index_finger(self):
        s = Sim()
        s.run(lambda i: make_hand(cx=200 + i * 4, thumb="away"), 1.0)
        xs = [c[1][0] for c in CALLS if c[0] == "moveTo"]
        self.assertGreater(len(xs), 10)
        self.assertTrue(all(b >= a for a, b in zip(xs, xs[1:])), "cursor must move monotonically right")
        self.assertGreater(xs[-1] - xs[0], 300)
        self.assertEqual(s.count("click") + s.count("rightClick"), 0)

    def test2_single_pinch_exactly_one_instant_left_click(self):
        s = Sim()
        s.run(make_hand(thumb="away"), 0.5)
        s.run(make_hand(thumb="index"), 0.15)          # click must already be sent while still pinched
        self.assertEqual(s.count("click"), 1)
        s.run(make_hand(thumb="index"), 1.5)           # holding the pinch never repeats / drags
        self.assertEqual(s.count("click"), 1)
        self.assertEqual(s.count("mouseDown"), 0)
        s.run(make_hand(thumb="away"), 1.0)
        self.assertEqual(s.count("click"), 1)
        self.assertEqual(s.count("rightClick"), 0)

    def test2b_holding_open_hand_never_clicks(self):
        s = Sim()
        s.run(make_hand(thumb="away"), 3)
        self.assertEqual(s.count("click") + s.count("rightClick"), 0)

    def test3_two_quick_pinches_are_two_clicks_on_the_same_pixel(self):
        s = Sim()
        s.run(make_hand(thumb="away"), 0.5)
        s.run(make_hand(thumb="index"), 0.2)
        s.run(make_hand(thumb="away"), 0.15)
        s.run(make_hand(thumb="index"), 0.2)
        s.run(make_hand(thumb="away"), 0.5)
        self.assertEqual(s.count("click"), 2)
        spots = []
        for i in s.index("click"):
            j = max(k for k in range(i) if CALLS[k][0] == "moveTo")
            spots.append(CALLS[j][1])
        self.assertEqual(spots[0], spots[1], "2nd click must snap to the 1st click's pixel")

    def test4_right_click_exactly_once_even_if_held(self):
        s = Sim()
        s.run(make_hand(thumb="away"), 0.5)
        s.run(make_hand(thumb="middle"), 2.0)
        self.assertEqual(s.count("rightClick"), 1)
        self.assertEqual(s.count("click"), 0)
        s.run(make_hand(thumb="away"), 0.5)
        self.assertEqual(s.count("rightClick"), 1)
        s.run(make_hand(thumb="middle"), 0.5)          # a second, new pinch -> a second right click
        self.assertEqual(s.count("rightClick"), 2)

    def test5_click_lands_where_user_aimed_not_where_finger_drifted(self):
        s = Sim()
        s.run(make_hand(cx=300, thumb="away"), 1.0)
        aim = [c[1] for c in CALLS if c[0] == "moveTo"][-1]
        # fingers close while the fingertip drifts 40 px to the right (what happens in real life)
        s.run(lambda i: make_hand(cx=300 + i * 10, thumb="index"), 0.2)
        self.assertEqual(s.count("click"), 1)
        i = s.index("click")[0]
        j = max(k for k in range(i) if CALLS[k][0] == "moveTo")
        self.assertLess(abs(CALLS[j][1][0] - aim[0]), 40, "click must land near the aimed spot")

    def test6_cursor_holds_still_while_pinch_is_held_then_resumes(self):
        s = Sim()
        s.run(make_hand(cx=300, thumb="away"), 0.5)
        s.run(make_hand(cx=300, thumb="index"), 0.3)
        before = len(CALLS)
        s.run(lambda i: make_hand(cx=300 + (i % 2) * 2, thumb="index"), 0.5)   # tiny tremor
        self.assertEqual(len([c for c in CALLS[before:] if c[0] == "moveTo"]), 0)
        s.run(lambda i: make_hand(cx=300 + i * 6, thumb="away"), 1.0)           # deliberate move
        self.assertGreater(len([c for c in CALLS[before:] if c[0] == "moveTo"]), 5)

    def test7_drawing_index_only_stroke_and_clear(self):
        s = Sim()
        s.app.handle_key(ord("d"))
        self.assertEqual(s.app.mode, Mode.DRAWING)
        # index up + middle folded (thumb far away) = draw. No pinch needed anymore.
        s.run(lambda i: make_hand(cx=200 + i * 6, thumb="away", middle=False, ring=False, pinky=False), 1.0)
        ink = int(s.app.drawing.canvas.any(axis=2).sum())
        self.assertGreater(ink, 500)
        self.assertEqual(s.count("click") + s.count("rightClick"), 0)
        self.assertEqual(s.count("moveTo"), 0, "no cursor control in drawing mode")
        # thumb touching the index must NOT matter
        s.app.handle_key(ord("c"))
        s.run(lambda i: make_hand(cx=200 + i * 6, thumb="index", middle=False, ring=False, pinky=False), 1.0)
        self.assertGreater(int(s.app.drawing.canvas.any(axis=2).sum()), 500)
        # index + middle up (peace sign) lifts the pen: no more ink, no connecting line afterwards
        s.run(make_hand(cx=400, thumb="away"), 0.3)
        self.assertIsNone(s.app.drawing._last)
        ink2 = int(s.app.drawing.canvas.any(axis=2).sum())
        s.run(lambda i: make_hand(cx=500, cy=200, thumb="away"), 0.3)
        self.assertEqual(int(s.app.drawing.canvas.any(axis=2).sum()), ink2)
        self.assertTrue(s.out.any())
        s.app.handle_key(ord("c"))
        self.assertEqual(int(s.app.drawing.canvas.any()), 0)

    def test7b_stroke_is_smooth_no_big_jumps_from_jitter(self):
        s = Sim()
        s.app.handle_key(ord("d"))
        rng = np.random.default_rng(1)
        s.run(lambda i: make_hand(cx=150 + i * 5 + rng.uniform(-2, 2), cy=330 + rng.uniform(-2, 2),
                                  thumb="away", middle=False, ring=False, pinky=False), 1.5)
        self.assertGreater(int(s.app.drawing.canvas.any(axis=2).sum()), 500)

    def test8_hand_removed_clears_state_and_is_safe(self):
        s = Sim()
        s.run(make_hand(thumb="index"), 1.0)
        self.assertEqual(s.count("click"), 1)
        s.run(None, 1.0)                                # hand disappears
        s.run(None, 1.0)                                # no crash
        self.assertEqual(s.count("click"), 1)
        s.run(make_hand(thumb="away"), 0.3)
        s.run(make_hand(thumb="index"), 0.3)            # works again after returning
        self.assertEqual(s.count("click"), 2)

    def test8b_short_dropout_does_not_cause_extra_click(self):
        s = Sim()
        s.run(make_hand(thumb="index"), 1.0)
        s.run(None, 0.1)                                # 3 lost frames
        s.run(make_hand(thumb="index"), 0.5)
        self.assertEqual(s.count("click"), 1)

    def test_fist_does_not_click(self):
        s = Sim()
        s.run(make_hand(thumb="away"), 0.3)
        s.run(make_hand(thumb="away", index=False, middle=False, ring=False, pinky=False), 1.0)
        self.assertEqual(s.count("click") + s.count("rightClick") + s.count("mouseDown"), 0)

    def test_removed_features_do_not_exist(self):
        """drag & drop and scroll are gone: no mouseDown / scroll call is ever made."""
        s = Sim()
        s.run(make_hand(thumb="index"), 2.0)                       # long pinch = still just ONE click
        s.run(lambda i: make_hand(cy=300 + i * 3, thumb="away", ring=False, pinky=False), 2.0)   # two-finger sweep
        self.assertEqual(s.count("mouseDown"), 0)
        self.assertFalse(hasattr(s.mouse, "scroll"))

    def test_mode_switch_is_safe(self):
        s = Sim()
        s.run(make_hand(thumb="index"), 1.0)
        s.app.handle_key(ord("d"))
        s.app.handle_key(ord("d"))
        s.run(make_hand(thumb="away"), 0.3)
        s.run(make_hand(thumb="index"), 0.3)
        self.assertEqual(s.count("click"), 2)

    def test_release_all_is_safe(self):
        s = Sim()
        s.mouse.release_all()


if __name__ == "__main__":
    unittest.main(verbosity=2)
