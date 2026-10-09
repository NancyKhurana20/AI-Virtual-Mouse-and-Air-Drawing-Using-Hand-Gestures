"""
config.py - every tunable value of the AI Virtual Mouse lives here.

Distances for pinches are expressed as a RATIO of the palm size
(wrist -> middle-finger knuckle), so they work at any distance from the
camera.  Times are in seconds.
"""
import sys

# ----------------------------------------------------------------- camera
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FOURCC = None            # e.g. "MJPG" can give more FPS on some webcams
CAMERA_MAX_READ_FAILURES = 30   # consecutive failed reads before giving up
MIRROR_FRAME = True             # selfie view: moving right moves cursor right

# size of the window you SEE (the camera itself still captures 640x480, so speed and
# tracking are unchanged). Height follows automatically. Try 800, 960, 1280. None = original size.
DISPLAY_WIDTH = 1000

# -------------------------------------------------------------- mediapipe
MAX_NUM_HANDS = 1
MODEL_COMPLEXITY = 0            # 0 = fastest, 1 = more accurate
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6

# ------------------------------------------------------- cursor movement
ACTIVE_MARGIN = 0.15            # fraction of the frame ignored on every side
SMOOTHING_FACTOR = 0.40         # EMA alpha when the finger moves fast (0..1)
SMOOTHING_FACTOR_SLOW = 0.12    # EMA alpha for tiny movements (kills jitter)
SMOOTHING_SLOW_DISTANCE = 10    # px (at 1920 wide) below which SLOW alpha is used
SMOOTHING_FAST_DISTANCE = 100   # px (at 1920 wide) above which full alpha is used
CURSOR_SCREEN_EDGE = 2          # keep cursor this far from screen edges (PyAutoGUI fail-safe)

# ----------------------------------------------------------- left / right click
# Both clicks fire the INSTANT a pinch is confirmed (no waiting for release, no
# double-click delay, no drag timer), then the pinch must be released to click again.
PINCH_THRESHOLD = 0.30          # thumb-index (or thumb-middle) distance / palm size -> pinch starts
PINCH_RELEASE_THRESHOLD = 0.45  # ... must exceed this to release (hysteresis)
PINCH_CONFIRM_FRAMES = 2        # consecutive frames needed to confirm a LEFT pinch
PINCH_RELEASE_FRAMES = 2        # consecutive frames needed to confirm a release
RIGHT_CLICK_CONFIRM_FRAMES = 3  # consecutive frames needed to confirm a RIGHT pinch
MIDDLE_PINCH_MARGIN = 0.10      # thumb-middle must beat thumb-index by this much (right click)
CLICK_COOLDOWN = 0.12           # min seconds between two LEFT clicks
RIGHT_CLICK_COOLDOWN = 0.40     # min seconds between two RIGHT clicks

# click accuracy helpers (the fingertip moves while the fingers close, so we click
# where the cursor was just BEFORE the pinch started closing)
CLICK_APPROACH_RATIO = 1.00     # fingers count as "still open" above this distance ratio (click aims at the cursor spot from that moment)
CLICK_ANCHOR_MIN_LOOKBACK = 0.06  # s: always look back at least this far
CLICK_ANCHOR_MAX_LOOKBACK = 0.25  # s: ...and never further than this
PINCH_STICKY_RADIUS = 25        # px (at 1920 wide): cursor holds still inside this radius while pinching
DOUBLE_CLICK_SNAP_TIME = 0.60   # s: a 2nd click this soon snaps to the 1st click's spot (so Windows sees a double click)

# ------------------------------------------------------ finger-state logic
FINGER_EXTENDED_ANGLE = 140     # deg at the PIP joint; larger = straight finger
FIST_REACH_RATIO = 0.95         # index tip closer than this (x palm) to wrist + others folded = fist

# --------------------------------------------------------------- hand loss
HAND_LOST_GRACE_TIME = 0.30     # tolerate tracking drop-outs this long before releasing everything
MIN_PALM_SIZE_PX = 20           # ignore hands smaller than this (tracking garbage)

# ----------------------------------------------------------------- drawing
# Pen is DOWN while the INDEX finger is up and the MIDDLE finger is folded (thumb is ignored).
DRAW_COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 255)]  # BGR, keys 1-5
DRAW_COLOR_INDEX = 0
DRAW_THICKNESS = 6
PEN_CONFIRM_FRAMES = 2          # frames of "index up, middle folded" before the pen goes down
PEN_RELEASE_FRAMES = 3          # frames of any other pose before the pen lifts (ignores flicker)
DRAW_SMOOTH_SLOW = 0.15         # EMA alpha for tiny movements (kills hand tremor)
DRAW_SMOOTH_FAST = 0.65         # EMA alpha for fast movements (low lag)
DRAW_SMOOTH_SLOW_DIST = 3       # px: below this the slow alpha is used
DRAW_SMOOTH_FAST_DIST = 40      # px: above this the fast alpha is used
DRAW_MIN_MOVE_PX = 1.0          # ignore pen movement smaller than this
DRAW_MAX_JUMP_PX = 120          # bigger jump between frames = tracking glitch -> new stroke
DRAW_CURVE_STEPS = 6            # segments per smoothed curve piece (higher = rounder)
DRAW_OVERLAY_ALPHA = 0.90       # canvas opacity in cv2.addWeighted

# ---------------------------------------------------------------------- UI
WINDOW_NAME = "AI Virtual Mouse"
WINDOW_TOPMOST = False
INSTRUCTION_SECONDS = 8         # startup help overlay duration (H toggles it)
ACTION_LABEL_SECONDS = 0.6      # how long "LEFT CLICK" etc. stay on screen
FPS_AVERAGE_FRAMES = 20
IS_WINDOWS = sys.platform.startswith("win")
