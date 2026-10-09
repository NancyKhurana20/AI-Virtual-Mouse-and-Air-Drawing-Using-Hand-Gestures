"""utils.py - small reusable helpers: FPS counter and exponential smoothing."""
import math
import time
from collections import deque
from typing import Optional, Tuple


class FPSCounter:
    """Real FPS = 1 / average time between the last N frames."""

    def __init__(self, window: int = 20):
        self._times = deque(maxlen=window)

    def tick(self) -> float:
        self._times.append(time.perf_counter())
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0


class EMA2D:
    """Exponential moving average of an (x, y) point: s = s + alpha * (raw - s)."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self._p: Optional[Tuple[float, float]] = None

    @property
    def value(self):
        return self._p

    def reset(self):
        self._p = None

    def set(self, x: float, y: float):
        self._p = (x, y)

    def update(self, x: float, y: float) -> Tuple[float, float]:
        if self._p is None:
            self._p = (x, y)
        else:
            px, py = self._p
            self._p = (px + self.alpha * (x - px), py + self.alpha * (y - py))
        return self._p


class AdaptiveEMA2D(EMA2D):
    """
    EMA whose alpha depends on how far the raw point is from the smoothed one:
    near  -> small alpha (suppress jitter), far -> large alpha (low lag).
    """

    def __init__(self, alpha_slow, alpha_fast, d_slow, d_fast):
        super().__init__(alpha_fast)
        self.alpha_slow, self.alpha_fast = alpha_slow, alpha_fast
        self.d_slow, self.d_fast = d_slow, d_fast

    def update(self, x, y):
        if self._p is None:
            self._p = (x, y)
            return self._p
        d = math.hypot(x - self._p[0], y - self._p[1])
        t = (d - self.d_slow) / max(self.d_fast - self.d_slow, 1e-6)
        t = min(max(t, 0.0), 1.0)
        self.alpha = self.alpha_slow + (self.alpha_fast - self.alpha_slow) * t
        return super().update(x, y)
