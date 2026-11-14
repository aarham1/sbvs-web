"""Shot attempts, makes and misses from ball and hoop positions.

Idea (same as SBVS 2.0's fixed-camera rules, but with a detected hoop instead of a
hand-marked one, so it also works when the camera pans):
  * ATTEMPT starts when the ball reaches the zone above the rim after coming from away
    from the hoop (or while rising). Rebounds bouncing on the rim do not start a new
    attempt because of a short cooldown and the "came from away" rule.
  * MADE when the ball crosses the rim line going down and the crossing point (linear
    interpolation between the last point above and the first point below) is inside
    the rim.
  * MISSED when it comes down outside the rim, flies away from the hoop, or nothing is
    decided within MAX_ATTEMPT_S.
  * UNCLEAR when the ball disappears near the rim (often hidden by the net/backboard).
All distances are in hoop widths so the rules work at any zoom level.
"""

import math
from collections import deque
from dataclasses import dataclass, field

RIM_FRACTION = 0.2        # rim line = hoop box top + this * hoop box height
ZONE_HALF_WIDTH = 2.5     # attempt zone reaches this many hoop widths left/right of the rim (threes peak far out)
ZONE_HEIGHT = 4.0         # ... and this many hoop widths above the rim line
AWAY_DIST = 2.5           # "came from away": ball was this many hoop widths from the hoop
LOOKBACK_S = 2.5          # ... within this many seconds before the attempt
RISE_MIN = 0.4            # or rose at least this many hoop widths in RISE_WINDOW_S
RISE_WINDOW_S = 0.6
BELOW_RIM_START = 0.6     # an attempt may also start this many hoop widths below the rim line
                          # (layups barely rise above the rim in side-on TV views)
RIM_MARGIN = 0.15         # extra rim width (fraction of hoop width) counted as "inside"
FAR_SIDE = 3.5            # during an attempt: ball this many hoop widths beside the rim,
FAR_BELOW = 2.5           # ... this far below the rim line, or above the zone -> missed
MAX_ATTEMPT_S = 3.0
LOST_S = 1.2              # no ball for this long during an attempt -> unclear (ball often hidden near the rim)
COOLDOWN_S = 1.2          # no new attempt this soon after the previous one ended
HOOP_STALE_S = 1.0        # forget the hoop if not seen for this long
HOOP_SMOOTH = 0.8         # moving-average weight of a new hoop detection (high: cameras pan)


@dataclass
class ShotEvent:
    t_attempt: float
    t_result: float
    result: str                     # made / missed / unclear
    hoop: tuple                     # hoop box at the attempt
    path: list = field(default_factory=list)   # (t, x, y) ball points of the attempt
    reason: str = ""


def hoop_geometry(hoop):
    x1, y1, x2, y2 = hoop[:4]
    w = max(x2 - x1, 1.0)
    rim_y = y1 + RIM_FRACTION * (y2 - y1)
    return x1, x2, w, rim_y, ((x1 + x2) / 2, rim_y)


class ShotTracker:
    def __init__(self):
        self.reset()
        self.last_end = -1e9

    def reset(self):
        """Call at a camera cut: positions before and after a cut are not comparable."""
        self.hoop = None
        self.hoop_t = -1e9
        self.history = deque()      # (t, x, y, hoop_w, hoop_center)
        self.attempt = None         # dict while an attempt is open

    # ---- hoop -------------------------------------------------------------------
    def _update_hoop(self, t, hoops, ball):
        if hoops:
            if ball is not None and len(hoops) > 1:
                best = min(hoops, key=lambda h: math.dist(((h[0] + h[2]) / 2, (h[1] + h[3]) / 2), ball[:2]))
            else:
                best = hoops[0]
            best = tuple(best[:4])
            if self.hoop is not None and t - self.hoop_t < HOOP_STALE_S and _overlap(self.hoop, best) > 0.2:
                a = HOOP_SMOOTH
                best = tuple(a * n + (1 - a) * o for n, o in zip(best, self.hoop))
            self.hoop, self.hoop_t = best, t
        elif self.hoop is not None and t - self.hoop_t > HOOP_STALE_S:
            self.hoop = None

    # ---- main -------------------------------------------------------------------
    def update(self, t, ball, hoops):
        """ball: (x, y) or None. hoops: list of boxes (x1, y1, x2, y2, ...).
        Returns a finished ShotEvent or None."""
        self._update_hoop(t, hoops, ball)
        if self.attempt is not None and self.hoop is None:
            self.hoop = self.attempt["hoop"]          # keep the attempt's hoop while it runs
        if self.hoop is None:
            if ball is not None:
                self.history.append((t, ball[0], ball[1], None, None))
            self._trim(t)
            return None
        x1, x2, w, rim_y, center = hoop_geometry(self.hoop)
        if ball is not None:
            self.history.append((t, ball[0], ball[1], w, center))
        self._trim(t)

        if self.attempt is None:
            if ball is not None and t - self.last_end >= COOLDOWN_S and self._in_zone(ball, x1, x2, w, rim_y) \
                    and self._came_in(t, ball, w, center):
                self.attempt = {"t0": t, "hoop": self.hoop, "path": [(t, ball[0], ball[1])],
                                "rel": [((ball[0] - center[0]) / w, (ball[1] - rim_y) / w)],
                                "last_seen": t}
            return None
        return self._update_attempt(t, ball)

    def _update_attempt(self, t, ball):
        """All tests use the ball's position RELATIVE to the hoop in the same frame
        (in hoop widths, origin = rim center), because TV cameras pan and zoom during
        shots: the hoop can move 100+ pixels on screen while the ball is in the air."""
        a = self.attempt
        a["hoop"] = self.hoop                      # follow the hoop as the camera moves
        x1, x2, w, rim_y, center = hoop_geometry(self.hoop)
        if ball is None:
            if t - a["last_seen"] > LOST_S:
                rx, ry = a["rel"][-1]
                near = (abs(rx) <= 0.5 + ZONE_HALF_WIDTH and -ZONE_HEIGHT <= ry < BELOW_RIM_START) \
                    or math.hypot(rx, ry) < 1.5
                return self._finish(t, "unclear" if near else "missed", "ball lost")
            if t - a["t0"] > MAX_ATTEMPT_S:
                return self._finish(t, "missed", "timeout")
            return None
        rx, ry = (ball[0] - center[0]) / w, (ball[1] - rim_y) / w
        prx, pry = a["rel"][-1]
        a["path"].append((t, ball[0], ball[1]))
        a["rel"].append((rx, ry))
        a["last_seen"] = t
        net_depth = (self.hoop[3] - rim_y) / w + 0.3
        if abs(rx) <= 0.5 and 0 <= ry <= net_depth and ry > pry:
            return self._finish(t, "made", "dropped through the net")
        if pry < 0 <= ry:                                  # crossed the rim line going down
            f = -pry / max(ry - pry, 1e-6)
            cross = prx + f * (rx - prx)
            if abs(cross) <= 0.5 + RIM_MARGIN:
                return self._finish(t, "made", "through rim")
            return self._finish(t, "missed", "came down outside rim")
        if self._left_area(ball[0], ball[1], x1, x2, w, rim_y):
            return self._finish(t, "missed", "ball left the hoop area")
        if t - a["t0"] > MAX_ATTEMPT_S:
            return self._finish(t, "missed", "timeout")
        return None

    def _finish(self, t, result, reason):
        a = self.attempt
        self.attempt = None
        self.last_end = t
        return ShotEvent(a["t0"], t, result, a["hoop"], a["path"], reason)

    def flush(self, t):
        """End of video or camera cut: close an open attempt."""
        if self.attempt is None:
            return None
        return self._finish(t, "unclear", "video cut during shot")

    # ---- helpers ----------------------------------------------------------------
    def _trim(self, t):
        while self.history and t - self.history[0][0] > LOOKBACK_S:
            self.history.popleft()

    @staticmethod
    def _in_zone(ball, x1, x2, w, rim_y):
        bx, by = ball[0], ball[1]
        return (x1 - ZONE_HALF_WIDTH * w <= bx <= x2 + ZONE_HALF_WIDTH * w
                and rim_y - ZONE_HEIGHT * w <= by < rim_y + BELOW_RIM_START * w)

    @staticmethod
    def _left_area(bx, by, x1, x2, w, rim_y):
        return (bx < x1 - FAR_SIDE * w or bx > x2 + FAR_SIDE * w
                or by > rim_y + FAR_BELOW * w or by < rim_y - (ZONE_HEIGHT + 1.5) * w)

    def _came_in(self, t, ball, w, center):
        # only points after the previous attempt ended (its own path must not count)
        for (ht, hx, hy, hw, hc) in self.history:
            if ht > self.last_end and hw is not None and math.dist((hx, hy), hc) > AWAY_DIST * hw:
                return True
        recent = [p for p in self.history if t - p[0] <= RISE_WINDOW_S and p[0] > self.last_end]
        if recent:
            lowest = max(p[2] for p in recent)
            if lowest - ball[1] >= RISE_MIN * w:
                return True
        return False


def _overlap(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / area if area > 0 else 0.0
