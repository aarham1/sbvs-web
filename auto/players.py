"""Follow players from frame to frame and keep clean jersey-color samples for each one.

Why: at the moment of a shot the shooter is often tangled with a defender, so one box
holds two jerseys and the team guess becomes a coin flip. Earlier in the play the same
player was usually standing alone. We follow every player with a simple tracker and
sample his chest colors only while no other player box touches his, then decide the
shooter's team from those clean samples.

Tracker: greedy matching of each new box to the existing track with the highest overlap
(IoU), falling back to the nearest center within MAX_JUMP box heights (players move a lot
between checked frames). This is the matching step of SORT/ByteTrack without the Kalman
filter (see docs/SOURCES.md, "Player tracking"). Tracks are cleared at camera cuts.
"""

import math
from collections import deque

import numpy as np

from .teams import torso_pixels

COLOR_WEIGHT = 0.6        # how much jersey color counts when matching boxes to tracks
COLOR_SCALE = 60.0        # Lab distance that counts as "completely different"
COLOR_SMOOTH = 0.3
HEAVY_OVERLAP = 0.3       # IoU between two current boxes above which color is not trusted

MIN_IOU = 0.25
MAX_JUMP = 0.6            # center distance, in box heights, allowed when IoU is too low
MAX_AGE_S = 1.0           # forget a track not seen for this long
SAMPLE_EVERY_S = 0.4      # at most one color sample per track per this many seconds
MAX_SAMPLES = 12          # newest samples kept per track
ISOLATION_GAP = 0.1       # boxes closer than this (fraction of width) count as touching
PIXELS_PER_SAMPLE = 250


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def isolated(box, others):
    """True when no other box touches this one (with a small gap)."""
    w = box[2] - box[0]
    g = ISOLATION_GAP * w
    for o in others:
        if o is box:
            continue
        if o[0] < box[2] + g and o[2] > box[0] - g and o[1] < box[3] and o[3] > box[1]:
            return False
    return True


class PlayerTracker:
    def __init__(self):
        self.next_id = 1
        self.reset()

    def reset(self):
        self.tracks = {}      # id -> {"box", "t", "samples": deque, "last_sample_t"}

    def update(self, t, boxes, frame=None, referee_boxes=()):
        """Match boxes to tracks; return a list of track ids (same order as boxes)."""
        for tid in [k for k, v in self.tracks.items() if t - v["t"] > MAX_AGE_S]:
            del self.tracks[tid]
        colors = [self._color(frame, b) for b in boxes] if frame is not None else [None] * len(boxes)
        # a box that heavily overlaps another one shows a mix of both jerseys: no color for it
        for i, b in enumerate(boxes):
            if any(j != i and iou(b, o) > HEAVY_OVERLAP for j, o in enumerate(boxes)):
                colors[i] = None
        pairs = []
        for i, b in enumerate(boxes):
            for tid, tr in self.tracks.items():
                pred = self._predict(tr, t)          # constant-velocity guess (as in SORT)
                o = iou(b, pred)
                h = max(b[3] - b[1], 1.0)
                d = math.dist(_center(b), _center(pred)) / h
                if o >= MIN_IOU or d <= MAX_JUMP:
                    score = o - 0.1 * d
                    if colors[i] is not None and tr.get("color") is not None:
                        # appearance term (as in DeepSORT, with jersey color instead of a CNN):
                        # stops two colliding players from swapping identities
                        score -= COLOR_WEIGHT * min(np.linalg.norm(colors[i] - tr["color"]) / COLOR_SCALE, 1.0)
                    pairs.append((score, i, tid))
        pairs.sort(reverse=True)
        ids = [None] * len(boxes)
        used = set()
        for _, i, tid in pairs:
            if ids[i] is None and tid not in used:
                ids[i] = tid
                used.add(tid)
        for i, b in enumerate(boxes):
            if ids[i] is None:
                ids[i] = self.next_id
                self.tracks[self.next_id] = {"box": b, "t": t, "samples": deque(maxlen=MAX_SAMPLES),
                                             "last_sample_t": -1e9}
                self.next_id += 1
            tr = self.tracks[ids[i]]
            dt = t - tr["t"]
            if dt > 0 and tr["t"] != t:
                v = [(n - o) / dt for n, o in zip(_center(b), _center(tr["box"]))]
                old_v = tr.get("v", v)
                tr["v"] = [0.5 * a + 0.5 * c for a, c in zip(v, old_v)]
            tr["box"], tr["t"] = b, t
            alone = isolated(b, list(boxes) + list(referee_boxes))
            if colors[i] is not None and alone:      # only learn a track's color while it is alone
                tr["color"] = colors[i] if tr.get("color") is None else \
                    (1 - COLOR_SMOOTH) * tr["color"] + COLOR_SMOOTH * colors[i]
            if frame is not None and t - tr["last_sample_t"] >= SAMPLE_EVERY_S and alone:
                px = torso_pixels(frame, b)
                if px is not None and len(px) >= 20:
                    keep = np.random.default_rng(ids[i] * 7919 + len(tr["samples"])).choice(
                        len(px), min(len(px), PIXELS_PER_SAMPLE), replace=False)
                    tr["samples"].append(px[keep])
                    tr["last_sample_t"] = t
        return ids

    @staticmethod
    def _predict(tr, t):
        vx, vy = tr.get("v", (0.0, 0.0))
        dt = min(t - tr["t"], 0.5)
        x1, y1, x2, y2 = tr["box"][:4]
        return (x1 + vx * dt, y1 + vy * dt, x2 + vx * dt, y2 + vy * dt)

    @staticmethod
    def _color(frame, box):
        px = torso_pixels(frame, box)
        return np.median(px, 0) if px is not None and len(px) else None

    def samples(self, tid):
        tr = self.tracks.get(tid)
        return list(tr["samples"]) if tr else []
