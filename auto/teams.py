"""Split players into two teams by jersey color, and give each team a color name.

For every player box we take the torso (the middle of the upper half, where the jersey
is), convert to Lab color space and keep the median color. K-means with k = 2 over all
sampled players gives the two team colors (referees are a separate detector class and
are left out). Lab distances match human color differences better than RGB.
Source: k-means jersey clustering is the standard unsupervised team-ID baseline
(see docs/SOURCES.md, "Team split").
"""

import cv2
import numpy as np

# Reference colors in BGR for naming a team.
COLOR_NAMES = {
    "White": (240, 240, 240), "Black": (25, 25, 25), "Gray": (128, 128, 128),
    "Red": (40, 40, 200), "Maroon": (40, 30, 120), "Orange": (30, 120, 240),
    "Gold": (40, 180, 220), "Yellow": (40, 220, 240), "Green": (60, 150, 50),
    "Dark green": (40, 80, 20), "Light blue": (230, 170, 100), "Blue": (190, 90, 30),
    "Navy": (90, 40, 20), "Purple": (130, 40, 100), "Pink": (180, 120, 240),
}


def _lab(bgr):
    return cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)


_NAME_LAB = {name: _lab(c) for name, c in COLOR_NAMES.items()}


TORSO_TOP, TORSO_BOTTOM = 0.28, 0.58   # chest area (box top can be raised hands)


def torso_crop(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    w, h = x2 - x1, y2 - y1
    if w < 6 or h < 12:
        return None
    crop = frame[y1 + int(TORSO_TOP * h): y1 + int(TORSO_BOTTOM * h), x1 + int(0.3 * w): x2 - int(0.3 * w)]
    return crop if crop.size else None


def jersey_color(frame, box):
    """Median Lab color of the torso, or None if the box is too small."""
    crop = torso_crop(frame, box)
    if crop is None:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    return np.median(lab, axis=0)


def torso_pixels(frame, box):
    crop = torso_crop(frame, box)
    if crop is None:
        return None
    return cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)


def chest_under_ball_pixels(frame, box, ball):
    """Shooter's chest: a strip just below the ball, inside the player's box. When two
    players overlap in one box, the one holding the ball is the one under it."""
    if ball is None:
        return torso_pixels(frame, box)
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    w, h = x2 - x1, y2 - y1
    bx, by = ball
    if not (x1 <= bx <= x2):
        return torso_pixels(frame, box)
    top = int(max(y1, by) + 0.12 * h)
    bottom = int(min(y2, top + 0.3 * h))
    left, right = int(max(x1, bx - 0.12 * w)), int(min(x2, bx + 0.12 * w))
    crop = frame[top:bottom, left:right]
    if crop.size < 30:
        return torso_pixels(frame, box)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)


def color_name(lab):
    """Plain color word for a Lab color. Nearly colorless jerseys are named by brightness
    (arena lighting and video compression tint whites slightly blue or yellow)."""
    chroma = float(np.hypot(lab[1] - 128, lab[2] - 128))
    if chroma < 15:
        return "White" if lab[0] > 165 else ("Gray" if lab[0] > 85 else "Black")
    return min(_NAME_LAB, key=lambda n: np.linalg.norm(_NAME_LAB[n] - lab))


BINS = 8   # Lab histogram bins per channel


def _quantize(px):
    q = np.clip((px / 256.0 * BINS).astype(int), 0, BINS - 1)
    return q[:, 0] * BINS * BINS + q[:, 1] * BINS + q[:, 2]


class TeamModel:
    """Two teams from jersey colors.

    1. k-means (k = 2) on each sampled player's median chest color gives the team split.
    2. For each team, a Lab color histogram of all its chest pixels is built. A pixel's
       vote is log(P(color | team 0) / P(color | team 1)), so colors that appear for both
       teams (crowd, court, skin) cancel out and only real jersey colors decide.
    """

    def __init__(self, samples, pixel_samples=None, seed=0):
        data = np.float32(samples)
        if len(data) < 4:
            raise ValueError("not enough player samples to split teams")
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        cv2.setRNGSeed(seed)
        _, labels, centers = cv2.kmeans(data, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        labels = labels.flatten()
        counts = np.bincount(labels, minlength=2)
        order = np.argsort(-counts)          # team 0 = the color seen most often
        remap = {int(order[0]): 0, int(order[1]): 1}
        labels = np.array([remap[int(l)] for l in labels])
        self.centers = centers[order].astype(float)
        self.hist = None
        if pixel_samples is not None and len(pixel_samples) == len(data):
            hists = np.ones((2, BINS ** 3))          # Laplace smoothing
            for lab_px, team in zip(pixel_samples, labels):
                if lab_px is not None and len(lab_px):
                    hists[team] += np.bincount(_quantize(lab_px), minlength=BINS ** 3)
            self.hist = hists / hists.sum(1, keepdims=True)
            # name each team by the most common color among its own distinctive pixels
            ratio = np.log(self.hist[0] / self.hist[1])
            named = []
            for team, sign in ((0, 1), (1, -1)):
                own = [p for p, t in zip(pixel_samples, labels) if t == team and p is not None and len(p)]
                px = np.vstack(own) if own else np.zeros((0, 3))
                distinct = px[sign * ratio[_quantize(px)] > 1.0] if len(px) else px
                named.append(np.median(distinct if len(distinct) >= 20 else px, 0) if len(px) else self.centers[team])
        else:
            named = list(self.centers)
        names = [color_name(c) for c in named]
        if names[0] == names[1]:
            light = int(named[1][0] > named[0][0])
            names[light] = "Light " + names[light].lower()
            names[1 - light] = "Dark " + names[1 - light].lower()
        self.names = names
        self.separation = float(np.linalg.norm(self.centers[0] - self.centers[1]))

    @staticmethod
    def _bin_center(b):
        l, a, bb = b // (BINS * BINS), (b // BINS) % BINS, b % BINS
        return (np.array([l, a, bb]) + 0.5) * 256.0 / BINS

    def assign_pixels(self, pixel_sets):
        """Team (0/1) of one player from his chest pixels in several frames, or None."""
        sets = [p for p in pixel_sets if p is not None and len(p)]
        if not sets:
            return None
        if self.hist is None:
            meds = [np.median(p, 0) for p in sets]
            return self.assign(np.median(meds, 0))
        ratio = np.log(self.hist[0] / self.hist[1])
        # median over frames: one bad frame (box jumped to a defender) cannot decide
        score = float(np.median([ratio[_quantize(p)].mean() for p in sets]))
        return 0 if score > 0 else 1

    def assign(self, lab):
        if lab is None:
            return None
        d = [np.linalg.norm(c - lab) for c in self.centers]
        return int(np.argmin(d))
