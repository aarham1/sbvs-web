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


PIXEL_MATCH = 30.0   # Lab distance for a pixel to count as a team color


def torso_pixels(frame, box):
    crop = torso_crop(frame, box)
    if crop is None:
        return None
    return cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)


def color_name(lab):
    return min(_NAME_LAB, key=lambda n: np.linalg.norm(_NAME_LAB[n] - lab))


class TeamModel:
    """Fit on many jersey colors, then assign any player to team 0 or 1."""

    def __init__(self, samples, seed=0):
        data = np.float32(samples)
        if len(data) < 4:
            raise ValueError("not enough player samples to split teams")
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        cv2.setRNGSeed(seed)
        _, labels, centers = cv2.kmeans(data, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.flatten(), minlength=2)
        order = np.argsort(-counts)          # team 0 = the color seen most often
        self.centers = centers[order].astype(float)
        names = [color_name(c) for c in self.centers]
        if names[0] == names[1]:
            light = int(self.centers[1][0] > self.centers[0][0])
            names[light] = "Light " + names[light].lower()
            names[1 - light] = "Dark " + names[1 - light].lower()
        self.names = names
        self.separation = float(np.linalg.norm(self.centers[0] - self.centers[1]))

    def assign_pixels(self, pixel_sets):
        """Vote over the torso pixels of one player in several frames: each pixel goes to
        the nearer team color if it is close to one of them (court, skin and shadows
        usually are not). Robust to a defender overlapping the box."""
        votes = np.zeros(2)
        for px in pixel_sets:
            if px is None or len(px) == 0:
                continue
            d = np.linalg.norm(px[:, None, :] - self.centers[None, :, :], axis=2)
            near = d.min(1) < PIXEL_MATCH
            if near.any():
                votes += np.bincount(d[near].argmin(1), minlength=2) / len(px)
        if votes.sum() == 0:
            return None
        return int(np.argmax(votes))

    def assign(self, lab):
        if lab is None:
            return None
        d = [np.linalg.norm(c - lab) for c in self.centers]
        return int(np.argmin(d))
