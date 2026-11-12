"""Camera-cut (shot boundary) detection.

Technique: compare the color histogram of each sampled frame with the previous one.
Inside one camera shot the colors barely change even while the camera pans; at a
hard cut they change all at once. This is the classic histogram-difference method
from the shot-boundary-detection literature (see docs/SOURCES.md, "Cut detection").
"""

import cv2
import numpy as np

# Correlation between consecutive histograms below this value counts as a cut.
# 1.0 = identical colors, 0 = unrelated. Tuned on the synthetic clip; check on real footage.
CUT_THRESHOLD = 0.6


def frame_histogram(frame):
    """Normalized hue/saturation histogram of a BGR frame (brightness ignored,
    so flashes and lighting changes matter less)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def is_cut(prev_hist, hist, threshold=CUT_THRESHOLD):
    """True when two histograms are different enough to be a camera cut."""
    if prev_hist is None:
        return False
    similarity = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
    return similarity < threshold


def segments_from_cut_times(cut_times, duration):
    """Turn a list of cut times (seconds) into [(start, end), ...] segments."""
    bounds = [0.0] + sorted(cut_times) + [duration]
    segments = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        if end - start > 0.05:
            segments.append((start, end))
    return segments


def mean_similarity(hists):
    """Average histogram correlation of a list (debug helper)."""
    if len(hists) < 2:
        return 1.0
    values = [cv2.compareHist(a, b, cv2.HISTCMP_CORREL) for a, b in zip(hists[:-1], hists[1:])]
    return float(np.mean(values))
