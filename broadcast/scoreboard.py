"""Read the on-screen scoreboard and turn score changes into scoring events.

Why this works on broadcasts: the camera moves, but the score graphic stays in the
same place on screen. The user draws a box around each team's score once; every
sampled frame we crop those boxes, read the digits with OCR, and watch for changes.
A change of +1, +2 or +3 is a made free throw, two or three.

OCR engine: EasyOCR (deep-learning text recognizer, see docs/SOURCES.md, "OCR").
Any object with a .read(image) -> str method can be passed instead (used in tests).
"""

import re

import cv2
import numpy as np

# A new score must be read this many samples in a row before we believe it.
# Protects against one-frame OCR mistakes and graphics animating in.
CONFIRM_SAMPLES = 3
VALID_JUMPS = (1, 2, 3)
# OCR pieces below this confidence are ignored. Captions or logos forced into digits
# usually score low; real scoreboard digits usually score near 1.0.
MIN_OCR_CONFIDENCE = 0.3


class ScoreBox:
    """A rectangle on screen (in original video pixels) that holds one team's score."""

    def __init__(self, team, x1, y1, x2, y2):
        self.team = team
        self.x1, self.y1, self.x2, self.y2 = int(x1), int(y1), int(x2), int(y2)

    def crop(self, frame):
        return frame[self.y1:self.y2, self.x1:self.x2]

    def to_dict(self):
        return {"team": self.team, "x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


def prepare_for_ocr(crop):
    """Grayscale + enlarge small score digits so OCR reads them more reliably."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    scale = max(1.0, 64.0 / max(1, gray.shape[0]))
    if scale > 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return gray


class EasyOcrReader:
    """Wraps EasyOCR so it only returns digits. Loads the model once (slow, ~100 MB)."""

    def __init__(self, gpu=False):
        import easyocr  # imported here so the rest of the app works without it
        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    def read(self, image):
        results = self.reader.readtext(image, allowlist="0123456789")
        pieces = []
        for _box, text, conf in results:
            if conf >= MIN_OCR_CONFIDENCE:
                pieces.append(text)
        return "".join(pieces)


def parse_score(text):
    """'  108 ' -> 108. Returns None when there is no believable score (0-199)."""
    match = re.search(r"\d{1,3}", text or "")
    if match is None:
        return None
    value = int(match.group(0))
    if value > 199:
        return None
    return value


class ScoreTracker:
    """Follows one team's score through noisy OCR readings and emits scoring events."""

    def __init__(self, team):
        self.team = team
        self.score = None          # last confirmed score
        self.candidate = None      # value we are currently confirming
        self.candidate_count = 0
        self.candidate_time = None
        self.events = []           # dicts: time_s, team, points, score_after, flag

    def update(self, time_s, value):
        if value is None:
            # Graphic hidden (replay, timeout, close-up) or unreadable: no information.
            self.candidate = None
            self.candidate_count = 0
            return
        if value == self.score:
            self.candidate = None
            self.candidate_count = 0
            return
        if value != self.candidate:
            self.candidate = value
            self.candidate_count = 1
            self.candidate_time = time_s
        else:
            self.candidate_count += 1
        if self.candidate_count >= CONFIRM_SAMPLES:
            self._confirm(self.candidate, self.candidate_time)
            self.candidate = None
            self.candidate_count = 0

    def _confirm(self, value, time_s):
        if self.score is None:
            self.score = value     # first reading of the clip is just the starting score
            return
        jump = value - self.score
        if jump in VALID_JUMPS:
            flag = ""
        else:
            # Missed part of the video, a correction by the scorer, or an OCR error.
            flag = "check"
        self.events.append({
            "time_s": round(time_s, 2),
            "team": self.team,
            "points": jump,
            "score_after": value,
            "flag": flag,
        })
        self.score = value


def read_scores(frame, boxes, ocr):
    """Read every score box in one frame. Returns {team: value or None}."""
    values = {}
    for box in boxes:
        crop = box.crop(frame)
        if crop.size == 0:
            values[box.team] = None
            continue
        values[box.team] = parse_score(ocr.read(prepare_for_ocr(crop)))
    return values
