"""Jersey number reading with Tesseract OCR (digits only).

On broadcast footage numbers are often small, blurred or turned away, so we read the
shooter's torso in several frames around the shot and only accept a number that is read
the same way at least MIN_VOTES times. Otherwise the shot counts for the team only.
"""

from collections import Counter

import cv2

from .teams import torso_crop

MIN_HEIGHT_PX = 70      # player boxes shorter than this are too small to read
MIN_VOTES = 2
_OCR_CONFIG = "--psm 7 -c tessedit_char_whitelist=0123456789"


def available():
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def read_number(frame, box):
    """Best single read ('23') from one player box, or None."""
    if box[3] - box[1] < MIN_HEIGHT_PX:
        return None
    import pytesseract
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    h = y2 - y1
    crop = frame[y1 + int(0.22 * h): y1 + int(0.62 * h), x1 + int(0.1 * (x2 - x1)): x2 - int(0.1 * (x2 - x1))]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    best = None
    for invert in (False, True):
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if invert:
            bw = 255 - bw
        data = pytesseract.image_to_data(bw, config=_OCR_CONFIG, output_type=pytesseract.Output.DICT)
        for text, conf in zip(data["text"], data["conf"]):
            text = text.strip()
            try:
                conf = float(conf)
            except ValueError:
                continue
            if 1 <= len(text) <= 2 and conf >= 60 and (best is None or conf > best[1]):
                best = (text, conf)
    return best[0] if best else None


def vote(reads):
    """Most common read if it appears at least MIN_VOTES times."""
    reads = [r for r in reads if r]
    if not reads:
        return None
    number, count = Counter(reads).most_common(1)[0]
    return number if count >= MIN_VOTES else None
