# tracker.py
# Basketball detection and frame-to-frame tracking for SBVS 2.0.
# Owner: Arham.
#
# What this file does, in plain words:
#   1. Reads a video one frame at a time (OpenCV).
#   2. In each frame, finds round shapes that could be the ball (HoughCircles,
#      the same OpenCV function SBVS 1.0 used).
#   3. Throws away shapes that are the wrong size, not orange, or not moving.
#   4. Picks the shape closest to where the ball was last seen.
#   5. Writes one row per frame: frame number, time, and ball x, y, radius
#      (NaN when no ball is found).
#
# Ideas kept from SBVS 1.0: frame differencing (absdiff -> blur -> threshold),
# HoughCircles, and the hoop box with a rim line in the middle.
# New in 2.0: the size, color, and motion checks, the "nearest to last
# position" rule, and the hoop box set by the user instead of hard-coded.

import os

import cv2
import numpy as np
import pandas as pd

import settings


# ---------------------------------------------------------------------------
# Tuning constants (global constants). Change these, not the code below.
# ---------------------------------------------------------------------------

# Frame differencing, same numbers as SBVS 1.0
BLUR_SIZE = 7
MOTION_THRESHOLD = 15

# Orange/brown ball colors in HSV (hue 0-179, saturation 0-255, value 0-255)
ORANGE_LOW = np.array([3, 80, 60])
ORANGE_HIGH = np.array([25, 255, 255])

# A candidate circle must be at least this orange / this moving (0 to 1)
MIN_ORANGE_SHARE = 0.30
MIN_MOTION_SHARE = 0.15

# Accept circles whose radius is within +-45% of the expected ball radius
RADIUS_TOLERANCE = 0.45

# The ball may move at most this many rim widths between two frames
MAX_JUMP_RIM_WIDTHS = 2.0

# After this many frames without the ball, forget where it was
MAX_MISSED_FRAMES = 8

# HoughCircles settings for the grayscale frame
HOUGH_DP = 1.2
HOUGH_PARAM1 = 80   # edge strength (Canny upper threshold)
HOUGH_PARAM2 = 18   # how circular a shape must be (lower = more candidates)


# ---------------------------------------------------------------------------
# The hoop
# ---------------------------------------------------------------------------

class Hoop:
    """The box the user draws around the rim, in working-frame pixels."""

    def __init__(self, x1, y1, x2, y2):
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)

    def width(self):
        return self.x2 - self.x1

    def center(self):
        cx = (self.x1 + self.x2) / 2
        cy = (self.y1 + self.y2) / 2
        return (cx, cy)

    def rim_y(self):
        # The "sensor line" from SBVS 1.0: the middle of the hoop box
        return (self.y1 + self.y2) / 2

    def inside_x(self, x):
        return self.x1 <= x <= self.x2

    def ball_radius(self):
        # A ball (9.4 in wide) is about 0.52 of a rim (18 in) wide,
        # so its radius is about 0.26 of the rim width.
        return 0.26 * self.width()

    def to_dict(self):
        return {'x1': self.x1, 'y1': self.y1, 'x2': self.x2, 'y2': self.y2}

    def __str__(self):
        return f'Hoop box x {self.x1:.0f}-{self.x2:.0f}, y {self.y1:.0f}-{self.y2:.0f}'


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def resize_frame(frame):
    """Resize a frame to settings.WORK_WIDTH pixels wide, keeping its shape."""
    height = frame.shape[0]
    width = frame.shape[1]
    scale = settings.WORK_WIDTH / width
    new_height = int(round(height * scale))
    # cv2.resize takes the new size as (width, height)
    return cv2.resize(frame, (settings.WORK_WIDTH, new_height))


def video_info(video_path):
    """Return a dictionary with the video's frame rate, length, and size."""
    # Open the video file with OpenCV
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    if fps <= 0:
        fps = 30.0   # some files don't report a frame rate; assume 30
    work_height = 0
    if width > 0:
        work_height = int(round(height * settings.WORK_WIDTH / width))

    info = {'fps': fps, 'frame_count': frame_count,
            'width': width, 'height': height,
            'work_width': settings.WORK_WIDTH, 'work_height': work_height}
    return info


def read_frame(video_path, frame_number):
    """Return one frame (resized), or None if it can't be read."""
    capture = cv2.VideoCapture(video_path)
    # Jump to the requested frame, then read it
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return None
    return resize_frame(frame)


def draw_hoop(frame, hoop):
    """Return a copy of the frame with the hoop box and rim line drawn."""
    picture = frame.copy()
    # Box in cyan and rim line in green, the same colors SBVS 1.0 used
    cv2.rectangle(picture, (int(hoop.x1), int(hoop.y1)),
                  (int(hoop.x2), int(hoop.y2)), (255, 255, 0), 2)
    rim = int(hoop.rim_y())
    cv2.line(picture, (int(hoop.x1), rim), (int(hoop.x2), rim), (0, 255, 0), 2)
    return picture


def draw_ball(frame, x, y, r):
    """Return a copy of the frame with a circle where the ball is."""
    picture = frame.copy()
    if np.isnan(x) or np.isnan(y):
        return picture
    radius = 10
    if not np.isnan(r):
        radius = int(r)
    cv2.circle(picture, (int(x), int(y)), radius, (0, 0, 255), 2)
    return picture


def save_positions(positions, path):
    folder = os.path.dirname(path)
    if folder != '' and not os.path.exists(folder):
        os.makedirs(folder)
    positions.to_csv(path, index=False)


def load_positions(path):
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Mask helpers (each mask is a NumPy array: 255 = yes, 0 = no)
# ---------------------------------------------------------------------------

def make_motion_mask(gray, prev_gray):
    """SBVS 1.0 frame differencing: pixels that changed since the last frame."""
    # Absolute difference between this frame and the previous one
    diff = cv2.absdiff(prev_gray, gray)
    # Blur to remove small noise (lighting flicker, camera grain)
    blur = cv2.GaussianBlur(diff, (BLUR_SIZE, BLUR_SIZE), 0)
    # Keep only pixels that changed by more than MOTION_THRESHOLD
    result, mask = cv2.threshold(blur, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
    # Grow the white areas a little so the moving ball is one solid blob
    mask = cv2.dilate(mask, None, iterations=2)
    return mask


def make_orange_mask(frame):
    """Pixels whose color is in the orange/brown ball range."""
    # Convert from BGR to HSV, where color (hue) is one number
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # inRange gives 255 for pixels between ORANGE_LOW and ORANGE_HIGH
    return cv2.inRange(hsv, ORANGE_LOW, ORANGE_HIGH)


def share_inside(mask, x, y, r):
    """Fraction (0 to 1) of mask pixels that are on in the square around a circle."""
    height = mask.shape[0]
    width = mask.shape[1]
    left = max(int(x - r), 0)
    right = min(int(x + r) + 1, width)
    top = max(int(y - r), 0)
    bottom = min(int(y + r) + 1, height)
    patch = mask[top:bottom, left:right]
    if patch.size == 0:
        return 0.0
    return float((patch > 0).mean())


# ---------------------------------------------------------------------------
# The tracker
# ---------------------------------------------------------------------------

class BasketballTracker:
    """Finds the ball in each frame and remembers where it was last seen."""

    def __init__(self, hoop):
        self.hoop = hoop
        self.reset()

    def reset(self):
        self.prev_gray = None
        self.last_x = np.nan
        self.last_y = np.nan
        self.missed_frames = 0

    def find_candidates(self, frame, gray, motion_mask, orange_mask):
        """Return a list of candidate balls as dictionaries x, y, r, orange, motion."""
        expected = self.hoop.ball_radius()
        min_r = max(int(expected * (1 - RADIUS_TOLERANCE)), 2)
        max_r = int(expected * (1 + RADIUS_TOLERANCE)) + 1

        # HoughCircles (from SBVS 1.0) on the blurred grayscale frame.
        # Radius limits now come from the hoop size instead of fixed numbers.
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=HOUGH_DP,
                                   minDist=2 * min_r, param1=HOUGH_PARAM1,
                                   param2=HOUGH_PARAM2, minRadius=min_r,
                                   maxRadius=max_r)

        candidates = []
        if circles is None:
            return candidates

        for circle in circles[0]:
            x = float(circle[0])
            y = float(circle[1])
            r = float(circle[2])
            orange = share_inside(orange_mask, x, y, r)
            motion = share_inside(motion_mask, x, y, r)
            if orange >= MIN_ORANGE_SHARE:
                candidates.append({'x': x, 'y': y, 'r': r,
                                   'orange': orange, 'motion': motion})
        return candidates

    def pick_ball(self, candidates):
        """Choose one candidate, or return None if none is believable."""
        if len(candidates) == 0:
            return None

        xs = np.zeros(len(candidates))
        ys = np.zeros(len(candidates))
        oranges = np.zeros(len(candidates))
        motions = np.zeros(len(candidates))
        for i in range(len(candidates)):
            xs[i] = candidates[i]['x']
            ys[i] = candidates[i]['y']
            oranges[i] = candidates[i]['orange']
            motions[i] = candidates[i]['motion']

        if not np.isnan(self.last_x):
            # We know where the ball was: take the closest candidate, as long
            # as it is not too far away. The allowed jump grows with each
            # frame we missed, because the ball kept moving while unseen.
            distances = np.sqrt((xs - self.last_x) ** 2 + (ys - self.last_y) ** 2)
            allowed = MAX_JUMP_RIM_WIDTHS * self.hoop.width() * (self.missed_frames + 1)
            best = int(np.argmin(distances))
            if distances[best] <= allowed:
                return candidates[best]
            return None

        # We don't know where the ball is: only accept a MOVING orange circle,
        # and take the most orange one. (Stops us locking onto the rim,
        # an orange shirt, or the floor while nothing is happening.)
        moving = motions >= MIN_MOTION_SHARE
        if moving.sum() == 0:
            return None
        scores = np.where(moving, oranges, -1.0)
        best = int(np.argmax(scores))
        return candidates[best]

    def find_ball(self, frame):
        """Return (x, y, r) of the ball in this frame, or (nan, nan, nan)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            # First frame: nothing to compare with yet
            self.prev_gray = gray
            return (np.nan, np.nan, np.nan)

        motion_mask = make_motion_mask(gray, self.prev_gray)
        orange_mask = make_orange_mask(frame)
        candidates = self.find_candidates(frame, gray, motion_mask, orange_mask)
        ball = self.pick_ball(candidates)
        self.prev_gray = gray

        if ball is None:
            self.missed_frames += 1
            if self.missed_frames > MAX_MISSED_FRAMES:
                # Lost it for too long: start fresh
                self.last_x = np.nan
                self.last_y = np.nan
            return (np.nan, np.nan, np.nan)

        self.last_x = ball['x']
        self.last_y = ball['y']
        self.missed_frames = 0
        return (ball['x'], ball['y'], ball['r'])

    def process_video(self, video_path, progress_bar=None):
        """Track the ball through a whole video. Returns a positions DataFrame."""
        info = video_info(video_path)
        self.reset()

        # Open the video and read it frame by frame
        capture = cv2.VideoCapture(video_path)
        rows = []
        frame_number = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = resize_frame(frame)
            x, y, r = self.find_ball(frame)
            rows.append({'frame': frame_number,
                         'time_s': frame_number / info['fps'],
                         'x': x, 'y': y, 'r': r})
            frame_number += 1

            if progress_bar is not None and frame_number % 15 == 0:
                if info['frame_count'] > 0:
                    fraction = min(frame_number / info['frame_count'], 1.0)
                    progress_bar.progress(fraction)
        capture.release()

        if progress_bar is not None:
            progress_bar.progress(1.0)

        positions = pd.DataFrame(rows, columns=settings.POSITION_COLUMNS)
        return positions
