"""Watch a game video and return every shot with result, team and jersey number.

Frame sampling: BASE_FPS frames per second normally, BURST_FPS while the ball is near a
hoop or a shot is open (shots last about a second, so they need more frames).
Camera cuts (histogram change) reset the ball/hoop state, as positions before and after
a cut are not comparable. A made shot within REPLAY_S seconds after another made shot,
with a cut in between, is flagged as a possible replay and not counted.
"""

import math
import time
from collections import deque

import cv2
import numpy as np
import pandas as pd

from . import jersey, shots as shot_rules, teams
from .detector import Detector

BASE_FPS = 5
BURST_FPS = 15
NEAR_HOOP = 4.0          # hoop widths: closer than this switches to BURST_FPS
MAX_WIDTH = 1280         # bigger frames are downscaled first (speed)
CUT_THRESHOLD = 0.55     # histogram correlation below this = camera cut
BUFFER_S = 3.0           # frames kept for finding the shooter
RELEASE_LOOKBACK_S = 2.5
TEAM_SAMPLE_EVERY_S = 1.0
MAX_TEAM_SAMPLES = 3000
REPLAY_S = 20.0
MAX_BALL_JUMP = 0.35     # fraction of frame width the ball may move between checks


def _hist(frame):
    hsv = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(h, h)
    return h


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def ball_in_hands(ball, box):
    """Ball center inside the player's box (a bit wider), at hand height or above."""
    x1, y1, x2, y2 = box[:4]
    w, h = x2 - x1, y2 - y1
    return (x1 - 0.2 * w <= ball[0] <= x2 + 0.2 * w) and (y1 - 0.35 * h <= ball[1] <= y1 + 0.65 * h)


def pick_ball(balls, last, width):
    if not balls:
        return None
    if last is not None:
        near = [b for b in balls if math.dist(_center(b), last) < MAX_BALL_JUMP * width]
        if near:
            return min(near, key=lambda b: math.dist(_center(b), last))
    return balls[0]


def find_shooter(buffer, event):
    """Latest buffered frame (up to the attempt) where a player holds the ball.
    Returns (frame_entry, player_box) or (None, None)."""
    for entry in reversed(buffer):
        if entry["t"] > event.t_attempt or event.t_attempt - entry["t"] > RELEASE_LOOKBACK_S:
            continue
        ball = entry["ball"]
        if ball is None:
            continue
        holders = [p for p in entry["players"] if ball_in_hands(ball, p)]
        if holders:
            best = min(holders, key=lambda p: math.dist(_center(p), ball))
            return entry, best
    return None, None


def follow_back(buffer, entry, box, n=6):
    """The same player in earlier buffered frames (IoU chaining), for more OCR tries."""
    out = [(entry, box)]
    idx = buffer.index(entry)
    cur = box
    for e in reversed(list(buffer)[:idx]):
        if not e["players"]:
            continue
        best = max(e["players"], key=lambda p: _iou(p, cur))
        if _iou(best, cur) < 0.3:
            break
        out.append((e, best))
        cur = best
        if len(out) >= n:
            break
    return out


def thumbnail(frame, shooter_box, hoop, result):
    img = frame.copy()
    color = {"made": (80, 200, 80), "missed": (60, 60, 230)}.get(result, (0, 200, 240))
    if shooter_box is not None:
        x1, y1, x2, y2 = [int(v) for v in shooter_box[:4]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (38, 132, 245), 3)
    if hoop is not None:
        x1, y1, x2, y2 = [int(v) for v in hoop[:4]]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    scale = 480 / img.shape[1]
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jpg.tobytes() if ok else None


def analyze_video(path, detector=None, progress=None, max_seconds=None):
    """Return dict(shots=DataFrame, team_names=[..], thumbs={shot_id: jpg}, meta={..})."""
    started = time.time()
    detector = detector or Detector()
    use_ocr = jersey.available()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open the video file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    duration = total / fps if total else 0.0
    if max_seconds:
        duration = min(duration, max_seconds) if duration else max_seconds

    tracker = shot_rules.ShotTracker()
    buffer = deque()
    team_samples = []
    team_pixels = []
    raw_events = []            # (event, shooter_color, number, thumb)
    cut_times = []
    prev_hist = None
    last_ball = None
    last_sample_t = -1e9
    frame_idx = -1
    next_t = 0.0
    checked = 0

    while True:
        ok = cap.grab()
        if not ok:
            break
        frame_idx += 1
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t <= 0 and frame_idx > 0:
            t = frame_idx / fps
        if max_seconds and t > max_seconds:
            break
        if t + 1e-6 < next_t:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        if frame.shape[1] > MAX_WIDTH:
            s = MAX_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        W = frame.shape[1]
        checked += 1

        hist = _hist(frame)
        if prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < CUT_THRESHOLD:
            ev = tracker.flush(t)
            if ev is not None:
                raw_events.append(_describe(ev, buffer, use_ocr))
            tracker.reset()
            buffer.clear()
            last_ball = None
            cut_times.append(t)
        prev_hist = hist

        det = detector.detect(frame)
        ball_box = pick_ball(det["ball"], last_ball, W)
        ball = _center(ball_box) if ball_box else None
        last_ball = ball if ball else last_ball
        players = [p[:4] for p in det["player"]]

        buffer.append({"t": t, "frame": frame, "ball": ball, "players": players})
        while buffer and t - buffer[0]["t"] > BUFFER_S:
            buffer.popleft()

        if t - last_sample_t >= TEAM_SAMPLE_EVERY_S and len(team_samples) < MAX_TEAM_SAMPLES:
            last_sample_t = t
            for p in players:
                px = teams.torso_pixels(frame, p)
                if px is not None:
                    team_samples.append(np.median(px, 0))
                    keep = np.random.default_rng(len(team_samples)).choice(len(px), min(len(px), 300), replace=False)
                    team_pixels.append(px[keep])

        ev = tracker.update(t, ball, det["hoop"])
        if ev is not None:
            raw_events.append(_describe(ev, buffer, use_ocr))

        near = tracker.attempt is not None
        if not near and ball is not None and tracker.hoop is not None:
            _, _, w, _, center = shot_rules.hoop_geometry(tracker.hoop)
            near = math.dist(ball, center) < NEAR_HOOP * w
        next_t = t + 1.0 / (BURST_FPS if near else BASE_FPS)

        if progress and checked % 10 == 0 and duration:
            progress(min(t / duration, 1.0), f"Watching the game: {t:.0f} of {duration:.0f} s")

    ev = tracker.flush(next_t)
    if ev is not None:
        raw_events.append(_describe(ev, buffer, use_ocr))
    cap.release()

    team_model = None
    if len(team_samples) >= 4:
        team_model = teams.TeamModel(team_samples, team_pixels)
    shots_df, thumbs = _build_table(raw_events, team_model, cut_times)
    meta = {"duration_s": round(duration, 1), "checked_frames": checked,
            "processing_s": round(time.time() - started, 1), "cuts": len(cut_times),
            "ocr": use_ocr, "team_separation": team_model.separation if team_model else None}
    if progress:
        progress(1.0, "Done")
    return {"shots": shots_df, "team_names": team_model.names if team_model else ["Team A", "Team B"],
            "thumbs": thumbs, "meta": meta}


def _describe(ev, buffer, use_ocr):
    entry, box = find_shooter(buffer, ev)
    pixels, number, thumb = None, None, None
    if entry is not None:
        chain = follow_back(buffer, entry, box)
        held = [(e, b) for e, b in chain if e["ball"] and ball_in_hands(e["ball"], b)]
        pixels = [teams.chest_under_ball_pixels(e["frame"], b, e["ball"]) for e, b in held] or \
                 [teams.torso_pixels(entry["frame"], box)]
        if use_ocr:
            number = jersey.vote([jersey.read_number(e["frame"], b) for e, b in chain])
        thumb = thumbnail(entry["frame"], box, ev.hoop, ev.result)
    elif buffer:
        thumb = thumbnail(buffer[-1]["frame"], None, ev.hoop, ev.result)
    return ev, pixels, number, thumb


def _build_table(raw_events, team_model, cut_times):
    rows, thumbs = [], {}
    last_made_t = None
    for i, (ev, pixels, number, thumb) in enumerate(raw_events, start=1):
        team_idx = team_model.assign_pixels(pixels) if (team_model is not None and pixels) else None
        flag = ""
        if ev.result == "made" and last_made_t is not None and ev.t_attempt - last_made_t < REPLAY_S \
                and any(last_made_t < c < ev.t_attempt for c in cut_times):
            flag = "possible replay"
        if ev.result == "made" and not flag:
            last_made_t = ev.t_result
        rows.append({"shot_id": i, "time_s": round(ev.t_attempt, 1), "result": ev.result,
                     "team": team_idx, "number": number or "", "shooter_found": pixels is not None,
                     "reason": ev.reason, "flag": flag})
        if thumb:
            thumbs[i] = thumb
    cols = ["shot_id", "time_s", "result", "team", "number", "shooter_found", "reason", "flag"]
    return pd.DataFrame(rows, columns=cols), thumbs
