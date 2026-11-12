"""Ball tracking on broadcast footage, and a video with the ball circled.

How the ball is found in each frame (all real detections, nothing drawn by hand):
  1. SEARCH: no recent ball -> run YOLO on the whole frame (COCO class "sports ball").
  2. FOLLOW: ball seen recently -> predict where it moved (constant velocity), cut a
     square around that spot and run YOLO on the crop. In the crop the ball is bigger
     in pixels, so the small-object detector finds it more often than in the full frame.
     The detection closest to the prediction wins; far-away "balls" are ignored.
  3. A camera cut resets the track (the ball jumps on screen at a cut).
  4. If YOLO misses the ball for a frame, the last detected ball image is searched for
     near the predicted spot with template matching (OpenCV matchTemplate). This bridges
     frames where the ball is blurred or partly hidden. It must be re-confirmed by YOLO
     within MAX_TEMPLATE_FRAMES or the track is dropped.
  5. Remaining short gaps are bridged with the predicted position and drawn dashed, so
     viewers can tell guesses from detections.
Sources: YOLO, COCO, small-object crop inference and constant-velocity prediction are
cited in docs/SOURCES.md, section 1 and 7.
"""

import math
import subprocess

import cv2
import numpy as np

from . import cuts

BALL_CLASS = 32
FULL_IMGSZ = 1280        # YOLO input size when searching the whole frame
CROP_IMGSZ = 640         # YOLO input size for the follow crop
SEARCH_CONF = 0.25       # stricter when searching the whole frame (fewer false balls)
FOLLOW_CONF = 0.15       # looser inside the crop, where the ball is expected
MAX_PREDICT_FRAMES = 4   # bridge gaps up to this many frames, then give up
TRAIL_LENGTH = 18
TEMPLATE_MIN_SCORE = 0.55  # matchTemplate similarity needed to accept a tracked ball
MAX_TEMPLATE_FRAMES = 15   # frames we may follow the ball by template alone
USE_TEMPLATE = False       # off by default: on broadcasts it drifted onto players in testing


class BallTracker:
    def __init__(self, model, width, height):
        self.model = model
        self.width = width
        self.height = height
        self.crop_size = int(min(width, height) * 0.45)
        self.reset()

    def reset(self):
        self.pos = None          # (x, y, r) last known
        self.vel = (0.0, 0.0)
        self.missed = 0
        self.template = None     # small image of the ball from the last YOLO detection
        self.since_detect = 0

    def _detect(self, image, conf, imgsz):
        result = self.model.predict(image, classes=[BALL_CLASS], conf=conf, imgsz=imgsz,
                                    verbose=False)[0]
        found = []
        for (x1, y1, x2, y2), c in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
            found.append(((x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) / 2, c))
        return found

    def step(self, frame):
        """Return (x, y, r, conf, kind) or None. kind: 'detected' or 'predicted'."""
        if self.pos is None:
            found = self._detect(frame, SEARCH_CONF, FULL_IMGSZ)
            if not found:
                return None
            best = max(found, key=lambda d: d[3])
            self.pos, self.vel, self.missed = best[:3], (0.0, 0.0), 0
            self._remember(frame, best)
            return best + ("detected",)

        px = self.pos[0] + self.vel[0]
        py = self.pos[1] + self.vel[1]
        half = self.crop_size // 2
        x0 = int(min(max(px - half, 0), self.width - self.crop_size))
        y0 = int(min(max(py - half, 0), self.height - self.crop_size))
        crop = frame[y0:y0 + self.crop_size, x0:x0 + self.crop_size]
        found = [(x + x0, y + y0, r, c) for x, y, r, c in self._detect(crop, FOLLOW_CONF, CROP_IMGSZ)]
        if not found:
            # Ball may have left the crop (fast pass, shot): look at the whole frame too
            found = self._detect(frame, SEARCH_CONF, FULL_IMGSZ)
        limit = max(4 * self.pos[2], 0.12 * self.crop_size) * (1 + self.missed)
        near = [d for d in found if math.hypot(d[0] - px, d[1] - py) < limit]
        if near:
            best = min(near, key=lambda d: math.hypot(d[0] - px, d[1] - py))
            new_vel = (best[0] - self.pos[0], best[1] - self.pos[1])
            self.vel = (0.6 * new_vel[0] + 0.4 * self.vel[0], 0.6 * new_vel[1] + 0.4 * self.vel[1])
            self.pos, self.missed = best[:3], 0
            self._remember(frame, best)
            return best + ("detected",)

        tracked = self._match_template(frame, px, py) if USE_TEMPLATE else None
        if tracked is not None:
            new_vel = (tracked[0] - self.pos[0], tracked[1] - self.pos[1])
            self.vel = (0.6 * new_vel[0] + 0.4 * self.vel[0], 0.6 * new_vel[1] + 0.4 * self.vel[1])
            self.pos, self.missed = (tracked[0], tracked[1], self.pos[2]), 0
            return (tracked[0], tracked[1], self.pos[2], tracked[2], "tracked")

        self.missed += 1
        if self.missed > MAX_PREDICT_FRAMES:
            self.reset()
            return None
        self.pos = (px, py, self.pos[2])
        self.vel = (self.vel[0] * 0.5, self.vel[1] * 0.5)   # trust the guess less each frame
        return (px, py, self.pos[2], 0.0, "predicted")


    def _remember(self, frame, det):
        x, y, r = det[0], det[1], max(det[2], 6)
        x1, y1 = int(max(x - r, 0)), int(max(y - r, 0))
        patch = frame[y1:int(y + r), x1:int(x + r)]
        self.template = patch.copy() if patch.size else None
        self.since_detect = 0

    def _match_template(self, frame, px, py):
        """Find the remembered ball image near (px, py). Returns (x, y, score) or None."""
        self.since_detect += 1
        if self.template is None or self.since_detect > MAX_TEMPLATE_FRAMES:
            return None
        th, tw = self.template.shape[:2]
        win = int(max(th, tw) * 4)
        x0, y0 = int(max(px - win, 0)), int(max(py - win, 0))
        x1, y1 = int(min(px + win, self.width)), int(min(py + win, self.height))
        area = frame[y0:y1, x0:x1]
        if area.shape[0] < th or area.shape[1] < tw:
            return None
        scores = cv2.matchTemplate(area, self.template, cv2.TM_CCOEFF_NORMED)
        _, best, _, loc = cv2.minMaxLoc(scores)
        if best < TEMPLATE_MIN_SCORE:
            return None
        return (x0 + loc[0] + tw / 2, y0 + loc[1] + th / 2, best)


# ---- drawing ------------------------------------------------------------------------

ORANGE = (0, 140, 255)
WHITE = (255, 255, 255)


def draw_dashed_circle(img, center, radius, color, thickness):
    for a in range(0, 360, 30):
        cv2.ellipse(img, center, (radius, radius), 0, a, a + 15, color, thickness, cv2.LINE_AA)


def draw_ball(img, track, trail, scale):
    for i in range(1, len(trail)):
        alpha = i / len(trail)
        p1 = (int(trail[i - 1][0] * scale), int(trail[i - 1][1] * scale))
        p2 = (int(trail[i][0] * scale), int(trail[i][1] * scale))
        cv2.line(img, p1, p2, ORANGE, max(1, int(5 * alpha)), cv2.LINE_AA)
    if track is None:
        return
    x, y, r, conf, kind = track
    c = (int(x * scale), int(y * scale))
    radius = max(10, int(r * scale * 1.6))
    if kind == "detected":
        overlay = img.copy()
        cv2.circle(overlay, c, radius + 6, ORANGE, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
        cv2.circle(img, c, radius, ORANGE, 3, cv2.LINE_AA)
        label = f"BALL {conf:.2f}"
    elif kind == "tracked":
        cv2.circle(img, c, radius, ORANGE, 2, cv2.LINE_AA)
        label = "BALL (tracked)"
    else:
        draw_dashed_circle(img, c, radius, ORANGE, 2)
        label = "BALL (predicted)"
    cv2.putText(img, label, (c[0] + radius + 6, c[1] - radius), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, WHITE, 2, cv2.LINE_AA)


def draw_hud(img, t, scores, flash, stats):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 44), (20, 20, 20), -1)
    cv2.putText(img, "SBVS 2.0  |  live ball tracking", (14, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, ORANGE, 2, cv2.LINE_AA)
    right = f"t={t:5.1f}s   ball found {stats['detected'] + stats['tracked']}/{stats['frames']} frames"
    if scores:
        right = "   ".join(f"{k} {v}" for k, v in scores.items()) + "   |   " + right
    size = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.putText(img, right, (w - size[0] - 14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)
    if flash:
        text = flash
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.6, 3)[0]
        x = (w - size[0]) // 2
        cv2.rectangle(img, (x - 20, 64), (x + size[0] + 20, 64 + size[1] + 30), ORANGE, -1)
        cv2.putText(img, text, (x, 64 + size[1] + 14), cv2.FONT_HERSHEY_DUPLEX, 1.6, WHITE, 3, cv2.LINE_AA)


# ---- whole clip ---------------------------------------------------------------------

def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def render_tracking_video(path, out_path, model, start_s=0.0, length_s=30.0, out_width=1280,
                          events=None, start_scores=None, progress=None):
    """Write an H.264 MP4 (plays in browsers) of [start_s, start_s+length_s] with the ball
    circled. `events` (optional) is the scoring-events table from analyze_broadcast; it
    drives the score in the top bar and a '+3 DUKE' banner when a basket is confirmed.
    Returns a stats dict."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
    scale = out_width / width
    out_h = int(round(height * scale / 2) * 2)
    proc = subprocess.Popen(
        [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{out_width}x{out_h}", "-r", f"{fps:.3f}", "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "veryfast",
         "-movflags", "+faststart", out_path], stdin=subprocess.PIPE)

    tracker = BallTracker(model, width, height)
    trail = []
    prev_hist = None
    stats = {"frames": 0, "detected": 0, "tracked": 0, "predicted": 0}
    total = int(length_s * fps)
    for i in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0   # real timestamp (variable frame rate safe)
        hist = cuts.frame_histogram(cv2.resize(frame, (320, 240)))
        if cuts.is_cut(prev_hist, hist):
            tracker.reset()
            trail = []
        prev_hist = hist

        was_tracking = tracker.pos is not None
        track = tracker.step(frame)
        stats["frames"] += 1
        if track is not None:
            stats[track[4]] += 1
            if not was_tracking:
                trail = []          # new track: do not connect it to the old one
            trail.append(track[:2])
            trail = trail[-TRAIL_LENGTH:]
        elif trail:
            trail = trail[1:]

        scores, flash = None, None
        if events is not None and start_scores:
            scores = dict(start_scores)
            clean = events[events["flag"] == ""]
            for _, ev in clean[clean["time_s"] <= t].iterrows():
                scores[ev["team"]] = int(ev["score_after"])
                if t - ev["time_s"] < 2.5:
                    flash = f"+{int(ev['points'])}  {str(ev['team']).upper()}"

        out = cv2.resize(frame, (out_width, out_h), interpolation=cv2.INTER_AREA)
        draw_ball(out, track, trail, scale)
        draw_hud(out, t, scores, flash, stats)
        proc.stdin.write(out.tobytes())
        if progress is not None:
            progress((i + 1) / total)
    cap.release()
    proc.stdin.close()
    proc.wait()
    stats["detected_share"] = round(stats["detected"] / max(1, stats["frames"]), 3)
    return stats
