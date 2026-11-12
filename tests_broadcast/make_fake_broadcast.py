"""Make a FAKE broadcast-style clip with known answers, to prove the pipeline runs.

It has: a panning camera over a court, hard cuts to a close-up and a replay, a score
graphic that is hidden during the replay, and score changes at known times.
It does NOT prove accuracy on real games (no real players, real ball, or real graphics).

Usage:  python tests_broadcast/make_fake_broadcast.py
Writes: tests_broadcast/fake_broadcast.mp4, fake_truth_events.csv, fake_truth_cuts.csv,
        fake_score_boxes.csv
"""

import os

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, FPS, SECONDS = 1280, 720, 30, 60

# (time_s, team, points) - the "truth" the pipeline should find
SCORING = [(5.0, "LAL", 2), (12.0, "BOS", 3), (18.0, "LAL", 1), (19.5, "LAL", 1),
           (31.0, "BOS", 2), (41.0, "LAL", 3), (47.0, "LAL", 2), (52.0, "BOS", 2)]
START = {"LAL": 98, "BOS": 95}

# (start, end, kind) - camera shots. kind: game / closeup / replay
SHOTS = [(0, 22, "game"), (22, 26, "closeup"), (26, 30, "replay"),
         (30, 44, "game"), (44, 46, "closeup"), (46, 60, "game")]

# Where the two scores sit on screen (what a user would draw in the app)
BOXES = {"LAL": (150, 640, 230, 690), "BOS": (390, 640, 470, 690)}


def make_court():
    court = np.zeros((H, W * 2, 3), np.uint8)
    court[:] = (60, 120, 190)                          # wood (BGR)
    rng = np.random.default_rng(1)
    for x in range(0, W * 2, 12):                      # floor boards
        shade = int(rng.integers(-15, 15))
        court[:, x:x + 10] = np.clip(np.array((60, 120, 190)) + shade, 0, 255)
    court[:200] = (40, 40, 50)                         # crowd / wall
    cv2.line(court, (W, 200), (W, H), (255, 255, 255), 4)
    cv2.circle(court, (W, 460), 120, (255, 255, 255), 4)
    for hoop_x in (160, W * 2 - 160):
        cv2.rectangle(court, (hoop_x - 60, 220), (hoop_x + 60, 300), (240, 240, 240), -1)
        cv2.ellipse(court, (hoop_x, 310), (35, 8), 0, 0, 360, (0, 90, 255), 4)
    return court


def score_at(t):
    score = dict(START)
    for when, team, pts in SCORING:
        if when <= t:
            score[team] += pts
    return score


def draw_scorebug(frame, t):
    s = score_at(t)
    cv2.rectangle(frame, (40, 630), (520, 700), (30, 20, 20), -1)
    cv2.putText(frame, "LAL", (60, 678), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (80, 200, 255), 3)
    cv2.putText(frame, str(s["LAL"]), (155, 680), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    cv2.putText(frame, "BOS", (300, 678), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (80, 200, 80), 3)
    cv2.putText(frame, str(s["BOS"]), (395, 680), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)


def shot_kind(t):
    for start, end, kind in SHOTS:
        if start <= t < end:
            return kind
    return "game"


def main():
    court = make_court()
    out_path = os.path.join(HERE, "fake_broadcast.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(FPS * SECONDS):
        t = i / FPS
        kind = shot_kind(t)
        if kind == "game":
            pan = int((np.sin(t / 6.0) * 0.5 + 0.5) * W)          # camera pans left/right
            frame = court[:, pan:pan + W].copy()
            for p in range(10):                                    # "players"
                px = int((p * 230 + t * 90 * (1 if p % 2 else -1)) % (W * 2)) - pan
                color = (80, 200, 255) if p < 5 else (80, 200, 80)
                cv2.rectangle(frame, (px, 380), (px + 40, 520), color, -1)
        elif kind == "closeup":
            frame = np.full((H, W, 3), (90, 60, 140), np.uint8)
            cv2.circle(frame, (W // 2, H // 2), 200, (150, 170, 230), -1)
        else:  # replay: zoomed, tinted, with a REPLAY tag
            zoom = cv2.resize(court[200:560, 300:940], (W, H))
            frame = cv2.addWeighted(zoom, 0.7, np.full_like(zoom, (120, 40, 40)), 0.3, 0)
            cv2.putText(frame, "REPLAY", (W - 300, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
        if kind != "replay":
            draw_scorebug(frame, t)
        writer.write(frame)
    writer.release()

    pd.DataFrame(SCORING, columns=["time_s", "team", "points"]).to_csv(
        os.path.join(HERE, "fake_truth_events.csv"), index=False)
    pd.DataFrame({"cut_time_s": [s for s, _, _ in SHOTS[1:]]}).to_csv(
        os.path.join(HERE, "fake_truth_cuts.csv"), index=False)
    rows = [{"team": k, "x1": v[0], "y1": v[1], "x2": v[2], "y2": v[3]} for k, v in BOXES.items()]
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "fake_score_boxes.csv"), index=False)
    print("wrote", out_path, "final score", score_at(SECONDS))


if __name__ == "__main__":
    main()
