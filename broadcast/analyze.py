"""Run the broadcast pipeline on one video and build the stats tables.

    result = analyze_broadcast("game.mp4", boxes, ocr=EasyOcrReader(), detector=Detector())

`boxes` is a list of scoreboard.ScoreBox (one per team). `detector` is optional; without
it there is no game-view / ball information, but scoring stats still work.
This file has no Streamlit code, so a future backend API (Phase 3) can call it unchanged.
"""

import cv2
import pandas as pd

from . import cuts
from . import detect as detect_mod
from .scoreboard import ScoreTracker, read_scores

SAMPLE_FPS = 2.0   # frames analyzed per second of video


def video_meta(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "frames": frames, "width": width, "height": height,
            "duration_s": frames / fps if fps else 0.0}


def read_frame_at(path, time_s):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000.0)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def analyze_broadcast(path, boxes, ocr, detector=None, sample_fps=SAMPLE_FPS, progress=None):
    meta = video_meta(path)
    step = max(1, int(round(meta["fps"] / sample_fps)))
    trackers = {box.team: ScoreTracker(box.team) for box in boxes}

    samples = []          # one row per analyzed frame
    cut_times = []
    prev_hist = None

    cap = cv2.VideoCapture(path)
    index = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if index % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            # Use the frame's own timestamp: screen recordings often have a variable
            # frame rate, so index / fps drifts by several seconds over a few minutes.
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            hist = cuts.frame_histogram(frame)
            if cuts.is_cut(prev_hist, hist):
                cut_times.append(t)
            prev_hist = hist

            values = read_scores(frame, boxes, ocr)
            for team, value in values.items():
                trackers[team].update(t, value)

            row = {"time_s": round(t, 2)}
            for team, value in values.items():
                row["score_" + team] = value
            if detector is not None:
                found = detector.detect(frame)
                row["people"] = found["people"]
                ball = found["ball"]
                row["ball_x"], row["ball_y"], row["ball_r"] = ball if ball else (None, None, None)
            samples.append(row)
            if progress is not None and meta["frames"]:
                progress(min(1.0, index / meta["frames"]))
        index += 1
    cap.release()

    samples_df = pd.DataFrame(samples)
    segments_df = build_segments(cut_times, meta["duration_s"], samples_df)
    events = []
    for tracker in trackers.values():
        events.extend(tracker.events)
    events_df = pd.DataFrame(events, columns=["time_s", "team", "points", "score_after", "flag"])
    events_df = events_df.sort_values("time_s").reset_index(drop=True)
    teams = [box.team for box in boxes]
    return {
        "meta": meta,
        "samples": samples_df,
        "segments": segments_df,
        "events": events_df,
        "team_stats": team_stats(events_df, teams),
        "runs": scoring_runs(events_df),
        "start_scores": start_scores(events_df, trackers),
    }


def build_segments(cut_times, duration, samples_df):
    rows = []
    for start, end in cuts.segments_from_cut_times(cut_times, duration):
        row = {"start_s": round(start, 2), "end_s": round(end, 2),
               "length_s": round(end - start, 2)}
        if "people" in samples_df.columns and len(samples_df):
            inside = samples_df[(samples_df["time_s"] >= start) & (samples_df["time_s"] < end)]
            people = float(inside["people"].median()) if len(inside) else 0.0
            row["median_people"] = people
            row["view"] = "game" if detect_mod.is_game_view(people) else "other"
        rows.append(row)
    return pd.DataFrame(rows)


def start_scores(events_df, trackers):
    """Score each team had before its first clean scoring event (for the chart).
    Readings before the scoreboard settles (flagged 'check') are skipped."""
    out = {}
    for team, tracker in trackers.items():
        clean = events_df[(events_df["team"] == team) & (events_df["flag"] == "")]
        if len(clean):
            first = clean.iloc[0]
            out[team] = int(first["score_after"] - first["points"])
        else:
            out[team] = tracker.score
    return out


def team_stats(events_df, teams):
    """Points and made baskets by type for each team (flagged changes excluded)."""
    rows = []
    clean = events_df[events_df["flag"] == ""]
    for team in teams:
        mine = clean[clean["team"] == team]
        rows.append({
            "team": team,
            "points": int(mine["points"].sum()),
            "made_FT": int((mine["points"] == 1).sum()),
            "made_2PT": int((mine["points"] == 2).sum()),
            "made_3PT": int((mine["points"] == 3).sum()),
            "points_from_3": int(mine.loc[mine["points"] == 3, "points"].sum()),
            "unclear_changes": int(((events_df["team"] == team) & (events_df["flag"] != "")).sum()),
        })
    return pd.DataFrame(rows)


def scoring_runs(events_df):
    """Consecutive points by one team while the other team does not score."""
    runs = []
    current = None
    for _, ev in events_df[events_df["flag"] == ""].iterrows():
        if current is not None and current["team"] == ev["team"]:
            current["points"] += int(ev["points"])
            current["end_s"] = ev["time_s"]
        else:
            if current is not None:
                runs.append(current)
            current = {"team": ev["team"], "points": int(ev["points"]),
                       "start_s": ev["time_s"], "end_s": ev["time_s"]}
    if current is not None:
        runs.append(current)
    df = pd.DataFrame(runs, columns=["team", "points", "start_s", "end_s"])
    return df.sort_values("points", ascending=False).reset_index(drop=True)
