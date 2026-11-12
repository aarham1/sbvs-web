"""SBVS website - Broadcast analysis page (Phase 2 MVP).

Upload TV-style game footage, mark where each team's score is on screen, and get back
team scoring stats, a scoring timeline, scoring runs and camera-cut segments.
"""

import os
import tempfile

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from broadcast.analyze import analyze_broadcast, read_frame_at, video_meta
from broadcast.scoreboard import ScoreBox

st.set_page_config(page_title="SBVS - Broadcast", layout="wide")
st.title("Broadcast game analysis")
st.caption("Experimental. Scoring stats come from reading the on-screen scoreboard; "
           "check every result against the video. Accuracy has not been measured on real games yet.")
st.info("On the free website this page is slow and has little memory: use clips of about "
        "1 to 3 minutes. For a ready-made example, open the **Demo** page.")

LOCAL_YOLO = os.path.join("broadcast", "yolo11n.pt")  # shipped with the repo, no download


def show_failure(step, err):
    """Show a friendly message instead of crashing the whole page."""
    if isinstance(err, ImportError):
        st.error(f"{step} needs PyTorch, EasyOCR and Ultralytics, which are not installed here. "
                 "Run the site on a laptop (`pip install -r requirements.txt`) or open the Demo page.")
    elif isinstance(err, MemoryError):
        st.error(f"{step} ran out of memory. Try a shorter clip, fewer frames per second, "
                 "or the Fast detector.")
    else:
        st.error(f"{step} failed: {err}. Try a shorter clip or open the Demo page.")


@st.cache_resource
def get_ocr():
    from broadcast.scoreboard import EasyOcrReader
    return EasyOcrReader()


@st.cache_resource
def get_detector():
    from broadcast.detect import Detector
    return Detector(LOCAL_YOLO) if os.path.exists(LOCAL_YOLO) else Detector()


# ---- 1. Upload --------------------------------------------------------------------
st.header("1. Upload a clip")
uploaded = st.file_uploader("Game footage (mp4, mov, mkv). Short clips work best.",
                            type=["mp4", "mov", "mkv"])
if uploaded is None:
    st.info("Tip: a few minutes of a broadcast with the scoreboard visible is a good first test.")
    st.stop()

if st.session_state.get("video_name") != uploaded.name:
    suffix = os.path.splitext(uploaded.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.close()
    st.session_state["video_path"] = tmp.name
    st.session_state["video_name"] = uploaded.name
    st.session_state.pop("result", None)

path = st.session_state["video_path"]
meta = video_meta(path)
st.write(f"{meta['width']}x{meta['height']}, {meta['fps']:.0f} fps, {meta['duration_s']:.0f} s")

# ---- 2. Mark the scoreboard --------------------------------------------------------
st.header("2. Mark each team's score")
st.write("Pick a moment where the scoreboard is visible, then move the boxes so each one "
         "covers only that team's score digits (not the team name).")
t_preview = st.slider("Preview time (s)", 0.0, max(0.0, meta["duration_s"] - 0.1), 1.0, 0.5)
frame = read_frame_at(path, t_preview)

w, h = meta["width"], meta["height"]
cols = st.columns(2)
boxes = []
defaults = [("Team A", 0.10, 0.88, 0.18, 0.96), ("Team B", 0.30, 0.88, 0.38, 0.96)]
for col, (name, fx1, fy1, fx2, fy2) in zip(cols, defaults):
    with col:
        team = st.text_input("Team name", name, key="name_" + name)
        x1 = st.number_input("left (px)", 0, w, int(fx1 * w), key="x1" + name)
        x2 = st.number_input("right (px)", 0, w, int(fx2 * w), key="x2" + name)
        y1 = st.number_input("top (px)", 0, h, int(fy1 * h), key="y1" + name)
        y2 = st.number_input("bottom (px)", 0, h, int(fy2 * h), key="y2" + name)
        boxes.append(ScoreBox(team, x1, y1, x2, y2))

if frame is not None:
    preview = frame.copy()
    for box, color in zip(boxes, [(0, 200, 255), (0, 255, 0)]):
        cv2.rectangle(preview, (box.x1, box.y1), (box.x2, box.y2), color, 3)
    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)
    if st.button("Test reading the score at this moment"):
        try:
            from broadcast.scoreboard import read_scores
            st.write(read_scores(frame, boxes, get_ocr()))
        except Exception as err:
            show_failure("Reading the score", err)

# ---- 3. Analyze --------------------------------------------------------------------
st.header("3. Analyze")
use_yolo = st.checkbox("Also detect players and ball with YOLO (slower; labels game vs. close-up shots)",
                       value=False)
sample_fps = st.select_slider("Frames checked per second", [1.0, 2.0, 4.0], value=2.0)
if st.button("Run analysis", type="primary"):
    bar = st.progress(0.0, "Analyzing...")
    try:
        result = analyze_broadcast(path, boxes, get_ocr(),
                                   detector=get_detector() if use_yolo else None,
                                   sample_fps=sample_fps,
                                   progress=lambda p: bar.progress(p, "Analyzing..."))
        bar.progress(1.0, "Done")
        st.session_state["result"] = result
    except Exception as err:
        bar.empty()
        show_failure("The analysis", err)

result = st.session_state.get("result")
if result is None:
    st.stop()

# ---- 4. Results --------------------------------------------------------------------
st.header("4. Results")
events = result["events"]
team_stats = result["team_stats"]
st.subheader("Team scoring")
st.dataframe(team_stats, hide_index=True)
if (events["flag"] != "").any():
    st.warning("Some score changes were not +1/+2/+3 (part of the clip missing, a scorer "
               "correction, or a misread). They are marked 'check' and left out of the totals.")

st.subheader("Score over time")
fig, ax = plt.subplots(figsize=(9, 3.2))
samples = result["samples"]
for box in boxes:
    col = "score_" + box.team
    start = result["start_scores"].get(box.team)
    if start is None:
        continue
    mine = events[(events["team"] == box.team) & (events["flag"] == "")]
    times = [0.0] + mine["time_s"].tolist() + [result["meta"]["duration_s"]]
    scores = [start] + mine["score_after"].tolist()
    scores = scores + [scores[-1]]
    ax.step(times, scores, where="post", label=box.team, linewidth=2)
ax.set_xlabel("Video time (s)")
ax.set_ylabel("Score")
ax.grid(alpha=0.3)
ax.legend()
st.pyplot(fig)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Scoring events")
    st.dataframe(events, hide_index=True)
with c2:
    st.subheader("Biggest scoring runs")
    st.dataframe(result["runs"].head(5), hide_index=True)

st.subheader("Camera segments")
st.dataframe(result["segments"], hide_index=True)

st.download_button("Download scoring events CSV", events.to_csv(index=False), "scoring_events.csv")
st.download_button("Download team stats CSV", team_stats.to_csv(index=False), "team_stats.csv")

# ---- 5. Ball tracking video ---------------------------------------------------------
st.header("5. Ball tracking video")
st.write("Draws a circle and trail around the ball (found by YOLO in each frame), with the "
         "score SBVS read in the top bar and a banner when it confirms a basket. "
         "Solid circle = detected in that frame; dashed = short guess between detections.")
v1, v2, v3 = st.columns(3)
with v1:
    clip_start = st.number_input("Start (s)", 0.0, max(0.0, meta["duration_s"] - 1), 0.0, 1.0)
with v2:
    clip_len = st.number_input("Length (s)", 2.0, 60.0, 10.0, 1.0)
with v3:
    quality = st.radio("Detector", ["Fast (yolo11n)", "Accurate (yolo11x, about 15x slower)"])


@st.cache_resource
def get_ball_model(name):
    from ultralytics import YOLO
    return YOLO(name)


if st.button("Make tracking video"):
    from broadcast.ball_track import render_tracking_video
    if quality.startswith("Fast"):
        model_name = LOCAL_YOLO if os.path.exists(LOCAL_YOLO) else "yolo11n.pt"
    else:
        model_name = "yolo11x.pt"
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    vbar = st.progress(0.0, "Tracking the ball...")
    try:
        stats = render_tracking_video(path, out_path, get_ball_model(model_name), clip_start,
                                      clip_len, events=events, start_scores=result["start_scores"],
                                      progress=lambda p: vbar.progress(p, "Tracking the ball..."))
        vbar.progress(1.0, "Done")
        st.session_state["tracking_video"] = out_path
        st.session_state["tracking_stats"] = stats
    except Exception as err:
        vbar.empty()
        show_failure("The tracking video", err)

if st.session_state.get("tracking_video"):
    stats = st.session_state["tracking_stats"]
    st.video(st.session_state["tracking_video"])
    st.caption(f"Ball detected in {stats['detected']} of {stats['frames']} frames "
               f"({100 * stats['detected_share']:.0f}%), guessed in {stats['predicted']}.")
    with open(st.session_state["tracking_video"], "rb") as f:
        st.download_button("Download tracking video", f.read(), "sbvs_ball_tracking.mp4")
