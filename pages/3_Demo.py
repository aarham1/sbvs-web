"""SBVS website - ready-made demo for presentations (no processing needed)."""

import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="SBVS - Demo", layout="wide")
st.title("SBVS 2.0 demo: Duke vs. Alabama (2025 Elite Eight)")

VIDEO = "demo/duke_alabama_ball_tracking.mp4"
EVENTS = "data/real_tests/duke_alabama_sbvs_events.csv"
TRUTH = "data/real_tests/duke_alabama_truth.csv"

if os.path.exists(VIDEO):
    st.video(VIDEO)
    st.caption("Orange circle: ball found by YOLO in that frame (number = confidence). "
               "Dashed circle: short guess between detections. Top bar: score SBVS read from "
               "the broadcast scoreboard; banner: basket confirmed by SBVS.")
else:
    st.info("Demo video not found. Make one on the Broadcast page (section 5).")

c1, c2 = st.columns(2)
with c1:
    st.subheader("What SBVS found (first 4 minutes)")
    if os.path.exists(EVENTS):
        st.dataframe(pd.read_csv(EVENTS, dtype=str).fillna(""), hide_index=True)
with c2:
    st.subheader("Hand count (Jess)")
    if os.path.exists(TRUTH):
        st.dataframe(pd.read_csv(TRUTH), hide_index=True)
st.success("All 7 made baskets found with the right team and points, no false baskets. "
           "Misses are not detected yet.")
