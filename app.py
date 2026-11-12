"""SBVS - upload game footage, get shooting stats and a next-game outlook.

Everything is automatic: the system finds the ball, hoop and players, decides makes and
misses, splits teams by jersey color, reads jersey numbers when it can, and forecasts
next game from the history file you keep. See auto/ for how each step works.
"""

import datetime as dt
import hashlib
import os
import tempfile

import altair as alt
import pandas as pd
import streamlit as st

from auto import predict
from auto.detector import MODEL_PATH

st.set_page_config(page_title="SBVS - Game Stats", page_icon="🏀", layout="wide")

DEFAULT_MINUTES = 10


@st.cache_resource(show_spinner="Loading the basketball detector...")
def get_detector():
    from auto.detector import Detector
    return Detector()


def fmt_time(s):
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}"


def sidebar():
    st.sidebar.title("SBVS")
    st.sidebar.subheader("Past games (optional)")
    st.sidebar.caption("Upload the history file you downloaded last time to get a better "
                       "next-game outlook. The site does not keep files between visits.")
    hist_file = st.sidebar.file_uploader("History CSV", type=["csv"], key="history_upload")
    history = None
    if hist_file is not None:
        try:
            history = predict.load_history(hist_file)
            st.sidebar.success(f"{history['game_id'].nunique()} past game(s) loaded")
        except Exception as exc:
            st.sidebar.error(f"Could not read that file: {exc}")
    st.sidebar.subheader("Settings")
    minutes = st.sidebar.number_input(
        "Watch at most this many minutes", 1, 60, DEFAULT_MINUTES,
        help="The free host is slow: roughly 1 to 2 minutes of processing per minute of video.")
    return history, minutes


def run_analysis(upload, minutes):
    from auto.analyze import analyze_video
    data = upload.getvalue()
    key = hashlib.md5(data).hexdigest() + f"-{minutes}"
    if st.session_state.get("result_key") == key:
        return
    suffix = os.path.splitext(upload.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        path = f.name
    bar = st.progress(0.0, "Starting...")
    try:
        result = analyze_video(path, detector=get_detector(), max_seconds=minutes * 60,
                               progress=lambda p, msg: bar.progress(p, msg))
    finally:
        os.remove(path)
    bar.empty()
    st.session_state["result"] = result
    st.session_state["result_key"] = key
    st.session_state["team_names"] = list(result["team_names"])


def show_results(history, game_label, game_date):
    result = st.session_state["result"]
    shots, meta = result["shots"], result["meta"]

    st.subheader("Team names")
    st.caption("Teams were named by jersey color. Rename them if you like (e.g. White -> Duke).")
    c1, c2 = st.columns(2)
    names = st.session_state["team_names"]
    names = [c1.text_input("Team 1", names[0], key="tn0"), c2.text_input("Team 2", names[1], key="tn1")]

    labeled = predict.label_shots(shots, names)
    counted = predict.counted(labeled)
    made = int((counted["result"] == "made").sum())
    m = st.columns(4)
    m[0].metric("Shots found", len(counted))
    m[1].metric("Made", made)
    m[2].metric("FG%", f"{100 * made / len(counted):.0f}%" if len(counted) else "–")
    m[3].metric("Video watched", fmt_time(meta["duration_s"]),
                help=f"Processing took {fmt_time(meta['processing_s'])}, {meta['checked_frames']} frames checked, "
                     f"{meta['cuts']} camera cuts.")

    if len(shots) == 0:
        st.warning("No shots were found. The hoop may not have been visible long enough, or the "
                   "clip is mostly close-ups. Wide game views work best.")
        return
    if meta.get("team_separation") is not None and meta["team_separation"] < 25:
        st.info("The two jersey colors look similar, so team assignment may be unreliable.")
    if not meta.get("ocr"):
        st.info("Jersey number reading is not available on this server, so stats are per team only.")

    left, right = st.columns([2, 3])
    with left:
        st.subheader("Teams")
        st.dataframe(predict.team_table(labeled), hide_index=True, use_container_width=True)
    with right:
        st.subheader("Players")
        st.caption("Shots where the number could not be read count under '(number not read)'.")
        st.dataframe(predict.player_table(labeled), hide_index=True, use_container_width=True)

    st.subheader("Shot timeline")
    chart_df = labeled.assign(minute=labeled["time_s"] / 60,
                              shown=labeled["flag"].where(labeled["flag"] != "", labeled["result"]))
    chart = alt.Chart(chart_df).mark_circle(size=140).encode(
        x=alt.X("minute:Q", title="Minute of video"),
        y=alt.Y("team_name:N", title=None),
        color=alt.Color("shown:N", title="Result",
                        scale=alt.Scale(domain=["made", "missed", "unclear", "possible replay"],
                                        range=["#2ca02c", "#d62728", "#ff9f1a", "#888888"])),
        tooltip=["shot_id", "player", "result", "reason", "flag"])
    st.altair_chart(chart, use_container_width=True)

    with st.expander("See every shot (check what the system saw)"):
        thumbs = result["thumbs"]
        cols = st.columns(4)
        for i, row in enumerate(labeled.itertuples()):
            with cols[i % 4]:
                if row.shot_id in thumbs:
                    st.image(thumbs[row.shot_id])
                note = f" · {row.flag}" if row.flag else ""
                st.caption(f"#{row.shot_id} · {fmt_time(row.time_s)} · {row.player} · **{row.result}**{note}")

    st.subheader("Next-game outlook")
    game_id = st.session_state["result_key"][:10]
    rows = predict.game_rows(labeled, game_id, str(game_date), game_label)
    full = predict.add_game(history, rows)
    out = predict.outlook(full)
    n_games = full["game_id"].nunique()
    if n_games < 3:
        st.caption(f"Based on {n_games} game(s). Forecasts get more reliable after 3 or more games: "
                   "download the history file below and upload it next time.")
    st.dataframe(out, hide_index=True, use_container_width=True)
    with st.expander("How the outlook is calculated"):
        st.markdown(
            "- **Expected attempts:** average of past games, recent games count more (weights 1, 0.6, 0.36, ...).\n"
            f"- **Expected FG%:** the player's own makes/attempts, pulled toward the overall average by "
            f"{predict.PRIOR_ATTEMPTS} 'virtual' shots, so a 2-for-2 game doesn't predict 100%.\n"
            "- **Likely made (80%):** the range the makes should fall in 8 times out of 10 (Poisson model).\n"
            "- **Trend:** last game's FG% compared with the player's overall FG%.")

    st.subheader("Downloads")
    d1, d2 = st.columns(2)
    d1.download_button("Download updated history (keep this!)", full.to_csv(index=False),
                       "sbvs_history.csv", "text/csv", type="primary")
    d2.download_button("Download this game's shots", labeled.drop(columns=["team"]).to_csv(index=False),
                       f"sbvs_shots_{game_id}.csv", "text/csv")


def main():
    history, minutes = sidebar()
    st.title("🏀 SBVS game stats")
    st.write("Upload game footage. The system watches it and finds every shot, who took it, "
             "and whether it went in, then predicts how each player should do next game.")
    if not os.path.exists(MODEL_PATH):
        st.error("The basketball detector model file is missing (models/BODD_yolov8n_0001.onnx).")
        return

    upload = st.file_uploader("Game video", type=["mp4", "mov", "m4v", "mkv", "avi"])
    c1, c2 = st.columns([3, 1])
    game_label = c1.text_input("Game name", os.path.splitext(upload.name)[0] if upload else "")
    game_date = c2.date_input("Game date", dt.date.today())
    st.caption("Works best on TV-style or wide footage where the hoop is visible. "
               "Clips of a few minutes are best on the free host.")

    if upload is not None and st.button("Analyze", type="primary"):
        try:
            run_analysis(upload, minutes)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            return
    if "result" in st.session_state:
        show_results(history, game_label or "Game", game_date)


main()
