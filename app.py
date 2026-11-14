"""SBVS - upload game footage, get team shooting stats and a next-game outlook.

Everything is automatic: the system finds the ball, hoop and players, decides makes and
misses, splits teams by jersey color, and forecasts each team's next game from the history
file you keep. Stats are per team (jersey numbers are not readable on TV footage).
See auto/ for how each step works.
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


NEW_NAME = "New team (type a name)"


def pick_team_name(col, color, known, key):
    """Team name for the team in jersey `color`: pick a team from the history file, or type one."""
    label = f"Team in {color.lower()}"
    if known:
        choice = col.selectbox(label, known + [NEW_NAME], index=len(known), key=key + "_sel",
                               help="Pick the same name you used in earlier games so the history lines up.")
        if choice != NEW_NAME:
            return choice
        return col.text_input("Name", color, key=key + "_txt").strip()
    return col.text_input(label, color, key=key + "_txt").strip()


def show_results(history, game_label, game_date):
    result = st.session_state["result"]
    shots, meta = result["shots"], result["meta"]
    colors = list(result["team_names"])

    st.subheader("1. Confirm the team names")
    st.caption("The system tells teams apart by jersey color. Type the real names (e.g. Duke, Alabama). "
               "Use the same names every game, or the next-game outlook will treat one team as two.")
    known = predict.known_teams(history)
    c1, c2 = st.columns(2)
    names = [pick_team_name(c1, colors[0], known, "t0") or colors[0],
             pick_team_name(c2, colors[1], known, "t1") or colors[1]]
    names_ok = names[0] != names[1]
    if not names_ok:
        st.error("The two teams need different names.")
    elif names == colors:
        st.info("Still using jersey colors as names. That works, but a team that wears a different "
                "color next game would start a new history.")

    labeled = predict.label_shots(shots, names)
    counted = predict.counted(labeled)
    teams_df = predict.team_table(labeled, names)

    st.subheader("2. Team stats")
    m = st.columns(4)
    for k in range(2):
        row = teams_df[teams_df["team"] == names[k]].iloc[0]
        m[k].metric(names[k], f"{row['made']}/{row['attempts']}",
                    f"{row['FG%']:.0f}% FG" if row["attempts"] else "no shots", delta_color="off")
    m[2].metric("Shots found", len(counted))
    m[3].metric("Video watched", fmt_time(meta["duration_s"]),
                help=f"Processing took {fmt_time(meta['processing_s'])}, {meta['checked_frames']} frames checked, "
                     f"{meta['cuts']} camera cuts.")

    if len(shots) == 0:
        st.warning("No shots were found. The hoop may not have been visible long enough, or the "
                   "clip is mostly close-ups. Wide game views work best.")
        return
    if meta.get("team_separation") is not None and meta["team_separation"] < 25:
        st.info("The two jersey colors look similar, so team assignment may be unreliable.")

    st.dataframe(teams_df, hide_index=True, use_container_width=True)
    unknown = int((counted["team_name"] == "Unknown team").sum())
    if unknown:
        st.caption(f"{unknown} shot(s) could not be given to a team (shooter not visible); "
                   "they are listed as 'Unknown team'.")

    runs = predict.scoring_runs(labeled)
    if len(runs):
        st.markdown("**Scoring runs** (2+ made baskets in a row with no make by the other team)")
        st.dataframe(runs.assign(start=runs["start_s"].map(fmt_time), end=runs["end_s"].map(fmt_time))
                     [["team", "makes", "start", "end"]], hide_index=True)

    st.markdown("**Shot timeline**")
    chart_df = labeled.assign(minute=labeled["time_s"] / 60,
                              shown=labeled["flag"].where(labeled["flag"] != "", labeled["result"]))
    chart = alt.Chart(chart_df).mark_circle(size=140).encode(
        x=alt.X("minute:Q", title="Minute of video"),
        y=alt.Y("team_name:N", title=None),
        color=alt.Color("shown:N", title="Result",
                        scale=alt.Scale(domain=["made", "missed", "unclear", "possible replay"],
                                        range=["#2ca02c", "#d62728", "#ff9f1a", "#888888"])),
        tooltip=["shot_id", "team_name", "result", "reason", "flag"])
    st.altair_chart(chart, use_container_width=True)

    with st.expander("See every shot (check what the system saw)"):
        thumbs = result["thumbs"]
        cols = st.columns(4)
        for i, row in enumerate(labeled.itertuples()):
            with cols[i % 4]:
                if row.shot_id in thumbs:
                    st.image(thumbs[row.shot_id])
                note = f" · {row.flag}" if row.flag else ""
                st.caption(f"#{row.shot_id} · {fmt_time(row.time_s)} · {row.team_name} · **{row.result}**{note}")

    st.subheader("3. Next-game outlook")
    game_id = st.session_state["result_key"][:10]
    rows = predict.game_rows(labeled, names, game_id, str(game_date), game_label, meta["duration_s"] / 60)
    full = predict.add_game(history, rows)
    out = predict.outlook(full, teams=names)
    st.dataframe(out, hide_index=True, use_container_width=True)
    games = {t: int((full["team"] == t).sum()) for t in names}
    if min(games.values()) < 3:
        st.caption("Low confidence until a team has 3 or more saved games. Download the history file "
                   "below and upload it next time to build it up.")
    st.caption("Counts depend on how much of the game the video covers; FG% does not. "
               "Upload similar-length footage each game for comparable attempts and makes.")
    with st.expander("How the outlook is calculated"):
        st.markdown(
            "- **Expected attempts:** average of the team's past games, recent games count more "
            "(weights 1, 0.6, 0.36, ...).\n"
            f"- **Expected FG%:** the team's own makes/attempts, pulled toward the average of all teams by "
            f"{predict.PRIOR_ATTEMPTS} 'virtual' shots, so one hot game doesn't predict 70%.\n"
            "- **Likely made (80%):** the range the makes should fall in 8 times out of 10 (Poisson model).\n"
            "- **Trend:** last game's FG% compared with the team's overall FG%.")

    st.subheader("4. Downloads")
    d1, d2 = st.columns(2)
    d1.download_button("Download updated history (keep this!)", full.to_csv(index=False),
                       "sbvs_history.csv", "text/csv", type="primary", disabled=not names_ok)
    d2.download_button("Download this game's shots",
                       labeled.drop(columns=["team"]).rename(columns={"team_name": "team"}).to_csv(index=False),
                       f"sbvs_shots_{game_id}.csv", "text/csv")


def main():
    history, minutes = sidebar()
    st.title("🏀 SBVS game stats")
    st.write("Upload game footage. The system watches it, finds every shot, which team took it "
             "and whether it went in, then predicts how each team should shoot next game.")
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
