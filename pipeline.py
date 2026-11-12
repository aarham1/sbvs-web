# pipeline.py
# SBVS 2.0 - runs the whole analysis for one video (owner: Jonathan)
# Drafted with help from Claude (an AI assistant); Jonathan reviews, runs,
# and commits it himself.
#
# What this file does:
#   1. session_id_from_name() and parse_video_name() read information out of
#      the video's file name with string methods and regex (course Ch8).
#   2. analyze_video() runs Arham's tracker and Kevin's shot rules in order and
#      returns one table of shots. The Streamlit app calls it, and a future
#      backend (for example an iOS app) could call the same function unchanged.
#
# This file does NOT import streamlit, so it also works in a plain notebook.
#
# Interface (see docs/INTERFACES.md):
#   session_id_from_name(file_name) -> '2026-11-04_practice_1'
#   parse_video_name(file_name)     -> {'date', 'kind', 'number'} or None
#   analyze_video(video_path, hoop, session_id, progress_bar=None) -> shots DataFrame
#   rerun_shot_rules(session_id, video_path, hoop) -> shots DataFrame (no tracking)
#   positions_path(session_id)      -> 'data/positions/<session_id>.csv'

import os
import re

import settings   # shared folder names and column lists (Arham's file)
import tracker    # Hoop, video_info, BasketballTracker, save_positions (Arham)
import shots      # find_shots, shots_to_dataframe (Kevin)


# A good video name looks like 2026-11-04_practice_1.mp4
#   ^            start of the name
#   (\d{4})      group 1: year, 4 digits
#   -(\d{2})     group 2: month, 2 digits
#   -(\d{2})     group 3: day, 2 digits
#   _([a-z]+)    group 4: kind, lowercase letters (practice or game)
#   _(\d+)       group 5: clip number, one or more digits
VIDEO_NAME_PATTERN = r'^(\d{4})-(\d{2})-(\d{2})_([a-z]+)_(\d+)'

VIDEO_KINDS = ['practice', 'game']


def file_name_only(path):
    """Return just the file name from a full path.
    Example: 'data/videos/2026-11-04_practice_1.mp4' -> '2026-11-04_practice_1.mp4'
    Windows paths use backslashes, so we turn them into forward slashes first.
    """
    path = path.replace('\\', '/')
    parts = path.split('/')
    return parts[-1]


def session_id_from_name(file_name):
    """The session id is the file name without its extension.
    Example: '2026-11-04_practice_1.mp4' -> '2026-11-04_practice_1'
    """
    name = file_name_only(file_name)
    parts = name.split('.')
    if len(parts) == 1:
        # No dot in the name, so there is no extension to remove
        return name
    # Join everything except the last piece (the extension) back together
    return '.'.join(parts[0:-1])


def parse_video_name(file_name):
    """Read the date, kind and clip number from a video name.
    Returns a dictionary, or None if the name does not follow the pattern.
    Example: '2026-11-04_practice_1.mp4'
             -> {'date': '2026-11-04', 'kind': 'practice', 'number': 1}
    """
    name = file_name_only(file_name)
    match = re.search(VIDEO_NAME_PATTERN, name)
    if match is None:
        return None

    kind = match.group(4)
    if kind in VIDEO_KINDS:
        date = match.group(1) + '-' + match.group(2) + '-' + match.group(3)
        number = int(match.group(5))
        return {'date': date, 'kind': kind, 'number': number}
    else:
        # The shape is right but the kind is something else, like 'scrimmage'
        return None


def positions_path(session_id):
    """Where the ball positions for one session are saved.
    Example: '2026-11-04_practice_1' -> 'data/positions/2026-11-04_practice_1.csv'
    """
    return settings.POSITIONS_FOLDER + '/' + session_id + '.csv'


def analyze_video(video_path, hoop, session_id, progress_bar=None):
    """Run the whole SBVS 2.0 analysis on one video.

    Steps:
      1. Find the ball in every frame (Arham's BasketballTracker).
      2. Save those ball positions to data/positions/<session_id>.csv, so the
         shot rules can be re-run later without tracking the video again.
      3. Find the shots and made/missed results (Kevin's find_shots).
      4. Return the shots as a DataFrame with the shots.csv columns.

    progress_bar can be Streamlit's st.progress bar, or None in a notebook.
    This function does not write shots.csv; the app does that with
    shots.save_session_shots() so the user decides when results are saved.
    """
    info = tracker.video_info(video_path)

    # Step 1: track the ball (the slow part)
    ball_tracker = tracker.BasketballTracker(hoop)
    positions = ball_tracker.process_video(video_path, progress_bar)

    # Step 2: save the positions table
    if not os.path.exists(settings.POSITIONS_FOLDER):
        os.makedirs(settings.POSITIONS_FOLDER)
    tracker.save_positions(positions, positions_path(session_id))

    # Step 3: shot rules (fast)
    shot_list = shots.find_shots(positions, hoop, info['fps'])

    # Step 4: one table, ready for shots.csv
    video_name = file_name_only(video_path)
    shots_df = shots.shots_to_dataframe(shot_list, session_id, video_name)
    return shots_df


def rerun_shot_rules(session_id, video_path, hoop):
    """Run only Kevin's shot rules again, using the ball positions that
    analyze_video already saved. Tracking is the slow part, so this makes it
    quick to tune shots.py: change a number, re-run, compare.

    Like analyze_video, this returns a shots DataFrame and does NOT write
    shots.csv; the caller decides whether to save it.
    The video is only opened to read its fps (frames per second).
    """
    positions = tracker.load_positions(positions_path(session_id))
    fps = tracker.video_info(video_path)['fps']
    shot_list = shots.find_shots(positions, hoop, fps)
    video_name = file_name_only(video_path)
    return shots.shots_to_dataframe(shot_list, session_id, video_name)


def main():
    # Quick check of the name helpers (no video needed)
    test_names = ['2026-11-04_practice_1.mp4',
                  '2026-11-05_game_12.mov',
                  '2026-11-04_scrimmage_1.mp4',
                  'IMG_4412.MOV',
                  'C:\\Users\\kevin\\Videos\\2026-11-06_game_2.mp4']
    for name in test_names:
        print(name, '->', session_id_from_name(name), parse_video_name(name))


# Run main() only when this file is run directly (python pipeline.py), not when another file imports it.
if __name__ == '__main__':
    main()
