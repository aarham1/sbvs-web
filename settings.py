# settings.py
# Shared constants for SBVS 2.0. Every module imports this file so that
# folder names and CSV column names are typed in exactly one place.
# Owner: Arham. Other members: ask before changing anything here.

# Every video frame is resized to this width (in pixels) before tracking.
# All positions in SBVS 2.0 are measured in this resized "working frame".
WORK_WIDTH = 960

# Folders and files (forward slashes work on Windows and Mac in Python)
DATA_FOLDER = 'data'
VIDEO_FOLDER = 'data/videos'
POSITIONS_FOLDER = 'data/positions'
TRUTH_FOLDER = 'data/ground_truth'
SAMPLE_FOLDER = 'data/sample'
SHOTS_FILE = 'data/shots.csv'
PLAYERS_FILE = 'data/players.csv'

# Column names for every CSV file in the project
POSITION_COLUMNS = ['frame', 'time_s', 'x', 'y', 'r']

SHOT_COLUMNS = ['session_id', 'video_name', 'shot_id', 'frame', 'time_s',
                'result', 'made', 'start_x', 'start_y', 'peak_y',
                'player_id', 'edited']

PLAYER_COLUMNS = ['player_id', 'name', 'jersey']

TRUTH_COLUMNS = ['shot_no', 'time_s', 'result', 'player']

# A detected shot "matches" a real (hand-labeled) shot if their times are
# within this many seconds of each other. Used by shots.evaluate_shots().
MATCH_TOLERANCE_SECONDS = 1.0

# Hand-made sample files (NOT real results) used to build and test screens
SAMPLE_SHOTS_FILE = 'data/sample/shots_sample.csv'
SAMPLE_PLAYERS_FILE = 'data/sample/players_sample.csv'
SAMPLE_POSITIONS_FILE = 'data/sample/positions_sample.csv'

# SBVS 1.0 baseline results, converted to SHOT_COLUMNS (written by sbvs1_baseline.ipynb)
SBVS1_SHOTS_FILE = 'data/sbvs1_shots.csv'
