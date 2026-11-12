# shots.py
# SBVS 2.0 - Shot detection and scoring
# Owner: Kevin. Drafted with help from Claude (an AI assistant); Kevin reviews,
# runs, and commits it himself.
#
# What this file does:
#   1. find_shots() walks through the ball positions table (one row per video
#      frame) and decides when a shot starts and whether it was made, missed,
#      or "unknown" (the ball was lost for too long to tell).
#   2. shots_to_dataframe(), save_session_shots(), load_all_shots() turn the
#      shots into a Pandas table and keep them in data/shots.csv.
#   3. load_truth() and evaluate_shots() compare detected shots to a
#      hand-counted ground truth file, so we can measure accuracy (1.0 vs 2.0).
#
# Reminder about the picture: y grows DOWNWARD, so a smaller y means the ball
# is HIGHER in the frame. "Rising" means y is getting smaller.

import numpy as np
import pandas as pd

import settings   # shared column lists and MATCH_TOLERANCE_SECONDS (Arham's file)


# ----- Tuning numbers for the shot rules (starting values, NOT measured yet) -----
RISING_FRAMES = 3            # seen frames in a row the ball must go up to start a shot
SHOT_GAP_SECONDS = 1.0       # ball unseen longer than this closes an open shot as 'unknown'
COOLDOWN_SECONDS = 1.0       # after each result, ignore the ball this long (rim bounces, net)
ZONE_MARGIN = 1.0            # hoop zone = rim box widened by this many rim widths each side
MAX_FLIGHT_SECONDS = 3.0     # a rising ball that never reaches the hoop zone is cancelled
MAX_NEAR_HOOP_SECONDS = 2.0  # a ball that stays near the hoop this long without going in = missed
REQUIRE_PEAK_ABOVE_RIM = True  # True: a rising ball that stops going up before it gets higher
                               # than the rim line is a dribble/pass, not a shot.
                               # Set to False for low camera angles where real shots may
                               # never look higher than the rim.


class Shot:
    """One shot attempt and what happened to it."""

    def __init__(self, shot_id, frame, time_s, result, start_x, start_y, peak_y,
                 player_id=0, edited=False):
        self.shot_id = shot_id      # 1, 2, 3... inside one video
        self.frame = frame          # frame when the ball reached the rim
        self.time_s = time_s        # same moment, in seconds
        self.result = result        # 'made', 'missed', or 'unknown'
        self.start_x = start_x      # where the flight started
        self.start_y = start_y
        self.peak_y = peak_y        # highest point of the arc (smallest y)
        self.player_id = player_id  # 0 = not assigned to a player yet
        self.edited = edited        # True if a person corrected it in the app

    def __str__(self):
        return f"Shot {self.shot_id} at {self.time_s:.1f} s: {self.result}"

    def to_dict(self):
        return {'shot_id': self.shot_id, 'frame': self.frame,
                'time_s': self.time_s, 'result': self.result,
                'start_x': self.start_x, 'start_y': self.start_y,
                'peak_y': self.peak_y, 'player_id': self.player_id,
                'edited': self.edited}


# ---------------- helper functions for find_shots ----------------

def in_hoop_zone(x, y, hoop):
    """True if (x, y) is inside the rim box widened by ZONE_MARGIN rim widths on each side."""
    w = ZONE_MARGIN * hoop.width()
    inside_left_right = (x >= hoop.x1 - w) and (x <= hoop.x2 + w)
    inside_top_bottom = (y >= hoop.y1 - w) and (y <= hoop.y2 + w)
    return inside_left_right and inside_top_bottom


def crossed_rim_inside(prev_x, prev_y, x, y, hoop):
    """True if the ball went DOWN through the rim line between two seen points,
    and at the crossing it was between the left and right edge of the rim.

    We check between two seen points (not just the current frame), so a fast
    ball that jumps over the line in one frame, or during a short tracking
    gap, is still counted. This fixes a 1.0 bug.
    """
    rim = hoop.rim_y()
    if prev_y < rim and y >= rim:
        # Straight line between the two points: how far along is the rim line?
        fraction = (rim - prev_y) / (y - prev_y)
        x_at_rim = prev_x + fraction * (x - prev_x)
        return hoop.inside_x(x_at_rim)
    return False


def distance_to_hoop(x, y, hoop):
    cx, cy = hoop.center()
    return np.sqrt((x - cx) ** 2 + (y - cy) ** 2)


# ---------------- main shot rules ----------------

def find_shots(positions, hoop, fps):
    """Walk the positions table frame by frame and return a list of Shot objects.

    positions: DataFrame with columns frame, time_s, x, y, r (x, y are NaN when
               the ball was not seen)
    hoop:      Hoop object (from tracker.py)
    fps:       frames per second of the video (kept for the interface; times
               come from the time_s column)

    The rules use a "state" string, one shot open at a time:
      'waiting'   -> ball goes up for RISING_FRAMES frames  -> 'rising'
      'rising'    -> ball enters the hoop zone                  -> 'near_hoop'
      'rising'    -> ball stops going up while still lower than
                     the rim (dribble, pass), or too slow       -> cancelled, 'waiting'
      'near_hoop' -> ball goes down through the rim, inside it  -> made, 'cooldown'
      'near_hoop' -> ball drops below the rim outside the zone,
                     or hangs around too long                   -> missed, 'cooldown'
      open shot   -> ball unseen longer than SHOT_GAP_SECONDS   -> unknown, 'cooldown'
      'cooldown'  -> COOLDOWN_SECONDS pass                      -> 'waiting'
    The cooldown stops rim bounces and the ball falling out of the net from
    being counted again (1.0 double-counted these).
    """
    frames = positions['frame'].values
    times = positions['time_s'].values
    xs = positions['x'].values
    ys = positions['y'].values

    shot_list = []
    state = 'waiting'

    # Last point where the ball was actually seen
    prev_x = np.nan
    prev_y = np.nan
    prev_t = np.nan
    prev_frame = -1

    rising_count = 0
    start_x = np.nan       # where the current flight started
    start_y = np.nan
    start_t = 0.0
    peak_y = np.nan
    near_start_t = 0.0     # when the ball entered the hoop zone
    best_dist = 0.0        # closest distance to the hoop so far
    best_frame = 0         # frame / time of that closest point
    best_t = 0.0
    cooldown_start = 0.0

    for i in range(len(frames)):
        t = times[i]
        x = xs[i]
        y = ys[i]
        ball_seen = not (np.isnan(x) or np.isnan(y))

        # ---- 1. cooldown: ignore everything until it is over ----
        if state == 'cooldown':
            if t - cooldown_start >= COOLDOWN_SECONDS:
                state = 'waiting'
                rising_count = 0
            if ball_seen:
                prev_x, prev_y, prev_t, prev_frame = x, y, t, frames[i]
            continue

        # ---- 2. ball not seen in this frame ----
        if not ball_seen:
            shot_open = (state == 'rising') or (state == 'near_hoop')
            if shot_open and t - prev_t > SHOT_GAP_SECONDS:
                # Lost the ball too long to judge. This is NOT a miss.
                if state == 'near_hoop':
                    shot_frame, shot_t = best_frame, best_t
                else:
                    shot_frame, shot_t = prev_frame, prev_t
                new_shot = Shot(len(shot_list) + 1, int(shot_frame), float(shot_t),
                                'unknown', start_x, start_y, peak_y)
                shot_list.append(new_shot)
                state = 'cooldown'
                cooldown_start = t
            continue   # short gaps are simply skipped

        # From here on the ball is seen in this frame.
        have_prev = not np.isnan(prev_y)
        if have_prev and t - prev_t > SHOT_GAP_SECONDS:
            have_prev = False   # last sighting is too old to compare with

        # ---- 3. waiting: look for the ball going up ----
        if state == 'waiting':
            if have_prev and y < prev_y:
                rising_count = rising_count + 1
                if rising_count == 1:
                    start_x, start_y, start_t = prev_x, prev_y, prev_t
            else:
                rising_count = 0
            if rising_count >= RISING_FRAMES:
                state = 'rising'
                peak_y = y

        # ---- 4. rising: is it heading for the hoop? ----
        elif state == 'rising':
            if y < peak_y:
                peak_y = y
            if have_prev and crossed_rim_inside(prev_x, prev_y, x, y, hoop):
                # Ball went straight from far away to through the rim
                new_shot = Shot(len(shot_list) + 1, int(frames[i]), float(t),
                                'made', start_x, start_y, peak_y)
                shot_list.append(new_shot)
                state = 'cooldown'
                cooldown_start = t
            elif in_hoop_zone(x, y, hoop):
                state = 'near_hoop'
                near_start_t = t
                best_dist = distance_to_hoop(x, y, hoop)
                best_frame, best_t = frames[i], t
            else:
                # Cancel quietly (not a shot) if the ball stopped going up
                # before it ever got higher than the rim line (a dribble, a
                # pass, or a ball being carried), or if it took too long.
                stopped_low = False
                if REQUIRE_PEAK_ABOVE_RIM:
                    stopped_low = (y >= prev_y) and (peak_y > hoop.rim_y())
                if stopped_low or t - start_t > MAX_FLIGHT_SECONDS:
                    state = 'waiting'
                    rising_count = 0

        # ---- 5. near the hoop: made or missed? ----
        elif state == 'near_hoop':
            if y < peak_y:
                peak_y = y
            d = distance_to_hoop(x, y, hoop)
            if d < best_dist:
                best_dist = d
                best_frame, best_t = frames[i], t

            if have_prev and crossed_rim_inside(prev_x, prev_y, x, y, hoop):
                new_shot = Shot(len(shot_list) + 1, int(frames[i]), float(t),
                                'made', start_x, start_y, peak_y)
                shot_list.append(new_shot)
                state = 'cooldown'
                cooldown_start = t
            else:
                below_rim = y > hoop.rim_y()
                left_zone = not in_hoop_zone(x, y, hoop)
                too_long = t - near_start_t > MAX_NEAR_HOOP_SECONDS
                if (left_zone and below_rim) or too_long:
                    new_shot = Shot(len(shot_list) + 1, int(best_frame), float(best_t),
                                    'missed', start_x, start_y, peak_y)
                    shot_list.append(new_shot)
                    state = 'cooldown'
                    cooldown_start = t

        prev_x, prev_y, prev_t, prev_frame = x, y, t, frames[i]

    # ---- 6. video ended while a shot was still open ----
    if state == 'near_hoop' or state == 'rising':
        if state == 'near_hoop':
            shot_frame, shot_t = best_frame, best_t
        else:
            shot_frame, shot_t = prev_frame, prev_t
        new_shot = Shot(len(shot_list) + 1, int(shot_frame), float(shot_t),
                        'unknown', start_x, start_y, peak_y)
        shot_list.append(new_shot)

    return shot_list


# ---------------- shots table and CSV storage ----------------

def shots_to_dataframe(shot_list, session_id, video_name):
    """Turn a list of Shot objects into a DataFrame with the shots.csv columns."""
    rows = []
    for shot in shot_list:
        row = shot.to_dict()
        row['session_id'] = session_id
        row['video_name'] = video_name
        # made = 1 only for a made shot (easy to add up later)
        if shot.result == 'made':
            row['made'] = 1
        else:
            row['made'] = 0
        rows.append(row)
    if len(rows) == 0:
        return pd.DataFrame(columns=settings.SHOT_COLUMNS)
    shots_df = pd.DataFrame(rows)
    return shots_df[settings.SHOT_COLUMNS]   # put the columns in the agreed order


def load_all_shots(path):
    """Read shots.csv. If the file does not exist yet, return an empty table."""
    try:
        shots_df = pd.read_csv(path)
    except FileNotFoundError:
        shots_df = pd.DataFrame(columns=settings.SHOT_COLUMNS)
    return shots_df[settings.SHOT_COLUMNS]


def save_session_shots(shots_df, path):
    """Save one session's shots into shots.csv.

    Old rows for the same session are removed first, so running the same
    video twice does NOT duplicate rows (1.0 appended to its CSV every run).
    """
    all_shots = load_all_shots(path)
    sessions_here = np.unique(shots_df['session_id'].values)
    for session_id in sessions_here:
        all_shots = all_shots[all_shots['session_id'] != session_id]
    if len(all_shots) == 0:
        combined = shots_df
    else:
        # pd.concat stacks two tables with the same columns on top of each other
        combined = pd.concat([all_shots, shots_df])
    combined.to_csv(path, index=False)


# ---------------- accuracy check against ground truth ----------------

def load_truth(path):
    """Read a hand-labeled ground truth CSV (shot_no, time_s, result, player)."""
    truth_df = pd.read_csv(path)
    return truth_df[settings.TRUTH_COLUMNS]


def evaluate_shots(shots_df, truth_df, tolerance_s=settings.MATCH_TOLERANCE_SECONDS):
    """Compare detected shots (one session) to hand-counted ground truth.

    Matching: go through the detected shots in time order. Each one matches
    the nearest real shot that is still unmatched and within tolerance_s
    seconds. Each real shot can be matched only once.

    A detection that matches nothing is a false attempt. If it is within
    tolerance of a real shot that was already matched, it is ALSO counted as
    a duplicate (the same shot counted twice).

    Returns a dict of the Step 5 metrics. Percentages are 0-100, NaN when
    there is nothing to divide by.
    """
    shots_sorted = shots_df.sort_values('time_s')
    det_times = shots_sorted['time_s'].values
    det_results = shots_sorted['result'].values
    real_times = truth_df['time_s'].values
    real_results = truth_df['result'].values

    real_used = np.zeros(len(real_times), dtype=bool)
    matched = 0
    correct = 0
    duplicates = 0
    false_attempts = 0

    for j in range(len(det_times)):
        best_r = -1
        best_gap = tolerance_s
        for r in range(len(real_times)):
            gap = abs(det_times[j] - real_times[r])
            if not real_used[r] and gap <= best_gap:
                best_gap = gap
                best_r = r
        if best_r >= 0:
            real_used[best_r] = True
            matched = matched + 1
            # 'unknown' never equals 'made' or 'missed', so it counts as wrong
            if det_results[j] == real_results[best_r]:
                correct = correct + 1
        else:
            false_attempts = false_attempts + 1
            near_real = np.abs(real_times - det_times[j]) <= tolerance_s
            if near_real.sum() > 0:
                duplicates = duplicates + 1

    n_real = len(real_times)
    n_det = len(det_times)

    attempts_found_pct = np.nan
    if n_real > 0:
        attempts_found_pct = 100 * matched / n_real
    false_attempts_pct = np.nan
    if n_det > 0:
        false_attempts_pct = 100 * false_attempts / n_det
    outcome_accuracy_pct = np.nan
    if matched > 0:
        outcome_accuracy_pct = 100 * correct / matched

    real_makes = (real_results == 'made').sum()
    real_counted = ((real_results == 'made') | (real_results == 'missed')).sum()
    det_makes = (det_results == 'made').sum()
    det_counted = ((det_results == 'made') | (det_results == 'missed')).sum()

    real_fg = np.nan
    if real_counted > 0:
        real_fg = 100 * real_makes / real_counted
    det_fg = np.nan
    if det_counted > 0:
        det_fg = 100 * det_makes / det_counted

    return {'real_shots': n_real,
            'detected_shots': n_det,
            'matched': matched,
            'duplicates': duplicates,
            'false_attempts': false_attempts,
            'attempts_found_pct': attempts_found_pct,
            'false_attempts_pct': false_attempts_pct,
            'outcome_accuracy_pct': outcome_accuracy_pct,
            'makes_error': int(det_makes - real_makes),
            'fg_error_pts': det_fg - real_fg}
