# stats.py
# SBVS 2.0 - shooting statistics with Pandas (owner: Jessica)
#
# Interface (see docs/INTERFACES.md):
#   player_stats(shots_df, players_df) -> DataFrame with columns
#       player_id, name, jersey, attempts, makes, misses, fg_pct
#   team_stats(shots_df) -> dict with keys
#       attempts, makes, misses, fg_pct, unknown, unassigned
#   session_stats(shots_df) -> DataFrame with columns
#       session_id, attempts, makes, misses, fg_pct, unknown
#
# Rules from the interface spec:
#   - Only shots with result 'made' or 'missed' are counted.
#   - 'unknown' shots ("needs review") are left out of FG%.
#   - player_id 0 means the shot has not been assigned to a player yet.
#   - fg_pct is a percent from 0 to 100, rounded to 1 decimal.
#     It is NaN (blank) when a player has 0 attempts, because 0/0 has no answer.

import numpy as np
import pandas as pd

STAT_COLUMNS = ['player_id', 'name', 'jersey',
                'attempts', 'makes', 'misses', 'fg_pct']


def counted_shots(shots_df):
    """Keep only the shots that count: result is 'made' or 'missed'."""
    mask = (shots_df['result'] == 'made') | (shots_df['result'] == 'missed')
    return shots_df[mask]


def player_stats(shots_df, players_df):
    """One row per player on the roster, with attempts, makes, misses, FG%.
    Players with no shots yet still get a row (0 attempts, blank FG%).
    Unassigned shots (player_id 0) are not in this table; see team_stats.
    Rows are sorted by jersey number.
    """
    shots = counted_shots(shots_df).copy()
    # A 1 for every made shot and 0 otherwise, so a sum counts the makes.
    shots['is_made'] = np.where(shots['result'] == 'made', 1, 0)

    # Group by player: count = attempts, sum = makes
    grouped = shots.groupby('player_id')['is_made'].agg(['count', 'sum'])
    grouped = grouped.reset_index()
    grouped.columns = ['player_id', 'attempts', 'makes']

    # Start from the roster so every player appears, then add the counts.
    table = pd.merge(players_df[['player_id', 'name', 'jersey']], grouped,
                     on='player_id', how='left')

    # Players with no shots get NaN from the merge; make those 0.
    table['attempts'] = table['attempts'].fillna(0).astype(int)
    table['makes'] = table['makes'].fillna(0).astype(int)
    table['misses'] = table['attempts'] - table['makes']

    # FG% = makes / attempts * 100.
    # When attempts is 0, Pandas gives 0 / 0 = NaN (blank), which is what we want.
    table['fg_pct'] = (table['makes'] / table['attempts'] * 100).round(1)

    # Sort by jersey number. Jersey is stored as text, and text sorts
    # '10' before '5', so we sort on a temporary number column instead.
    table['jersey_number'] = table['jersey'].astype(int)
    table = table.sort_values('jersey_number')
    return table[STAT_COLUMNS]


def team_stats(shots_df):
    """Totals for the whole team, returned as a dictionary."""
    shots = counted_shots(shots_df)

    attempts = len(shots)
    makes = int((shots['result'] == 'made').sum())
    misses = attempts - makes

    if attempts > 0:
        fg_pct = round(makes / attempts * 100, 1)
    else:
        fg_pct = np.nan

    # 'unknown' shots need a person to review them on the review screen.
    unknown = int((shots_df['result'] == 'unknown').sum())
    # Counted shots that have not been given to a player yet.
    unassigned = int((shots['player_id'] == 0).sum())

    return {'attempts': attempts,
            'makes': makes,
            'misses': misses,
            'fg_pct': fg_pct,
            'unknown': unknown,
            'unassigned': unassigned}


def session_stats(shots_df):
    """One row per session (video clip): attempts, makes, misses, FG%, unknown.
    Every session_id in shots_df gets a row, even one with only 'unknown' shots
    (then attempts is 0 and fg_pct is NaN). Rows are sorted by session_id,
    which is date order because session ids start with YYYY-MM-DD.
    """
    shots = shots_df.copy()
    # Turn each result into 1/0 columns so that adding them up counts them
    counted_mask = (shots['result'] == 'made') | (shots['result'] == 'missed')
    shots['is_counted'] = np.where(counted_mask, 1, 0)
    shots['is_made'] = np.where(shots['result'] == 'made', 1, 0)
    shots['is_unknown'] = np.where(shots['result'] == 'unknown', 1, 0)

    # Add up the 1/0 columns inside each session
    table = shots.groupby('session_id')[['is_counted', 'is_made', 'is_unknown']].agg('sum')
    table = table.reset_index()   # makes session_id a normal column again
    table.columns = ['session_id', 'attempts', 'makes', 'unknown']

    table['misses'] = table['attempts'] - table['makes']
    # 0 / 0 gives NaN (blank) for a session with no counted shots
    table['fg_pct'] = (table['makes'] / table['attempts'] * 100).round(1)

    table = table.sort_values('session_id')
    return table[['session_id', 'attempts', 'makes', 'misses', 'fg_pct', 'unknown']]
