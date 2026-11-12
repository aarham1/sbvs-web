# players.py
# SBVS 2.0 - player profiles (owner: Jessica)
#
# What this file does:
#   - Defines a Player class (id, name, jersey number).
#   - Loads and saves the team roster to data/players.csv with Pandas.
#   - Checks user input with strings and regex (course Ch8).
#
# Interface (see docs/INTERFACES.md):
#   Player(player_id, name, jersey)   -> .to_dict(), str(player) = '#23 Arham Ahmad'
#   load_players(path)                -> list of Player
#   save_players(player_list, path)   -> writes players.csv
#   is_valid_jersey(text)             -> True/False, jersey must be 1 or 2 digits
#   clean_name(text)                  -> name with extra spaces removed
#   next_player_id(player_list)       -> next free player_id (ids are never reused)
#   players_to_dataframe(player_list) -> DataFrame with PLAYER_COLUMNS

import re
import pandas as pd
import settings


class Player:
    """One player on the roster."""

    def __init__(self, player_id, name, jersey):
        self.player_id = int(player_id)
        self.name = name
        # The jersey is kept as text so '00' and '0' stay different.
        self.jersey = str(jersey)

    def __str__(self):
        # Example: '#23 Arham Ahmad'
        return f"#{self.jersey} {self.name}"

    def to_dict(self):
        # One row of players.csv, as a dictionary.
        return {'player_id': self.player_id,
                'name': self.name,
                'jersey': self.jersey}


def clean_name(text):
    """Remove spaces at the ends and extra spaces in the middle of a name.
    Example: '  Arham    Ahmad ' -> 'Arham Ahmad'
    """
    text = text.strip()
    # \s+ means "one or more spaces/tabs"; replace each run with one space.
    text = re.sub(r'\s+', ' ', text)
    return text


def is_valid_jersey(text):
    """A jersey number must be 1 or 2 digits, for example '5', '23' or '00'."""
    text = text.strip()
    # ^ = start of text, \d{1,2} = one or two digits, $ = end of text
    if re.search(r'^\d{1,2}$', text):
        return True
    else:
        return False


def next_player_id(player_list):
    """Return the next free player_id.
    We always go one higher than the biggest id so a deleted player's id
    is never given to someone new (old shots keep pointing to the right person).
    player_id 0 is reserved for 'unassigned' shots, so the first player is 1.
    """
    biggest = 0
    for player in player_list:
        if player.player_id > biggest:
            biggest = player.player_id
    return biggest + 1


def load_players(path):
    """Read players.csv and return a list of Player objects.
    If the file does not exist yet, return an empty list.
    """
    try:
        # dtype keeps jersey as text, so '00' is not turned into the number 0
        df = pd.read_csv(path, dtype={'jersey': str})
    except FileNotFoundError:
        return []

    player_list = []
    for i in range(len(df)):
        row = df.iloc[i]
        player = Player(row['player_id'], row['name'], row['jersey'])
        player_list.append(player)
    return player_list


def save_players(player_list, path):
    """Write a list of Player objects to players.csv."""
    rows = []
    for player in player_list:
        rows.append(player.to_dict())
    # columns= keeps the right headers even when the list is empty
    df = pd.DataFrame(rows, columns=settings.PLAYER_COLUMNS)
    df.to_csv(path, index=False)


def players_to_dataframe(player_list):
    """Turn a list of Player objects into a DataFrame (used by stats.py)."""
    rows = []
    for player in player_list:
        rows.append(player.to_dict())
    return pd.DataFrame(rows, columns=settings.PLAYER_COLUMNS)
