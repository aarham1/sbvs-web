"""Team stats for one game, a game-history CSV, and a next-game outlook per team.

Team-only on purpose: on TV footage jersey numbers are too small to read reliably, and a
misread number puts a shot on the wrong player. Team totals are much more robust.

History file: one row per team per game. The user downloads it after each game and
uploads it next time (the free host forgets saved files). Team names must be the same
every game (the page asks the user to confirm them) or one team splits into two histories.

Outlook method (simple on purpose, so it can be explained in a class report):
  * Expected attempts = weighted average of past attempts, newest game weighted most
    (weights 1, 0.6, 0.36, ... : exponential smoothing).
  * Expected FG% = the team's weighted makes / attempts, pulled toward the average of
    every team in the history by PRIOR_ATTEMPTS "virtual" shots (empirical Bayes /
    additive smoothing), so one hot game does not predict 70%.
  * Expected makes = expected attempts x expected FG%.
  * Likely range = 10th to 90th percentile of a Poisson distribution around the expected
    makes (makes are counts of independent-ish events).
  * Confidence: low with 1-2 games, medium with 3-5, high with 6 or more.
"""

import math

import pandas as pd

HISTORY_COLUMNS = ["game_id", "game_date", "game_label", "team", "opponent", "minutes",
                   "attempts", "makes"]
DECAY = 0.6
PRIOR_ATTEMPTS = 10
DEFAULT_FG = 0.45


# ---------- stats from one analyzed game ----------------------------------------------

def counted(shots):
    """Shots that count: not flagged as a replay."""
    return shots[shots["flag"] == ""] if len(shots) else shots


def label_shots(shots, team_names):
    df = shots.copy()
    df["team_name"] = df["team"].map(lambda i: team_names[int(i)] if pd.notna(i) else "Unknown team")
    return df


def _fg(makes, attempts):
    return round(100 * makes / attempts, 1) if attempts else None


def team_table(labeled, team_names=None):
    """Made / attempts / FG% per team (both teams listed even with 0 shots)."""
    df = counted(labeled)
    teams = list(team_names or []) + [t for t in pd.unique(df["team_name"]) if t not in (team_names or [])]
    rows = []
    for team in teams:
        g = df[df["team_name"] == team]
        made = int((g["result"] == "made").sum())
        rows.append({"team": team, "made": made, "attempts": len(g),
                     "missed": int((g["result"] == "missed").sum()),
                     "unclear": int((g["result"] == "unclear").sum()), "FG%": _fg(made, len(g))})
    return pd.DataFrame(rows, columns=["team", "made", "attempts", "missed", "unclear", "FG%"])


def scoring_runs(labeled, min_makes=2):
    """Stretches where one team made min_makes or more baskets in a row with no make by
    the other team (misses in between do not end a run)."""
    made = counted(labeled)
    made = made[(made["result"] == "made") & (made["team_name"] != "Unknown team")].sort_values("time_s")
    runs, cur = [], None
    for row in made.itertuples():
        if cur and cur["team"] == row.team_name:
            cur["makes"] += 1
            cur["end_s"] = row.time_s
        else:
            if cur and cur["makes"] >= min_makes:
                runs.append(cur)
            cur = {"team": row.team_name, "makes": 1, "start_s": row.time_s, "end_s": row.time_s}
    if cur and cur["makes"] >= min_makes:
        runs.append(cur)
    return pd.DataFrame(runs, columns=["team", "makes", "start_s", "end_s"])


def game_rows(labeled, team_names, game_id, game_date, game_label, minutes):
    """History rows for this game: one per team."""
    df = counted(labeled)
    rows = []
    for k, team in enumerate(team_names):
        g = df[df["team_name"] == team]
        opponent = team_names[1 - k] if len(team_names) == 2 else ""
        rows.append([game_id, game_date, game_label, team, opponent, round(minutes, 1),
                     len(g), int((g["result"] == "made").sum())])
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


# ---------- history file ---------------------------------------------------------------

def load_history(file):
    """Read a history CSV. Older files (with a 'player' column) are converted: only their
    '(team total)' rows are kept."""
    df = pd.read_csv(file, dtype={"team": str})
    if "player" in df.columns:
        df = df[df["player"].astype(str) == "(team total)"].copy()
    for col, default in (("opponent", ""), ("minutes", float("nan"))):
        if col not in df.columns:
            df[col] = default
    missing = [c for c in HISTORY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("History file is missing columns: " + ", ".join(missing))
    df = df[HISTORY_COLUMNS].copy()
    df["opponent"] = df["opponent"].fillna("").astype(str)
    df["attempts"] = pd.to_numeric(df["attempts"], errors="coerce").fillna(0).astype(int)
    df["makes"] = pd.to_numeric(df["makes"], errors="coerce").fillna(0).astype(int)
    return df


def known_teams(history):
    if history is None or history.empty:
        return []
    return sorted(pd.unique(history["team"].astype(str)))


def add_game(history, new_rows):
    """Append a game (replacing an earlier upload of the same game_id)."""
    if history is None or history.empty:
        return new_rows.copy()
    gid = new_rows["game_id"].iloc[0] if len(new_rows) else None
    history = history[history["game_id"] != gid]
    return pd.concat([history, new_rows], ignore_index=True)


# ---------- outlook --------------------------------------------------------------------

def poisson_quantile(lam, q):
    if lam <= 0:
        return 0
    k, p = 0, math.exp(-lam)
    cdf = p
    while cdf < q and k < 1000:
        k += 1
        p *= lam / k
        cdf += p
    return k


def outlook(history, teams=None):
    """Next-game forecast for each team in the history (or only `teams`)."""
    if history is None or history.empty:
        return pd.DataFrame()
    h = history.copy()
    tot_a, tot_m = h["attempts"].sum(), h["makes"].sum()
    league_fg = tot_m / tot_a if tot_a else DEFAULT_FG
    order = {g: i for i, g in enumerate(pd.unique(h.sort_values(["game_date", "game_id"])["game_id"]))}
    h["order"] = h["game_id"].map(order)
    rows = []
    for team, g in h.groupby("team", sort=False):
        if teams is not None and team not in teams:
            continue
        g = g.sort_values("order")
        n = len(g)
        weights = [DECAY ** (n - 1 - i) for i in range(n)]
        wsum = sum(weights)
        exp_att = sum(w * a for w, a in zip(weights, g["attempts"])) / wsum
        w_m = sum(w * m for w, m in zip(weights, g["makes"]))
        w_a = sum(w * a for w, a in zip(weights, g["attempts"]))
        fg = (w_m + PRIOR_ATTEMPTS * league_fg) / (w_a + PRIOR_ATTEMPTS)
        exp_made = exp_att * fg
        all_a, all_m = g["attempts"].sum(), g["makes"].sum()
        last = g.iloc[-1]
        last_fg = last["makes"] / last["attempts"] if last["attempts"] else None
        all_fg = all_m / all_a if all_a else None
        if last_fg is None or all_fg is None or n < 2:
            trend = "–"
        elif last_fg > all_fg + 0.05:
            trend = "Heating up"
        elif last_fg < all_fg - 0.05:
            trend = "Cooling off"
        else:
            trend = "Steady"
        rows.append({
            "team": team, "games": n, "so far": f"{all_m}/{all_a}",
            "expected made": round(exp_made, 1), "expected attempts": round(exp_att, 1),
            "expected FG%": round(100 * fg, 1),
            "likely made (80%)": f"{poisson_quantile(exp_made, 0.1)}–{poisson_quantile(exp_made, 0.9)}",
            "trend": trend,
            "confidence": "Low" if n <= 2 else ("Medium" if n <= 5 else "High"),
        })
    return pd.DataFrame(rows).sort_values("team").reset_index(drop=True) if rows else pd.DataFrame()
