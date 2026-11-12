"""Game stats, a game-history CSV, and a next-game outlook per player and team.

History file: one row per player (or team total) per game. The user downloads it after
each game and uploads it next time (the free host forgets saved files).

Outlook method (simple on purpose, so it can be explained in a class report):
  * Expected attempts = weighted average of past attempts, newest game weighted most
    (weights 1, 0.6, 0.36, ... : exponential smoothing).
  * Expected FG% = the player's weighted makes / attempts, pulled toward the average of
    everyone in the history by PRIOR_ATTEMPTS "virtual" shots (empirical Bayes / additive
    smoothing). A player with 2 shots does not get a 100% or 0% forecast.
  * Expected makes = expected attempts x expected FG%.
  * Likely range = 10th to 90th percentile of a Poisson distribution around the expected
    makes (shot makes are counts of rare-ish events).
  * Confidence: low with 1-2 games, medium with 3-5, high with 6 or more.
"""

import math

import pandas as pd

HISTORY_COLUMNS = ["game_id", "game_date", "game_label", "team", "player", "attempts", "makes"]
TEAM_TOTAL = "(team total)"
DECAY = 0.6
PRIOR_ATTEMPTS = 10
DEFAULT_FG = 0.45


# ---------- stats from one analyzed game ----------------------------------------------

def counted(shots):
    """Shots that count: not flagged as a replay."""
    return shots[shots["flag"] == ""] if len(shots) else shots


def player_label(team_name, number):
    return f"{team_name} #{number}" if number else f"{team_name} (number not read)"


def label_shots(shots, team_names):
    df = shots.copy()
    df["team_name"] = df["team"].map(lambda i: team_names[int(i)] if pd.notna(i) else "Unknown team")
    df["player"] = [player_label(t, n) for t, n in zip(df["team_name"], df["number"])]
    return df


def _fg(makes, attempts):
    return round(100 * makes / attempts, 1) if attempts else None


def team_table(labeled):
    df = counted(labeled)
    rows = []
    for team, g in df.groupby("team_name", sort=False):
        made = int((g["result"] == "made").sum())
        rows.append({"team": team, "made": made, "attempts": len(g),
                     "missed": int((g["result"] == "missed").sum()),
                     "unclear": int((g["result"] == "unclear").sum()), "FG%": _fg(made, len(g))})
    return pd.DataFrame(rows, columns=["team", "made", "attempts", "missed", "unclear", "FG%"])


def player_table(labeled):
    df = counted(labeled)
    rows = []
    for (team, player), g in df.groupby(["team_name", "player"], sort=False):
        made = int((g["result"] == "made").sum())
        rows.append({"team": team, "player": player, "made": made, "attempts": len(g),
                     "FG%": _fg(made, len(g))})
    out = pd.DataFrame(rows, columns=["team", "player", "made", "attempts", "FG%"])
    return out.sort_values(["team", "attempts"], ascending=[True, False]).reset_index(drop=True)


def game_rows(labeled, game_id, game_date, game_label):
    """History rows for this game: one per identified player plus one team total."""
    df = counted(labeled)
    rows = []
    for team, g in df.groupby("team_name", sort=False):
        rows.append([game_id, game_date, game_label, team, TEAM_TOTAL, len(g), int((g["result"] == "made").sum())])
        for player, p in g[g["number"] != ""].groupby("player"):
            rows.append([game_id, game_date, game_label, team, player, len(p), int((p["result"] == "made").sum())])
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


# ---------- history file ---------------------------------------------------------------

def load_history(file):
    df = pd.read_csv(file, dtype={"player": str, "team": str})
    missing = [c for c in HISTORY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("History file is missing columns: " + ", ".join(missing))
    df = df[HISTORY_COLUMNS].copy()
    df["attempts"] = pd.to_numeric(df["attempts"], errors="coerce").fillna(0).astype(int)
    df["makes"] = pd.to_numeric(df["makes"], errors="coerce").fillna(0).astype(int)
    return df


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


def outlook(history):
    """Next-game forecast for every team total and player in the history."""
    if history is None or history.empty:
        return pd.DataFrame()
    h = history.copy()
    tot_a, tot_m = h["attempts"].sum(), h["makes"].sum()
    league_fg = tot_m / tot_a if tot_a else DEFAULT_FG
    order = {g: i for i, g in enumerate(pd.unique(h.sort_values(["game_date", "game_id"])["game_id"]))}
    h["order"] = h["game_id"].map(order)
    rows = []
    for (team, player), g in h.groupby(["team", "player"], sort=False):
        g = g.sort_values("order")
        n = len(g)
        weights = [DECAY ** (n - 1 - i) for i in range(n)]
        wsum = sum(weights)
        exp_att = sum(w * a for w, a in zip(weights, g["attempts"])) / wsum
        w_m = sum(w * m for w, m in zip(weights, g["makes"]))
        w_a = sum(w * a for w, a in zip(weights, g["attempts"]))
        fg = (w_m + PRIOR_ATTEMPTS * league_fg) / (w_a + PRIOR_ATTEMPTS)
        exp_made = exp_att * fg
        career_a, career_m = g["attempts"].sum(), g["makes"].sum()
        last = g.iloc[-1]
        last_fg = last["makes"] / last["attempts"] if last["attempts"] else None
        career_fg = career_m / career_a if career_a else None
        if last_fg is None or career_fg is None or n < 2:
            trend = "–"
        elif last_fg > career_fg + 0.05:
            trend = "Heating up"
        elif last_fg < career_fg - 0.05:
            trend = "Cooling off"
        else:
            trend = "Steady"
        rows.append({
            "team": team, "player": player, "games": n,
            "career": f"{career_m}/{career_a}",
            "expected made": round(exp_made, 1), "expected attempts": round(exp_att, 1),
            "expected FG%": round(100 * fg, 1),
            "likely made (80%)": f"{poisson_quantile(exp_made, 0.1)}–{poisson_quantile(exp_made, 0.9)}",
            "trend": trend,
            "confidence": "Low" if n <= 2 else ("Medium" if n <= 5 else "High"),
        })
    out = pd.DataFrame(rows)
    out["_t"] = out["player"] != TEAM_TOTAL
    return out.sort_values(["team", "_t", "expected attempts"], ascending=[True, True, False]) \
              .drop(columns="_t").reset_index(drop=True)
