"""Checks for the auto-analysis logic with made-up ball paths (no video, no model).
Run: python -m pytest tests_auto   (or: python tests_auto/test_auto.py)"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auto import players, predict, shots, teams  # noqa: E402

HOOP = (500, 200, 540, 240, 0.9)   # 40 px wide hoop, rim line at y = 208


def run(points, dt=1 / 15, hoop=HOOP):
    """Feed (x, y) or None points; return finished events."""
    tr = shots.ShotTracker()
    out = []
    t = 0.0
    for p in points:
        ev = tr.update(t, p, [hoop])
        if ev:
            out.append(ev)
        t += dt
    ev = tr.flush(t)
    if ev:
        out.append(ev)
    return out


def arc(start, end, peak_y, n=20):
    xs = np.linspace(start[0], end[0], n)
    ts = np.linspace(0, 1, n)
    # parabola through start, peak, end
    ys = (1 - ts) ** 2 * start[1] + 2 * (1 - ts) * ts * (2 * peak_y - (start[1] + end[1]) / 2) + ts ** 2 * end[1]
    return list(zip(xs, ys))


def test_made_shot():
    path = arc((300, 400), (520, 230), 80)
    ev = run(path)
    assert len(ev) == 1 and ev[0].result == "made", ev


def test_miss_short():
    path = arc((300, 400), (470, 260), 90)
    ev = run(path)
    assert len(ev) == 1 and ev[0].result == "missed", ev


def test_miss_long():
    path = arc((300, 400), (575, 260), 80)
    ev = run(path)
    assert len(ev) == 1 and ev[0].result == "missed", ev


def test_layup_that_barely_rises():
    # side-on view: ball comes in low, peaks just under the rim line, drops through the net
    path = [(380, 330), (430, 290), (480, 250), (505, 215), (515, 222), (520, 232), (522, 260)]
    ev = run(path)
    assert len(ev) == 1 and ev[0].result == "made", ev


def test_ball_held_under_hoop_is_not_a_make():
    path = [(520, 300)] * 10 + [(522, 305)] * 10
    assert run(path) == []


def test_make_while_camera_pans():
    # ball drops straight through the rim, but the camera pans: hoop and ball both move
    # down/right on screen by the same amount each frame
    tr = shots.ShotTracker()
    events = []
    rel = [(-3.0, 1.5), (-2.0, -1.5), (-1.0, -2.5), (-0.3, -1.5), (0.0, -0.6), (0.05, 0.3), (0.05, 1.0)]
    for i, (rx, ry) in enumerate(rel):
        dx, dy = 25 * i, 15 * i          # camera movement
        hoop = (500 + dx, 200 + dy, 540 + dx, 240 + dy, 0.9)
        rim_y = 200 + dy + 0.2 * 40
        ev = tr.update(i / 10, (520 + dx + rx * 40, rim_y + ry * 40), [hoop])
        if ev:
            events.append(ev)
    assert len(events) == 1 and events[0].result == "made", events


def test_dribble_far_away_is_not_a_shot():
    path = [(200 + i, 400 + (i % 2) * 30) for i in range(40)]
    assert run(path) == []


def test_rebound_bounce_does_not_add_attempt():
    shot = arc((300, 400), (575, 240), 80, n=20)
    bounce = [(560, 230), (555, 190), (550, 170), (548, 190), (545, 230), (560, 300)]
    ev = run(shot + bounce)
    assert len(ev) == 1, ev


def test_two_shots():
    a = arc((300, 400), (520, 230), 80)
    gap = [(300, 420)] * 30
    b = arc((700, 400), (522, 230), 70)
    ev = run(a + gap + b)
    assert [e.result for e in ev] == ["made", "made"], ev


def test_ball_lost_at_rim_is_unclear():
    path = arc((300, 400), (520, 190), 80)[:-3] + [None] * 20
    ev = run(path)
    assert len(ev) == 1 and ev[0].result == "unclear", ev


def test_team_split_and_names():
    rng = np.random.default_rng(0)
    white = teams._lab((240, 240, 240)) + rng.normal(0, 3, (30, 3))
    navy = teams._lab((90, 40, 20)) + rng.normal(0, 3, (20, 3))
    # chest pixels: jersey color mixed with the same crowd background for both teams
    crowd = rng.normal(teams._lab((60, 50, 70)), 8, (150, 3))
    def px(base):
        return np.vstack([rng.normal(base, 4, (150, 3)), crowd])
    pixels = [px(c) for c in white] + [px(c) for c in navy]
    model = teams.TeamModel(list(white) + list(navy), pixels)
    assert model.names == ["White", "Navy"], model.names
    # a white shooter whose crop is mostly dark crowd still goes to the white team
    shooter = np.vstack([rng.normal(teams._lab((240, 240, 240)), 4, (40, 3)), rng.normal(teams._lab((60, 50, 70)), 8, (300, 3))])
    assert model.assign_pixels([shooter]) == 0
    assert model.assign_pixels([px(teams._lab((90, 40, 20)))]) == 1


def _frame_with(players_boxes):
    """Gray court frame with solid-color 'players' (box, BGR color)."""
    import numpy as _np
    f = _np.full((400, 600, 3), 150, _np.uint8)
    for (x1, y1, x2, y2), c in players_boxes:
        f[y1:y2, x1:x2] = c
    return f


def test_tracker_keeps_ids_when_players_cross():
    # white player moves right, maroon player moves left; they overlap in the middle frames
    tr = players.PlayerTracker()
    white, maroon = (240, 240, 240), (40, 30, 120)
    seen = []
    for k in range(9):
        a = (100 + 22 * k, 100, 160 + 22 * k, 260)
        b = (400 - 22 * k, 100, 460 - 22 * k, 260)
        # list order flips halfway, like a detector would
        boxes = [a, b] if k < 4 else [b, a]
        f = _frame_with([(a, white), (b, maroon)])
        ids = tr.update(k * 0.2, boxes, f)
        seen.append(dict(zip([tuple(x) for x in boxes], ids))[tuple(a)])
    assert len(set(seen)) == 1, seen


def test_clean_samples_only_when_alone():
    tr = players.PlayerTracker()
    a, b = (100, 100, 160, 260), (150, 100, 210, 260)    # touching
    f = _frame_with([(a, (240, 240, 240)), (b, (40, 30, 120))])
    ids = tr.update(0.0, [a, b], f)
    assert tr.samples(ids[0]) == [] and tr.samples(ids[1]) == []
    far = (400, 100, 460, 260)
    ids = tr.update(0.5, [a, far], _frame_with([(a, (240, 240, 240)), (far, (40, 30, 120))]))
    assert len(tr.samples(ids[0])) == 1


def test_stats_and_outlook():
    shots_df = pd.DataFrame({
        "shot_id": [1, 2, 3, 4, 5], "time_s": [1, 2, 3, 4, 5],
        "result": ["made", "missed", "made", "made", "made"],
        "team": [0, 0, 1, 0, 0], "number": ["23", "23", "", "5", "23"],
        "shooter_found": [True] * 5, "reason": [""] * 5,
        "flag": ["", "", "", "", "possible replay"]})
    lab = predict.label_shots(shots_df, ["White", "Navy"])
    tt = predict.team_table(lab).set_index("team")
    assert tt.loc["White", "made"] == 2 and tt.loc["White", "attempts"] == 3
    pt = predict.player_table(lab).set_index("player")
    assert pt.loc["White #23", "attempts"] == 2 and pt.loc["White #23", "FG%"] == 50.0
    g1 = predict.game_rows(lab, "g1", "2026-11-01", "Game 1")
    g2 = g1.assign(game_id="g2", game_date="2026-11-08")
    hist = predict.add_game(predict.add_game(None, g1), g2)
    hist = predict.add_game(hist, g2)            # re-adding a game replaces it
    assert len(hist) == 2 * len(g1)
    out = predict.outlook(hist).set_index("player")
    row = out.loc["White #23"]
    assert row["games"] == 2 and 0 < row["expected FG%"] < 100
    assert predict.poisson_quantile(3.0, 0.1) == 1 and predict.poisson_quantile(3.0, 0.9) == 5


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("PASS", name)
            except AssertionError as e:
                fails += 1
                print("FAIL", name, e)
    sys.exit(1 if fails else 0)
