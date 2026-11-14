# SBVS – Cowork handoff log

Written by Claude (Cowork, cloud session) for Arham and for the Claude in Arham's SBVS project.
The Cowork session has no access to Arham's computer, so this file lives in two places:
- GitHub: `aarham1/sbvs-web` → `HANDOFF.md` (repo root), the copy to read
- Download from the Cowork chat: `HANDOFF.md` plus `sbvs-web-cowork.zip` (full working folder)

Project rules (from Arham, 2026-11-14):
1. Class code (`sbvs-2/`) stays at IME 2130L level (Python, NumPy, Pandas, OOP, Matplotlib, regex, APIs).
   **This session never touched `sbvs-2/`.** It only worked on `sbvs-web/`.
2. Website code (`sbvs-web/`) may go further, but every new library or technique needs a cited source
   in `docs/SOURCES.md`.
3. No fake commits and no commits as teammates (Kevin, Jessica, Jonathan). All commits so far were made
   by Arham's own GitHub account (`aarham1`) through his browser using GitHub's web upload page.
   Most have GitHub's default message "Add files via upload". No git CLI was used and no author
   was faked.

---

## 1. Deploy the existing app (2026-11-12) – DONE

**Asked:** deploy the uploaded `sbvs-web-deploy.zip` to Streamlit Community Cloud following DEPLOY.md,
in a new GitHub repo, then confirm the Demo page video plays. Ask before publishing.

**Done:**
- Arham approved a **public** repo and app (class project, no commercial use). I flagged two things:
  the demo video is a clip of the Duke vs. Alabama TV broadcast, which could draw a copyright
  takedown, and the CSVs contain first names (Jess, Kevin). Both were kept as is.
- Created the public repo **github.com/aarham1/sbvs-web** and uploaded every file from the zip.
  Arham dragged in `demo/duke_alabama_ball_tracking.mp4` himself (17 MB, over the browser
  tool's 10 MB limit).
- Arham signed in to Streamlit with GitHub and authorized it himself.
- Deployed **https://sbvs-web.streamlit.app** (branch `main`, `app.py`, Python 3.12).
- **Verified:** the Demo page video loads and plays (50 s, 1280×960, playback advanced, no errors).

## 2. Turn the site into "upload footage → stats → next-game outlook" (2026-11-12 to 11-14) – MOSTLY DONE

**Asked:** the site should only need game footage. The computer vision finds the hoop, ball,
players and teams on its own. Output: player accuracy, makes vs. attempts, and a prediction for
next game. Decisions Arham made: **TV broadcast footage**, players identified as
**team + jersey number**, game history kept as a **CSV the user downloads and re-uploads**,
and processing on the **free host** (lighter models).

**Done:**
- **Detector:** E-BARD `BODD_yolov8n_0001.onnx` (CC BY 4.0, trained on 60 NBA broadcast games;
  classes basketball, hoop, player, referee). No custom training was needed. It runs with
  `onnxruntime`, so **PyTorch, ultralytics and EasyOCR were removed** from requirements, which
  makes the build much lighter.
  Stored at `models/BODD_yolov8n_0001.onnx`.
- **New package `auto/`:**
  - `detector.py`: ONNX YOLO runner (letterbox, NMS, class roles matched by name).
  - `shots.py`: attempt / made / missed / unclear rules. All tests use the ball's position
    **relative to the hoop in the same frame**, because the TV camera pans and zooms
    mid-shot. Includes a rim-line crossing rule and a "dropped through the net" rule
    for layups.
  - `teams.py`: k-means on chest colors, then a per-team Lab color histogram classifier
    (log-likelihood ratio, so crowd and court colors cancel out); color names for teams.
  - `jersey.py`: Tesseract digit OCR on the shooter's chest, accepted only when 2+ frames agree.
  - `analyze.py`: video loop. Checks 5 frames/s normally and 15 frames/s near the hoop.
    Camera cuts reset tracking. Shooter = the player holding the ball just before the
    attempt. A "possible replay" rule. A thumbnail for every shot.
  - `predict.py`: team and player tables, history CSV, and the outlook. Expected attempts use
    exponential smoothing (decay 0.6). FG% is smoothed toward the league average with 10
    virtual shots. The 80% range uses Poisson quantiles. Confidence is Low/Medium/High by
    number of games.
- **New `app.py` (home page):** upload video → Analyze (progress bar) → editable team names →
  metrics, team table, player table, shot timeline, shot thumbnails, next-game outlook,
  downloads (updated history CSV + this game's shots CSV). Sidebar: upload a past history CSV,
  and a cap on minutes watched (default 10).
- **Old pages moved off the site:** `app.py` → `legacy_fixed_camera_app.py`,
  `pages/2_Broadcast.py` → `legacy/broadcast_page.py`. The Demo page stays.
- `requirements.txt`: streamlit, opencv-python-headless, numpy, pandas, altair, onnxruntime, pytesseract.
  `packages.txt`: libglib2.0-0, tesseract-ocr.
- `docs/SOURCES.md`: new section "Auto-analysis home page" (E-BARD credit and license note,
  make/miss rule, team split, OCR, outlook math). `README.md` updated.
- `tests_auto/test_auto.py`: 12 logic tests with made-up ball paths (no video needed).

**Test results:**
- `python tests_auto/test_auto.py` → **12/12 pass**: make, short miss, long miss, low layup,
  ball held under the hoop is not a make, make while the camera pans, dribble is not a shot,
  rebound bounce is not a second attempt, two shots in a row, ball lost at the rim is unclear,
  team split with crowd pixels, stats and outlook math.
- **Real footage:** the only footage available is the Demo video. It covers source time
  138–187 s of the 4-minute Duke vs. Alabama clip at about real speed, with the old SBVS
  tracking overlays drawn on it. Jess's hand count has 5 shots in that window:

  | Jess (source s) | Truth | SBVS (latest code) | Result | Team |
  |---|---|---|---|---|
  | 141 | Alabama made | 5.6 s → made | ✅ | ✅ Alabama |
  | 152 | Duke made (layup) | 18.6 s → made | ✅ | ❌ (said Alabama; a defender overlaps the shooter) |
  | 167 | Alabama missed | 31.6 s → missed | ✅ | ✅ Alabama |
  | 176 | Duke missed | 40.8 s → missed | ✅ | ✅ Duke |
  | 180 | Duke made 3 | 45.7 s → made | ✅ | ❌ (said Alabama) |

  **Shots found 5/5, make/miss 5/5, team 3/5, jersey numbers 0/5** (none readable).
  Processing: about 32 s for 50 s of video in the cloud sandbox. On the live site a 25 s clip
  took about 40 s.
- **Live site check (2026-11-12):** a 25 s test clip uploaded to sbvs-web.streamlit.app ran end
  to end: metrics, tables, timeline, outlook and download buttons all rendered.

**Bugs found and fixed along the way:**
- Layups that barely rise above the rim in side-on views → added the "dropped through the net" rule.
- The camera panning during a shot made makes look like misses (the hoop position had been
  frozen at shot start) → all rules now use hoop-relative coordinates.
- Threes peak several hoop widths from the rim, and the ball can be hidden for about 1 s
  near the rim → widened the attempt zone (2.5 hoop widths) and the lost-ball timeout (1.2 s).
- Team colors had been learned as averages ("gray" instead of white) → histogram classifier.

## 3. Team assignment when players overlap (2026-11-14) – IMPROVED, needs more footage

**Asked:** fix shots being credited to the wrong team.

**Found:** the color classifier itself is fine. On 30 isolated players from the Demo clip it
was about 28/30 correct; it missed one player turned sideways. The errors came from
(a) picking the wrong shooter: a defender's contesting hand touched the ball last;
(b) identity swaps when the shooter and a defender collide; and
(c) shooters who are never alone during the play.

**Done:**
- New `auto/players.py`: SORT-style player tracker (IoU with constant-velocity prediction)
  plus a jersey-color appearance term (DeepSORT idea, color instead of a CNN). Color is
  ignored during heavy overlap. Clean jersey samples are kept per player, taken only while
  no other box touches him.
- `auto/analyze.py`: the shooter is the player who held the ball in the most frames during
  the 2.5 s before the attempt. His team comes from his tracked clean samples (2 or more);
  otherwise from the chest strip under the ball at release. New `team_from` column in the
  shots CSV (`tracked` / `at release`).
- `docs/SOURCES.md`: cites SORT and DeepSORT, plus the shooter rule.
- `tests_auto/test_auto.py`: 2 new tests (players crossing keep their IDs; samples only when
  alone). **14/14 pass.**

**Result on the Demo window (5 shots):** shots 5/5, make/miss 5/5, **team 4/5** (was 3/5).
While tuning, every variant got 4/5 but missed a *different* shot, so 5 labeled shots are
too few to tune further without overfitting. More hand-counted footage is needed.

## 4. Switch to team-only stats (2026-11-14) – DONE

**Asked:** Arham asked whether jersey numbers are a bigger problem and decided to go team-only.

**Why:** numbers on wide TV shots are about 10-20 px tall, blurred and often turned away.
Tesseract read 0 of 5. The 2025 state of the art needs a fine-tuned model on a GPU.
A misread number also corrupts a player's stats, while team totals hold up.

**Done:**
- `auto/predict.py` rewritten for teams. The history CSV has one row per team per game
  (`game_id, game_date, game_label, team, opponent, minutes, attempts, makes`). Old history
  files with a `player` column still load (their team-total rows are kept). New: scoring runs
  (2+ makes in a row with no make by the other team), `known_teams()`, and a team-only outlook.
- `auto/analyze.py`: the jersey OCR call and the `number` column were removed.
  `auto/jersey.py` is kept but unused.
- `app.py`: steps 1-4 on the page. (1) **Confirm team names**: pick a name already in the
  history file or type one; the download is blocked if both teams have the same name.
  (2) Team stats: made/attempts/FG% per team, an "Unknown team" count, scoring runs,
  timeline, shot thumbnails. (3) Next-game outlook per team. (4) Downloads.
- `requirements.txt`: pytesseract removed. `packages.txt`: tesseract-ocr removed.
- README and `docs/SOURCES.md` updated (why numbers were dropped).
- Tests: **15/15 pass** (team table with the replay excluded and an unknown team, scoring
  runs, history with opponent, loading old-format history). The page code was also
  dry-run with a stubbed Streamlit: no errors, and names matched against the history.

**Demo window result (unchanged detection):** with names Duke / Alabama, SBVS shows
Duke 1/2 and Alabama 2/3. Jess's count is Duke 2/3 and Alabama 1/2. The difference is the
one shot credited to the wrong team (the Duke layup at 2:32), which also creates a false
2-make Alabama run.

## 5. Goal checklist vs. current code (2026-11-14)

Checked against the `sbvs-web` repo as it is now. **`sbvs-2/` (the class submission) has never
been shared with this Cowork session**, so goals 1-6 were judged from the class-era modules
that are copied into `sbvs-web` (`tracker.py`, `shots.py`, `players.py`, `stats.py`,
`charts.py`, `legacy_fixed_camera_app.py`). Check them again in `sbvs-2/`.

| # | Goal | Status | Evidence |
|---|---|---|---|
| 1 | Better ball/shot tracking, measured vs 1.0 | Partial | `shots.evaluate_shots()` and `settings.SBVS1_SHOTS_FILE` exist, but `data/sbvs1_shots.csv` and `sbvs1_baseline.ipynb` are not in this repo and no recorded 1.0-vs-2.0 result was found |
| 2 | Streamlit app UI | Done | `legacy_fixed_camera_app.py` (6 screens: upload, hoop, analyze, review, players, dashboard). It is off the website now but still in the repo |
| 3 | Player roster + manual shot assignment | Done (class app only) | `players.py` (`Player` class, load/save) and `review_screen()` "Review shots and assign players" in `legacy_fixed_camera_app.py` |
| 4 | Stats dashboard with Pandas + Matplotlib (FG%, makes vs attempts, running FG%) | Partial | `stats.py` + `charts.py`. The dashboard shows `fg_by_player_chart` and `attempts_vs_makes_chart`; `charts.running_fg_chart()` exists but is never called by the app |
| 5 | Modular OOP code in separate files | Done | `tracker.py` (`Hoop`, `BasketballTracker`), `shots.py` (`Shot`), `players.py` (`Player`), `stats.py`, `charts.py`, app file |
| 6 | Final submission is a Jupyter notebook | Not done (here) | No `.ipynb` in `sbvs-web`. It may exist in `sbvs-2/`, which this session cannot see |
| 7 | Upload footage incl. TV broadcasts, stats automatic | Done (accuracy only checked on 1 clip) | `app.py` upload → `auto/analyze.py`. Demo window: 5/5 shots, 5/5 make/miss, 4/5 team. Live site verified with a 25 s clip |
| 8 | Team-level stats as main output; per-player as optional manual tagging | Partial | Team-only is done (`predict.team_table`, team-name confirmation in `app.py`). **No optional per-player manual tagging on the website**: the roster/assign screen exists only in the off-site class app |
| 9 | Next-game outlook per team | Done | `predict.outlook()` (exponential smoothing, empirical-Bayes FG%, Poisson 80% range), section 3 of `app.py` |
| 10 | Scoreboard decides makes/points/team; detector adds attempts/misses | Not done | `broadcast/scoreboard.py` (EasyOCR) exists, but `auto/` never uses it; makes come only from the ball/hoop rules. The "4 games" result is not in this session or repo: `data/real_tests/RESULTS.md` covers 1 game (Duke vs Alabama, scoreboard 7/7 makes) |
| 11 | Test on clean raw footage, not the Demo video | Not done | Every accuracy number so far is on the Demo video (old overlays) or a cut of it. Arham's 3 raw clips arrived as 0-byte files |

**Rules**
- Sources: every new technique and library is cited. Altair and onnxruntime links were missing
  and were added to `docs/SOURCES.md` on 2026-11-14.
- Commits: every commit is by `aarham1`, checked through the GitHub API. None are by Kevin,
  Jessica or Jonathan.
- `HANDOFF.md`: updated after every task, including this one.

**Note on goal 10 vs. this session's testing:** the scoreboard-first design fits what is
known. The scoreboard was 7/7 on Duke vs Alabama in `RESULTS.md`, and Arham reports
every make right on 4 games, against about half of makes missed by the automatic
pipeline on new games. Doing it would mean bringing EasyOCR/PyTorch back (heavy on the
free host) or finding a lighter OCR, plus finding the scoreboard automatically, since the
old page needed the score boxes marked by hand.

## Files created or changed in `sbvs-web/` (this session)
New: `app.py` (rewritten), `auto/__init__.py`, `auto/detector.py`, `auto/players.py`, `auto/shots.py`, `auto/teams.py`,
`auto/jersey.py`, `auto/analyze.py`, `auto/predict.py`, `tests_auto/test_auto.py`,
`models/BODD_yolov8n_0001.onnx`, `HANDOFF.md`.
Moved: `app.py` → `legacy_fixed_camera_app.py`; `pages/2_Broadcast.py` → `legacy/broadcast_page.py`.
Changed: `requirements.txt`, `packages.txt`, `README.md`, `docs/SOURCES.md`.
Not changed: everything else from the zip (class-era modules, Demo page, data, DEPLOY.md).

## What's left / blocked
1. **Team assignment** (4/5 after section 3; now the main accuracy limit, since stats are team-only): still fails when the shooter is never alone
   during the play. Next ideas: a possession rule (after a make, the other team usually
   shoots next), or a learned embedding (Roboflow uses SigLIP + k-means; that needs PyTorch
   or an ONNX export). Needs more labeled footage first.
2. **Jersey numbers:** dropped by Arham's choice (team-only). Revisit only with a GPU or a
   small ONNX digit model.
3. **More real footage:** Arham tried to attach 3 broadcast clips from one game (2026-11-14);
   they arrived as 0-byte files (too large for chat). He is re-exporting smaller versions.
   The full 4-minute Duke vs. Alabama clip (`sbvs 2.0 video test 1.mp4`) has not been
   shared with this session. Only the 50 s Demo cut has.
4. Speed on the free host: about 1.5× real time. A full game is not practical there.

## Log
- 2026-11-12: repo created, app deployed, Demo verified.
- 2026-11-12: auto/ package, new app.py, model, docs pushed; live site tested with a 25 s clip;
  histogram team classifier and layup rule pushed.
- 2026-11-14: hoop-relative shot rules, wider three-point zone, longer lost-ball timeout; 12/12 tests; Demo window 5/5 shots, 5/5 make/miss, 3/5 team.
  Pushed to GitHub with this HANDOFF.md (see commit list on GitHub).
- 2026-11-14: player tracking and vote-based shooter (section 3); 14/14 tests; Demo team 4/5. Pushed.
- 2026-11-14: team-only stats, team-name confirmation, team outlook, OCR removed (section 4); 15/15 tests. Pushed.
- 2026-11-14: live site re-checked (25 s clip: 2/2 shots right, both teams right); rendered sbvs_in_action.mp4.
- 2026-11-14: goal checklist (section 5); Altair/onnxruntime citations added. Pushed.
