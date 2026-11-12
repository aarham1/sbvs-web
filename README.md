# SBVS website (post-class version)

Upload basketball footage in a browser and get analytics back. Built on the IME 2130L
SBVS 2.0 class project (kept unchanged in `../sbvs-2/`). This version is NOT limited to
course-level code. Every new technique is cited in `docs/SOURCES.md`.

## Pages
- **app** (from the class project): fixed-camera footage, so a phone on a tripod. You mark the hoop,
  and it tracks the ball, finds shots, and lets you assign them to players.
- **Broadcast** (new, experimental): TV-style footage with camera cuts and replays. You mark where
  each team's score is on screen. It reads the scoreboard (EasyOCR) to find every made FT/2/3,
  then shows team points, a score-over-time chart, scoring runs and camera segments. Optional
  YOLO detection labels game views vs. close-ups.

## What the Broadcast page can't do yet
- Missed shots and FG% need a basketball model that detects the ball AND the hoop (Phase 2b).
- Player-level stats on broadcasts (Phase 4).
- Accuracy has only been checked on a fake clip (`tests_broadcast/`), not on a real game.

## Run on your laptop
    pip install -r requirements.txt
    streamlit run app.py
Then open the Broadcast page in the sidebar. To check the fake clip:
    python tests_broadcast/make_fake_broadcast.py
Score boxes for it are in `tests_broadcast/fake_score_boxes.csv`.

## Put it online
See `DEPLOY.md` (Streamlit Community Cloud).

## Layout
    app.py, tracker.py, shots.py, ...   class-project modules (fixed camera)
    broadcast/                          broadcast pipeline (cuts, scoreboard OCR, YOLO, analyze)
    pages/2_Broadcast.py                Broadcast web page
    tests_broadcast/                    fake broadcast clip generator + its known answers
    docs/SOURCES.md                     research and docs behind every new technique
