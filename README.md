# SBVS website (post-class version)

Upload basketball footage in a browser and get analytics back. Built on the IME 2130L
SBVS 2.0 class project (kept unchanged in `../sbvs-2/`). This version is NOT limited to
course-level code. Every new technique is cited in `docs/SOURCES.md`.

## Home page (automatic)
Upload game footage and the system does everything: it finds the ball, hoop and players
(E-BARD YOLOv8n basketball detector, run with onnxruntime), detects shot attempts and
makes/misses, splits teams by jersey color (with player tracking), and shows each team's
made/attempted/FG%, scoring runs and a shot timeline. You confirm the real team names, then
download the history CSV and upload it next time to get a next-game outlook per team.
Stats are team-only: jersey numbers are too small to read reliably on TV footage.
Code: auto/ (see auto/__init__.py). Logic tests: python tests_auto/test_auto.py

Demo page: the Duke vs. Alabama ball-tracking video from SBVS 2.0.

Older pages (not shown on the site): legacy_fixed_camera_app.py (the class version's
step-by-step tracker) and legacy/broadcast_page.py (scoreboard OCR). The Broadcast page
needs torch/ultralytics/easyocr, which were removed from requirements.txt.

Limits: team assignment fails when the shooter is never alone during the play; replays are
filtered with a simple rule; counts depend on how much of the game the video covers.

Run on your laptop
    pip install -r requirements.txt
    streamlit run app.py

## Put it online
See `DEPLOY.md` (Streamlit Community Cloud).

## Layout
    app.py                              home page: upload -> stats -> outlook
    auto/                               automatic analysis (detector, shots, teams, jersey, predict)
    models/BODD_yolov8n_0001.onnx       E-BARD basketball detector (CC BY 4.0)
    tests_auto/                         logic tests for auto/
    pages/3_Demo.py                     Duke vs. Alabama demo page
    tracker.py, shots.py, ...           class-project modules (fixed camera, legacy)
    broadcast/, legacy/                 earlier broadcast pipeline (scoreboard OCR, legacy)
    tests_broadcast/                    fake broadcast clip generator + its known answers
    docs/SOURCES.md                     research and docs behind every new technique
