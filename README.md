# SBVS website (post-class version)

Upload basketball footage in a browser and get analytics back. Built on the IME 2130L
SBVS 2.0 class project (kept unchanged in `../sbvs-2/`). This version is NOT limited to
course-level code. Every new technique is cited in `docs/SOURCES.md`.

## Home page (automatic)
Upload game footage and the system does everything: it finds the ball, hoop and players
(E-BARD YOLOv8n basketball detector, run with onnxruntime), detects shot attempts and
makes/misses, splits teams by jersey color, reads jersey numbers with Tesseract when they
are big enough, and shows team and player made/attempted/FG%. Download the history CSV
after each game and upload it next time to get a next-game outlook per player.
Code: auto/ (see auto/__init__.py). Logic tests: python tests_auto/test_auto.py

Demo page: the Duke vs. Alabama ball-tracking video from SBVS 2.0.

Older pages (not shown on the site): legacy_fixed_camera_app.py (the class version's
step-by-step tracker) and legacy/broadcast_page.py (scoreboard OCR). The Broadcast page
needs torch/ultralytics/easyocr, which were removed from requirements.txt.

Limits: jersey numbers are often unreadable on wide TV shots (those shots count for the
team only); misses are less reliable than makes; replays are filtered with a simple rule.

Run on your laptop
    pip install -r requirements.txt
    streamlit run app.py
(Jersey numbers also need Tesseract installed: `brew install tesseract` / `apt install tesseract-ocr`.)

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
