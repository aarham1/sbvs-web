# Sources for the new (non-course) techniques in the SBVS website

The class version (`sbvs-2/`) only used IME 2130L material plus OpenCV and Streamlit.
The website version adds the techniques below. For each one: what it does in SBVS,
where the idea comes from, and the official docs to learn it from.

Links were checked on 2026-11-10. Read the starred (★) items first: they are short and
cover the most important ideas.

---

## 1. Object detection with YOLO (`broadcast/detect.py`)

**What it does in SBVS:** finds people and the ball in each sampled frame. People
count tells live game action (6+ people) from close-ups and crowd shots.

- ★ Redmon, Divvala, Girshick, Farhadi. *You Only Look Once: Unified, Real-Time Object
  Detection.* CVPR 2016. The original YOLO paper: one neural network looks at the whole
  image once and outputs boxes + classes.
  https://arxiv.org/abs/1506.02640 ·
  https://openaccess.thecvf.com/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html
- ★ Ultralytics YOLO docs, Python usage (the library SBVS calls: `YOLO(...)`, `.predict()`).
  https://docs.ultralytics.com/python/ · tracking mode: https://docs.ultralytics.com/modes/track
- Lin et al. *Microsoft COCO: Common Objects in Context.* ECCV 2014. The dataset the
  default YOLO model was trained on. It has "person" and "sports ball" classes but no
  basketball hoop, which is why SBVS can't detect the hoop yet.
  https://arxiv.org/abs/1405.0312
- **License note:** Ultralytics is AGPL-3.0. Free for open-source projects, but the
  license page says a SaaS/web service that uses YOLO must either release its full
  source code or buy an Enterprise license. https://www.ultralytics.com/license
  For a public SBVS site, either keep the GitHub repo public, or switch to a detector
  under a permissive license (for example torchvision's detection models, BSD license).

## 2. Reading the scoreboard with OCR (`broadcast/scoreboard.py`)

**What it does in SBVS:** reads each team's score from a fixed box on screen every
half second. A confirmed change of +1/+2/+3 becomes a made free throw / 2 / 3.

- ★ EasyOCR (the OCR library SBVS uses): https://github.com/JaidedAI/EasyOCR
- GDELT Project, *OCR'ing Television News: Experiments With EasyOCR*: a real-world
  test of EasyOCR on TV broadcast graphics, close to our use case.
  https://blog.gdeltproject.org/ocring-television-news-experiments-with-easyocr/
- ScoreSight: an open-source tool that reads sports scoreboards from video with OCR,
  which shows the "fixed box + OCR" approach is used in practice.
  https://github.com/occ-ai/scoresight
- Comparing EasyOCR with Tesseract in Python (to justify the choice):
  https://www.nutrient.io/blog/easyocr-vs-tesseract-python/
- The "must see the same value 3 samples in a row" rule is the same idea Roboflow
  uses to confirm jersey numbers (they require three matching predictions in a row;
  see section 5).

## 3. Camera-cut detection (`broadcast/cuts.py`)

**What it does in SBVS:** splits the broadcast into camera shots by comparing color
histograms of consecutive frames. A big drop in similarity is a cut.

- ★ Lienhart, R. (1999). *Comparison of Automatic Shot Boundary Detection Algorithms.*
  Found color-histogram differences reliable for hard cuts, but no single threshold fits
  every video type, which is why `CUT_THRESHOLD` must be checked on real footage.
  https://opus.bibliothek.uni-augsburg.de/opus4/files/67685/Lienhart_67685.pdf
- Koprinska, I. & Carrato, S. (2001). *Temporal Video Segmentation: A Survey.*
  https://www.ee.columbia.edu/~sfchang/course/vis/REF/temporal-video-segmentation-a.pdf
- OpenCV histogram docs (`cv2.calcHist`, `cv2.compareHist`):
  https://docs.opencv.org/4.x/d8/dc8/tutorial_histogram_comparison.html
- PySceneDetect: a full library for the same task, which SBVS could switch to later.
  https://www.scenedetect.com/docs/

## 4. Basketball-specific shot detection (Phase 2b, not built yet)

**What it would add:** a model trained on basketball images that detects the ball AND
the hoop, so SBVS can find missed shots, not just made ones.

- ★ Roboflow, *How to Build an AI Basketball Shot Evaluator*: YOLO ball + hoop detection
  with made/missed logic. https://blog.roboflow.com/how-to-build-an-ai-basketball-shot-evaluator/
- avishah3, *AI Basketball Shot Detection Tracker* (YOLOv8, ball + hoop, open source):
  https://github.com/avishah3/AI-Basketball-Shot-Detection-Tracker
- Labeled basketball/hoop datasets for training, e.g.
  https://universe.roboflow.com/amrita-hlhw6/basketball-and-hoop-detection

## 5. Player tracking and teams (Phase 4, not built yet)

- ★ Roboflow, *How to Detect, Track, and Identify Basketball Players with Computer Vision*:
  detector for players and numbers, SAM2 tracking, team split by clustering player
  image features with K-means, jersey numbers by a fine-tuned classifier (93%).
  https://roboflow.com/blog/identify-basketball-players
- Zhang et al. *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.*
  ECCV 2022. The tracker Ultralytics uses by default. https://arxiv.org/abs/2110.06864

## 6. Hosting (Phase 1 and 3)

- Streamlit Community Cloud, deploy and limits (about 2 CPU cores and up to 2.7 GB memory
  at most; apps sleep after 12 hours with no visitors):
  https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app
- Streamlit multipage apps (how `pages/2_Broadcast.py` becomes a second page):
  https://docs.streamlit.io/develop/concepts/multipage-apps/pages-directory

---

## How to show you learned it (not only that you cited it)

Sources show where the ideas come from. What convinces a reviewer is being able to explain:
1. Why the old color + motion tracker fails on broadcasts (moving camera, gold jerseys, wood floor).
2. Why reading the scoreboard is more reliable than tracking the ball through the rim on TV footage.
3. What a confirmed +1/+2/+3 means, and why other jumps are marked "check".
4. What COCO can and cannot detect, and why that blocks miss detection for now.
5. How you measured accuracy: your hand-counted clip vs. SBVS output.

## 7. Ball tracking video (`broadcast/ball_track.py`)

**What it does in SBVS:** circles the ball in every frame for the demo video.

- Detect-then-follow: search the whole frame, then run the detector on a crop around the
  predicted position. Running detectors on enlarged crops/tiles is a standard trick for small
  objects; see SAHI, *Slicing Aided Hyper Inference and Fine-tuning for Small Object
  Detection* (Akyon et al., 2022): https://arxiv.org/abs/2202.06934
- Constant-velocity prediction between detections is the motion model inside the Kalman
  filter used by SORT/ByteTrack (section 5). SORT paper: https://arxiv.org/abs/1602.00763
- Template matching (`cv2.matchTemplate`) was tried for bridging gaps and turned off because
  it drifted onto players: https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
- Measured on the Duke clip (8 s, 266 frames): yolo11n found the ball in about 22% of
  frames and yolo11x in 46%. A basketball-trained model is the next improvement (section 4).

## Auto-analysis home page (auto/)

- **Detector:** E-BARD `BODD_yolov8n_0001.onnx` by Gabriele Giudici, Hugging Face
  `GabrieleGiudici/E-BARD-detection-models`, license CC BY 4.0. A YOLOv8n trained on 60 NBA
  broadcast games (2024-25), classes basketball, hoop, player, referee, input 704x704.
  The file's metadata also carries Ultralytics' AGPL-3.0 notice (YOLOv8 code); this repo is
  public and open source, which AGPL allows. Run with onnxruntime, so PyTorch is not needed.
- **Make/miss rule:** the ball crossing the rim line going down, with the crossing point found by
  linear interpolation between the frames before and after, inside the rim. This is the same idea
  as SBVS 2.0's fixed-camera rule, now using a detected hoop.
- **Team split:** k-means (k = 2) on the median Lab color of each player's chest area. A shooter is
  assigned by per-pixel voting over several frames, so a defender overlapping the box matters less.
  This is the usual unsupervised baseline; Roboflow's 2025 basketball tutorial uses the same
  k-means step on learned (SigLIP) features instead of raw colors.
- **Jersey numbers:** Tesseract OCR (digits only), accepted only when two or more frames agree.
- **Next-game outlook:** exponential smoothing of attempts (decay 0.6). FG% uses additive
  (empirical-Bayes) smoothing toward the overall average with 10 virtual shots. The 80% range
  comes from Poisson quantiles.
