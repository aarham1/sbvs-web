"""Object detection with a pretrained YOLO model (Ultralytics).

The default model is trained on the COCO dataset, which includes "person" (class 0)
and "sports ball" (class 32) but NOT basketball hoops. We use it to:
  - count people, to tell live game action (many players) from close-ups and crowd shots
  - find the ball when it is visible
A basketball-specific model (ball + hoop) can be plugged in later via MODEL_PATH to
also detect shot attempts and misses. See docs/SOURCES.md, "Object detection".
"""

PERSON = 0
SPORTS_BALL = 32
DEFAULT_MODEL = "yolo11n.pt"   # smallest/fastest COCO model; downloads on first use
MIN_PLAYERS_FOR_GAME_VIEW = 6  # wide court shots usually show 6+ people


class Detector:
    def __init__(self, model_path=DEFAULT_MODEL, conf=0.3):
        from ultralytics import YOLO  # heavy import, kept local
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        """Return {'people': n, 'ball': (x, y, r) or None} for one BGR frame."""
        result = self.model.predict(frame, conf=self.conf, classes=[PERSON, SPORTS_BALL],
                                    verbose=False)[0]
        people = 0
        best_ball = None
        best_conf = 0.0
        for box, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                  result.boxes.conf.tolist()):
            if int(cls) == PERSON:
                people += 1
            elif int(cls) == SPORTS_BALL and conf > best_conf:
                x1, y1, x2, y2 = box
                best_ball = ((x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) / 2)
                best_conf = conf
        return {"people": people, "ball": best_ball}


def is_game_view(people_count):
    return people_count >= MIN_PLAYERS_FOR_GAME_VIEW
