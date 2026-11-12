"""Basketball object detection with an ONNX YOLOv8 model, run by onnxruntime.

Default model: E-BARD BODD_yolov8n_0001.onnx (Giudici, CC BY 4.0), trained on 60 NBA
broadcast games, classes Basketball, Hoop, Player, Referee. See docs/SOURCES.md.
Any Ultralytics YOLOv8/YOLO11 detection model exported to ONNX works: class names are
read from the file and matched by name (see CLASS_ALIASES).
"""

import ast
import os

import cv2
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "BODD_yolov8n_0001.onnx")

# Our role -> words that may appear in a model's class names (lower case).
CLASS_ALIASES = {
    "ball": ("basketball", "ball", "sports ball"),
    "hoop": ("hoop", "rim", "basket"),
    "player": ("player", "person"),
    "referee": ("referee", "ref"),
}

CONF = {"ball": 0.15, "hoop": 0.30, "player": 0.35, "referee": 0.35}
NMS_IOU = 0.5


def role_of(name):
    n = name.lower().strip()
    for role, words in CLASS_ALIASES.items():
        if n in words:
            return role
    for role, words in CLASS_ALIASES.items():
        if any(w in n for w in words):
            return role
    return None


class Detector:
    def __init__(self, model_path=MODEL_PATH, threads=None):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        self.size = shape[2] if isinstance(shape[2], int) else 640
        meta = self.session.get_modelmeta().custom_metadata_map
        names = ast.literal_eval(meta["names"]) if "names" in meta else {0: "ball"}
        self.names = {int(k): v for k, v in names.items()}
        self.roles = {k: role_of(v) for k, v in self.names.items()}

    def _letterbox(self, frame):
        h, w = frame.shape[:2]
        scale = self.size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        img = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = img
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return blob, scale, left, top

    def detect(self, frame):
        """Return {'ball': [...], 'hoop': [...], 'player': [...], 'referee': [...]}.
        Each item is (x1, y1, x2, y2, conf) in frame pixels, best first."""
        blob, scale, left, top = self._letterbox(frame)
        out = self.session.run(None, {self.input_name: blob})[0][0]   # (4+nc, N)
        if out.shape[0] > out.shape[1]:
            out = out.T
        boxes, scores = out[:4].T, out[4:].T
        cls = scores.argmax(1)
        conf = scores.max(1)
        result = {role: [] for role in CLASS_ALIASES}
        keep = conf > min(CONF.values())
        boxes, cls, conf = boxes[keep], cls[keep], conf[keep]
        if len(conf) == 0:
            return result
        cx, cy, bw, bh = boxes.T
        x1 = (cx - bw / 2 - left) / scale
        y1 = (cy - bh / 2 - top) / scale
        x2 = (cx + bw / 2 - left) / scale
        y2 = (cy + bh / 2 - top) / scale
        H, W = frame.shape[:2]
        x1, x2 = x1.clip(0, W), x2.clip(0, W)
        y1, y2 = y1.clip(0, H), y2.clip(0, H)
        for c in np.unique(cls):
            role = self.roles.get(int(c))
            if role is None:
                continue
            idx = np.where((cls == c) & (conf >= CONF[role]))[0]
            if len(idx) == 0:
                continue
            rects = [[float(x1[i]), float(y1[i]), float(x2[i] - x1[i]), float(y2[i] - y1[i])] for i in idx]
            kept = cv2.dnn.NMSBoxes(rects, [float(conf[i]) for i in idx], CONF[role], NMS_IOU)
            for k in np.array(kept).flatten():
                i = idx[int(k)]
                result[role].append((float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]), float(conf[i])))
        for role in result:
            result[role].sort(key=lambda b: -b[4])
        return result
