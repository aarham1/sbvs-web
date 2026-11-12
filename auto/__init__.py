"""SBVS auto-analysis: upload game footage, get shooting stats and a next-game outlook.

Nothing is marked by hand. For every checked frame a basketball detector (ball, hoop,
player, referee) runs; the shot logic watches the ball around the hoop, the shooter is
the player holding the ball just before it went up, teams come from jersey colors and
jersey numbers are read with OCR when they are big enough.

Modules
  detector.py  ONNX object detector (no PyTorch needed)
  shots.py     shot attempt / make / miss logic from ball and hoop positions
  teams.py     jersey-color team split and color names
  jersey.py    jersey number OCR (Tesseract)
  analyze.py   the video loop that ties it together
  predict.py   game history and next-game prediction
"""
