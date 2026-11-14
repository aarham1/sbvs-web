"""SBVS auto-analysis: upload game footage, get shooting stats and a next-game outlook.

Nothing is marked by hand. For every checked frame a basketball detector (ball, hoop,
player, referee) runs; the shot logic watches the ball around the hoop, the shooter is
the player holding the ball just before it went up, and teams come from jersey colors.
Stats are per team: jersey numbers are too small to read reliably on TV footage
(jersey.py is kept but not used).

Modules
  detector.py  ONNX object detector (no PyTorch needed)
  shots.py     shot attempt / make / miss logic from ball and hoop positions
  teams.py     jersey-color team split and color names
  players.py   player tracking, so each player's jersey color is sampled while he is alone
  jersey.py    jersey number OCR (Tesseract), NOT USED: kept for a possible later version
  analyze.py   the video loop that ties it together
  predict.py   game history and next-game prediction
"""
