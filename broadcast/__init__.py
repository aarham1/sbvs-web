"""SBVS broadcast analysis (Phase 2).

Works on TV-style footage: moving camera, cuts, replays, an on-screen scoreboard.
Unlike the class version (sbvs-2/), this package uses pretrained machine-learning
models. See docs/SOURCES.md for where each technique comes from.

Modules:
    cuts.py        find camera cuts (color-histogram difference)
    scoreboard.py  read the score graphic with OCR and turn changes into scoring events
    detect.py      YOLO person / ball detection to tell game action from close-ups
    analyze.py     run everything on one video and build the stats tables
"""
