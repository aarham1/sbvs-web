# Real clip test 1: Duke vs. Alabama, 2025 NCAA Elite Eight (first 4 minutes)

Clip: `sbvs 2.0 video test 1.mp4` (screen recording, 2732x2048, 240 s). Hand count by Jess
(`duke_alabama_truth.csv`, times in mm:ss converted to seconds). Score boxes:
Alabama (891,1605)-(1000,1715), Duke (1601,1605)-(1710,1715).

| | Jess (made) | SBVS |
|---|---|---|
| Made baskets found | 7 | 7, right team and right points for every one, same order |
| False scoring events | | 0 |
| Final points (clip) | Duke 13, Alabama 5 | Duke 13, Alabama 5 |
| Misses | 5 | not detected (no hoop model yet, Phase 2b) |

Timing: SBVS times are a steady 4.5 to 5.6 s later than Jess's for every basket (the
scoreboard updates after the ball goes in, plus 1 s of confirmation). An earlier run showed
5 to 10 s because the clip has a variable frame rate (screen recording) and times were
computed as frame number / fps; now each frame's own timestamp is used.

Processing time: about 160 s for the 240 s clip on a cloud CPU, at 2 checked frames/s.

Fix made during this test: before the scoreboard appeared, YouTube captions inside the box
were read as digits. OCR pieces with confidence below 0.3 are now ignored, and the chart's
starting score comes from the first clean event. With 0.5 one real "8" was read 20 s late,
so 0.3 was kept. One clip only: tune again on more footage.
