# Tacta scorer

Scores a finished Tacta game from a photo: one point per visible dot of each
player's color; most points wins, ties share the victory.

Setup (Python 3.10+):

    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt

Usage:

    .venv/bin/python tacta_score.py photo.jpg --players red blue green --debug check.png

- `--players` - colors in the game; each is shown even if it has 0 visible dots.
  Without it, the colors shown are those with at least one dot or a visible
  center-number marker, so a color whose dots are all covered shows as 0 as
  long as one of its center numbers is visible (dots are always classified
  against every palette color)
- `--debug` - writes the photo with every counted dot circled and labelled with a
  per-color running count (`r1`, `r2`, ... `pi1`, `pu1`, ...), numbered along each
  run of nearby dots, runs top to bottom, plus a totals legend in the top-left
  corner; check this
- `--palette` - JSON `{"color": hue}` on OpenCV's 0-179 hue scale
- `--dot-radius` - expected dot radius in photo pixels if auto-sizing misfires
- `--json` - machine-readable output

Photo tips: shoot from directly above, even light, no glare, whole table in frame.
A dot only counts if it sits in a solid patch of one card color (which rules out
flecks in a colorful rug) and that patch is larger than a card's center-number
disc (which rules out the white digit, which can look like a dot when the cards
are small in the photo).

Illegally placed cards are not detected; remove them before taking the photo.

Tests (the real-photo tests use the hand-counted photos in `samples/`):

    .venv/bin/python -m pytest
