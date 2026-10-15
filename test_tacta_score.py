import cv2
import numpy as np
import pytest

from tacta_score import DEFAULT_PALETTE, analyse, detect_dots, main, score, winners


def bgr(color):
    hsv = np.uint8([[[DEFAULT_PALETTE[color], 230, 235]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


WHITE = (245, 245, 245)


def draw_card(img, x, y, color, dots, w=180, h=260):
    """A colored card with `dots` white scoring dots plus non-scoring marks."""
    cv2.rectangle(img, (x, y), (x + w, y + h), bgr(color), -1)
    cv2.circle(img, (x + 40, y + 40), 22, WHITE, 4)                      # ring
    cv2.rectangle(img, (x + 110, y + 20), (x + 160, y + 70), WHITE, -1)  # square
    tri = np.array([[x + 20, y + 240], [x + 70, y + 240], [x + 45, y + 195]])
    cv2.fillPoly(img, [tri], WHITE)                                      # triangle
    cv2.circle(img, (x + 130, y + 215), 30, WHITE, -1)                   # big disc
    cv2.putText(img, "7", (x + 150, y + 175), cv2.FONT_HERSHEY_SIMPLEX,
                1, WHITE, 3)                                             # number
    for i in range(dots):
        cv2.circle(img, (x + 40 + 50 * (i % 3), y + 110 + 40 * (i // 3)), 8, WHITE, -1)


def table():
    img = np.full((900, 1300, 3), 20, np.uint8)
    draw_card(img, 50, 50, "red", 5)
    draw_card(img, 280, 50, "blue", 6)
    draw_card(img, 510, 50, "green", 4)
    draw_card(img, 50, 400, "pink", 3)
    draw_card(img, 280, 400, "orange", 2)
    draw_card(img, 510, 400, "purple", 6)
    draw_card(img, 900, 300, "blue", 3)
    # A red card over the blue card's first dot row hides 3 blue dots and
    # shows 2 red ones.
    cv2.rectangle(img, (300, 140), (480, 180), bgr("red"), -1)
    for x in (340, 440):
        cv2.circle(img, (x, 160), 8, WHITE, -1)
    # Dot-sized glare on the bare table: white but not on a card.
    cv2.circle(img, (1150, 750), 8, WHITE, -1)
    rng = np.random.default_rng(0)
    # A card's center number: a small colored disc whose white digit has
    # blurred into a dot-like blob.
    cv2.circle(img, (1150, 120), 20, bgr("green"), -1)
    cv2.circle(img, (1150, 120), 8, WHITE, -1)
    # A multicolored rug: mostly orange clumps of yarn with other colors mixed in,
    # and a white fleck the size of a dot.
    rug = rng.choice([bgr("orange")] * 3 + [bgr("green"), (150, 160, 170)], (30, 40))
    rug = cv2.resize(np.array(rug, np.uint8), (160, 120), interpolation=cv2.INTER_NEAREST)
    img[600:720, 1000:1160] = rug
    cv2.circle(img, (1080, 660), 8, WHITE, -1)
    noisy = img.astype(np.int16) + rng.normal(0, 6, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


EXPECTED = {"red": 7, "blue": 6, "green": 4, "pink": 3, "orange": 2, "purple": 6}


def test_counts_visible_dots_per_color():
    assert score(detect_dots(table()), list(EXPECTED)) == EXPECTED


def test_scale_independent():
    big = cv2.resize(table(), None, fx=3.5, fy=3.5, interpolation=cv2.INTER_CUBIC)
    assert score(detect_dots(big), list(EXPECTED)) == EXPECTED


def test_ties_share_victory():
    assert winners({"red": 7, "blue": 7, "pink": 3}) == ["red", "blue"]


def test_cli_restricted_players(tmp_path, capsys):
    photo = tmp_path / "game.png"
    cv2.imwrite(str(photo), table())
    main([str(photo), "--players", "red", "blue", "--json", "--debug",
          str(tmp_path / "debug.png")])
    out = capsys.readouterr().out
    assert '"scores": {"red": 7, "blue": 6}' in out
    assert '"winners": ["red"]' in out
    assert (tmp_path / "debug.png").exists()


REAL_PHOTOS = {
    # Hand-counted from each photo; None until counted.
    "game1.jpg": {"red": 15, "pink": 14, "purple": 10, "green": 6, "orange": 5, "blue": 5},
    # Orange cards with every dot covered; only their center markers show.
    "No_orange.jpg": {"pink": 15, "red": 11, "orange": 0},
    # Center-number digits and a multicolored rug in frame.
    "Three_color.jpg": {"green": 47, "red": 44, "blue": 35},
    "Tie.jpg": {"red": 15, "pink": 15, "purple": 12, "green": 6, "orange": 5, "blue": 5},
}


@pytest.mark.parametrize("name", REAL_PHOTOS)
def test_real_photo(name):
    expected = REAL_PHOTOS[name]
    if expected is None:
        pytest.skip(f"no hand count for {name} yet")
    dots, present = analyse(cv2.imread(f"samples/{name}"))
    assert score(dots, list(DEFAULT_PALETTE)) == {
        c: expected.get(c, 0) for c in DEFAULT_PALETTE}
    # Every color in the hand count is on the table, even with 0 dots.
    assert set(present) == set(expected)


def test_cli_shows_only_colors_present(tmp_path, capsys):
    img = table()
    img[400:700, 280:500] = 20  # remove the orange card
    photo = tmp_path / "game.png"
    cv2.imwrite(str(photo), img)
    main([str(photo), "--json"])
    assert "orange" not in capsys.readouterr().out
    main([str(photo), "--json", "--players", "red", "orange"])
    assert '"scores": {"red": 7, "orange": 0}' in capsys.readouterr().out


def test_cli_no_dots(tmp_path, capsys):
    photo = tmp_path / "empty.png"
    cv2.imwrite(str(photo), np.full((300, 300, 3), 20, np.uint8))
    main([str(photo)])
    assert capsys.readouterr().out == "No scoring dots found.\n"


def test_color_present_from_center_marker_alone():
    img = table()
    # An orange card face with its center marker but every dot covered.
    cv2.rectangle(img, (760, 650), (940, 860), (25, 25, 25), -1)
    cv2.circle(img, (850, 755), 21, (240, 240, 240), 2)
    cv2.circle(img, (850, 755), 19, bgr("orange"), -1)
    cv2.putText(img, "3", (841, 766), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 3)
    dots, present = analyse(img)
    assert score(dots, ["orange"]) == {"orange": 2}
    img[400:700, 280:500] = 20  # remove the orange card that has dots
    dots, present = analyse(img)
    assert "orange" in present and score(dots, ["orange"]) == {"orange": 0}
