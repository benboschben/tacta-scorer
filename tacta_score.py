"""Score a finished Tacta game from a photo of the table.

Rule: each player scores one point per visible dot of their own color; the
most points wins and ties share the victory.

Scoring dots are small white circles printed on the deck's colored
background. We find white blobs that are round, solid (not a ring, square,
triangle or digit) and close to the typical dot size in the photo, then read
the dot's color from a thin ring of pixels just outside it. White blobs not
surrounded by one palette color (glare, the table, the start card) are
rejected.
"""
import argparse
import json
import math
import sys
from dataclasses import dataclass

import cv2
import numpy as np

# Hue centers on OpenCV's 0-179 scale, measured from a photo of the real
# cards. Override with --palette if your lighting shifts them.
DEFAULT_PALETTE = {
    "red": 178,
    "orange": 19,
    "green": 43,
    "blue": 103,
    "purple": 131,
    "pink": 167,
}


@dataclass
class Settings:
    min_saturation: int = 90
    min_value: int = 90
    hue_tolerance: int = 12
    # What counts as a white dot pixel.
    max_white_saturation: int = 70
    min_white_value: int = 170
    # Ring sampled around each dot, as multiples of the dot radius, and the
    # share of it that must be one palette color. Printed dots measure >= 0.96;
    # white flecks in a multicolored rug and border-line junctions <= 0.86.
    surround: tuple = (1.3, 1.8)
    min_surround_share: float = 0.9
    # Radius of the colored patch the dot sits in (its minimum enclosing
    # circle), in dot radii. Scoring dots sit in strips and corners (>= 5);
    # the digit in a card's center number sits in a small disc or triangle
    # (<= 3.2) and can look like a dot when the photo is low resolution.
    min_patch_radius: float = 4.0
    # Center-number marker that shows a color is on the table even when all
    # its dots are covered: enclosing radius in dot radii (measured 2.5-3.9),
    # white digit area inside in dot areas (>= 0.6), and the largest share of
    # its border that may be another palette color (<= 0.09).
    marker_radius: tuple = (2.0, 4.5)
    min_marker_digit: float = 0.4
    max_marker_other: float = 0.2
    # Blob pixel area / area of its minimum enclosing circle. A filled circle
    # is ~0.9; squares (0.64), triangles (0.41) and rings fall below this.
    min_roundness: float = 0.75
    # Allowed blob area relative to the median candidate area.
    size_band: tuple = (0.4, 2.5)
    min_area: int = 12
    work_size: int = 2000
    # Expected dot radius in original-image pixels; None = estimate.
    dot_radius: float | None = None


@dataclass
class Dot:
    x: float
    y: float
    radius: float
    color: str
    area: float = 0.0


def hue_distance(hue, center):
    d = np.abs(hue.astype(np.int16) - center)
    return np.minimum(d, 180 - d)


def classify_pixels(hsv, palette, settings):
    """Return a label image: index into palette, or -1 for unassigned."""
    hue, sat, val = cv2.split(hsv)
    dists = np.stack([hue_distance(hue, c) for c in palette.values()])
    labels = np.argmin(dists, axis=0).astype(np.int16)
    ok = (
        (np.min(dists, axis=0) <= settings.hue_tolerance)
        & (sat >= settings.min_saturation)
        & (val >= settings.min_value)
    )
    labels[~ok] = -1
    return labels


def round_blobs(mask, settings):
    """Yield (x, y, radius, area) for solid, round connected blobs."""
    # RETR_LIST, not RETR_EXTERNAL: the cards' white outlines are closed loops
    # and the dots sit inside them. Inner edges of rings come back as contours
    # too, but they enclose dark pixels and fail the roundness test.
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour) + len(contour) / 2  # include boundary
        if area < settings.min_area:
            continue
        (x, y), r = cv2.minEnclosingCircle(contour)
        bx, by, bw, bh = cv2.boundingRect(contour)
        filled = np.zeros((bh, bw), np.uint8)
        cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED, offset=(-bx, -by))
        roi = mask[by:by + bh, bx:bx + bw]
        pixels = cv2.countNonZero(cv2.bitwise_and(roi, filled))
        if pixels / (math.pi * max(r, 1) ** 2) >= settings.min_roundness:
            yield x, y, r, pixels


def surround_ring(shape, x, y, r, settings):
    """Row and column indices of the ring of pixels just outside (x, y, r)."""
    inner, outer = settings.surround
    h, w = shape
    x0, x1 = max(int(x - r * outer) - 1, 0), min(int(x + r * outer) + 2, w)
    y0, y1 = max(int(y - r * outer) - 1, 0), min(int(y + r * outer) + 2, h)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist = np.hypot(xx - x, yy - y)
    ring = (dist >= r * inner) & (dist <= r * outer)
    return yy[ring], xx[ring]


def surround_color(labels, ring, n_colors, settings):
    """Index of the palette color filling the ring, or None."""
    values = labels[ring]
    if values.size == 0:
        return None
    counts = np.bincount(values[values >= 0], minlength=n_colors)
    best = int(np.argmax(counts))
    if counts[best] < settings.min_surround_share * values.size:
        return None
    return best


class Patches:
    """Connected regions of each palette color, built lazily per color."""

    def __init__(self, labels):
        self.labels = labels
        self.components = {}
        self.radii = {}

    def radius(self, color, ring):
        """Enclosing-circle radius of the `color` region covering most of the ring."""
        if color not in self.components:
            mask = (self.labels == color).astype(np.uint8)
            self.components[color] = cv2.connectedComponentsWithStats(mask, connectivity=8)
        _, comp, stats, _ = self.components[color]
        ids = comp[ring]
        ids = ids[ids > 0]
        if ids.size == 0:
            return 0.0
        cid = int(np.bincount(ids).argmax())
        if (color, cid) not in self.radii:
            x, y, w, h = stats[cid, :4]
            pts = np.argwhere(comp[y:y + h, x:x + w] == cid)[:, ::-1] + (x, y)
            self.radii[color, cid] = cv2.minEnclosingCircle(pts.astype(np.float32))[1]
        return self.radii[color, cid]


def center_markers(labels, white, r, n_colors, settings):
    """Palette colors with a visible center-number marker: a small solid patch
    of one color, about 2.5-4 dot radii across, holding a white digit and
    bordered only by white or dark pixels. A card whose dots are all covered
    usually still shows this; colored background does not."""
    lo, hi = settings.marker_radius
    found = set()
    for color in range(n_colors):
        mask = (labels == color).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if not lo * r <= cv2.minEnclosingCircle(contour)[1] <= hi * r:
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            pad = max(round(r), 1)
            x0, y0 = max(bx - pad, 0), max(by - pad, 0)
            x1, y1 = min(bx + bw + pad, mask.shape[1]), min(by + bh + pad, mask.shape[0])
            shape = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.drawContours(shape, [contour], -1, 1, cv2.FILLED, offset=(-x0, -y0))
            inside = shape > 0
            digit = inside & (mask[y0:y1, x0:x1] == 0) & (white[y0:y1, x0:x1] > 0)
            if digit.sum() < settings.min_marker_digit * math.pi * r * r:
                continue
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1,) * 2)
            border = (cv2.dilate(shape, kernel) > 0) & ~inside
            around = labels[y0:y1, x0:x1][border]
            if np.mean((around >= 0) & (around != color)) <= settings.max_marker_other:
                found.add(color)
                break
    return found


def analyse(image, palette=DEFAULT_PALETTE, settings=Settings()):
    """Find scoring dots and the colors present in a BGR image.

    Returns (dots, colors): dots in image pixels, and the palette colors that
    have a dot or a visible center marker, in palette order."""
    scale = min(1.0, settings.work_size / max(image.shape[:2]))
    work = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    work = cv2.GaussianBlur(work, (3, 3), 0)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    labels = classify_pixels(hsv, palette, settings)
    names = list(palette)

    raw_white = cv2.inRange(
        hsv, (0, 0, settings.min_white_value), (179, settings.max_white_saturation, 255)
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    white = cv2.morphologyEx(raw_white, cv2.MORPH_OPEN, kernel)

    patches = Patches(labels)
    candidates = []
    for x, y, r, area in round_blobs(white, settings):
        ring = surround_ring(labels.shape, x, y, r, settings)
        color = surround_color(labels, ring, len(names), settings)
        if color is None:
            continue
        if patches.radius(color, ring) < settings.min_patch_radius * r:
            continue
        candidates.append(Dot(x / scale, y / scale, r / scale, names[color], area))

    dots = []
    if candidates:
        if settings.dot_radius:
            typical = math.pi * (settings.dot_radius * scale) ** 2
        else:
            typical = float(np.median([d.area for d in candidates]))
        low, high = settings.size_band
        dots = [d for d in candidates if low * typical <= d.area <= high * typical]

    present = {d.color for d in dots}
    if settings.dot_radius:
        dot_r = settings.dot_radius * scale
    elif dots:
        dot_r = float(np.median([d.radius for d in dots])) * scale
    else:
        dot_r = None  # nothing to size markers against
    if dot_r:
        markers = center_markers(labels, raw_white, dot_r, len(names), settings)
        present |= {names[c] for c in markers}
    return dots, [c for c in names if c in present]


def detect_dots(image, palette=DEFAULT_PALETTE, settings=Settings()):
    """Find scoring dots in a BGR image. Coordinates are in image pixels."""
    return analyse(image, palette, settings)[0]


def score(dots, players):
    totals = {p: 0 for p in players}
    for dot in dots:
        if dot.color in totals:
            totals[dot.color] += 1
    return totals


def winners(totals):
    best = max(totals.values(), default=0)
    return [p for p, s in totals.items() if s == best]


def short_names(names):
    """Shortest unique prefix of each name: red->r, pink->pi, purple->pu."""
    out = {}
    for name in names:
        n = 1
        while any(o != name and o.startswith(name[:n]) for o in names):
            n += 1
        out[name] = name[:n]
    return out


def reading_order(dots, link=12):
    """Order dots the way a person counts them: group dots within `link`
    radii of each other into runs (a card edge's row of dots), take runs
    top-to-bottom, and walk each run end to end, left-to-right (top-to-bottom
    if it is closer to vertical)."""
    if not dots:
        return []
    reach = link * float(np.median([d.radius for d in dots]))
    pts = np.array([(d.x, d.y) for d in dots])
    group = list(range(len(dots)))

    def root(i):
        while group[i] != i:
            group[i] = group[group[i]]
            i = group[i]
        return i

    for i in range(len(dots)):
        near = np.hypot(*(pts[i + 1:] - pts[i]).T) <= reach
        for j in np.nonzero(near)[0] + i + 1:
            group[root(i)] = root(j)

    runs = {}
    for i in range(len(dots)):
        runs.setdefault(root(i), []).append(i)
    ordered = []
    for members in sorted(runs.values(), key=lambda m: tuple(pts[m].mean(axis=0)[::-1])):
        p = pts[members]
        if len(members) > 1:
            _, _, vt = np.linalg.svd(p - p.mean(axis=0))
            axis = vt[0]
            if abs(axis[0]) >= abs(axis[1]) and axis[0] < 0 or \
                    abs(axis[0]) < abs(axis[1]) and axis[1] < 0:
                axis = -axis
            members = [members[k] for k in np.argsort(p @ axis)]
        ordered += [dots[i] for i in members]
    return ordered


def put_label(img, text, org, font_scale, thickness):
    """Black text with a white halo so it reads on any background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, org, font, font_scale, (255, 255, 255), thickness * 3, cv2.LINE_AA)
    cv2.putText(img, text, org, font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


def annotate(image, dots, colors=()):
    """Circle each dot and label it with a per-color running count, e.g. pu7,
    numbered in reading order so the photo can be checked by counting along.
    The legend lists every color with a dot plus `colors`, which show as 0
    when they have none."""
    out = image.copy()
    tags = short_names(sorted({d.color for d in dots} | set(colors)))
    counts = dict.fromkeys(colors, 0)
    for dot in reading_order(dots):
        counts[dot.color] = counts.get(dot.color, 0) + 1
        center = (round(dot.x), round(dot.y))
        radius = max(round(dot.radius * 1.6), 4)
        cv2.circle(out, center, radius, (0, 0, 0), 4)
        cv2.circle(out, center, max(round(dot.radius * 0.5), 2), (0, 0, 0), -1)
        # Hershey simplex caps are ~22px tall at scale 1; make labels about
        # one dot diameter tall.
        font_scale = max(dot.radius * 2 / 22, 0.4)
        thickness = max(round(font_scale * 2), 1)
        org = (center[0] + radius + 2, center[1] + round(dot.radius))
        put_label(out, f"{tags[dot.color]}{counts[dot.color]}", org, font_scale, thickness)

    # Totals legend, top-left, sized to the image.
    font_scale = max(image.shape[1] / 1500, 0.5)
    thickness = max(round(font_scale * 2), 1)
    line = round(32 * font_scale)
    for i, color in enumerate(sorted(counts, key=lambda c: -counts[c])):
        put_label(out, f"{tags[color]} = {color}: {counts[color]}",
                  (line // 2, line * (i + 1)), font_scale, thickness)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score a Tacta game from a photo.")
    parser.add_argument("photo")
    parser.add_argument("--players", nargs="+", metavar="COLOR",
                        help="colors in the game, shown even with 0 dots "
                             "(default: every color with a dot or center marker)")
    parser.add_argument("--palette", help="JSON file mapping color name to hue (0-179)")
    parser.add_argument("--dot-radius", type=float,
                        help="expected dot radius in photo pixels (default: estimate)")
    parser.add_argument("--debug", metavar="OUT.png",
                        help="write the photo with every counted dot circled and "
                             "numbered per color in reading order")
    parser.add_argument("--json", action="store_true", help="print scores as JSON")
    args = parser.parse_args(argv)

    palette = DEFAULT_PALETTE
    if args.palette:
        with open(args.palette) as f:
            palette = json.load(f)
    unknown = [p for p in args.players or [] if p not in palette]
    if unknown:
        parser.error(f"not in palette: {', '.join(unknown)}")

    image = cv2.imread(args.photo)
    if image is None:
        parser.error(f"cannot read image: {args.photo}")

    dots, present = analyse(image, palette, Settings(dot_radius=args.dot_radius))
    players = args.players or present
    totals = score(dots, players)
    if args.debug:
        cv2.imwrite(args.debug, annotate(image, dots, players))

    if args.json:
        print(json.dumps({"scores": totals, "winners": winners(totals)}))
    else:
        for player, points in sorted(totals.items(), key=lambda kv: -kv[1]):
            print(f"{player:>8}: {points}")
        top = winners(totals)
        if not top:
            print("No scoring dots found.")
        else:
            print(("Tie: " if len(top) > 1 else "Winner: ") + ", ".join(top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
