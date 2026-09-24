"""Classical front end: find the handwritten numbers on a slip.

Neural networks only ever see ONE text line (line/CTC model, recommended) or
ONE digit (per-digit model, smallest). Finding the lines, parsing the digits
into numbers and adding them up is plain code. Everything here is simple
integer / loop code that maps line-for-line onto the C port in
firmware/digit_pipeline.c (which is tested against this file).

Pipeline for a grayscale frame (e.g. 320x240 from an OV2640):
  1. background = box mean over a (2R+1)^2 window (integral image)
  2. ink mask   = pixel darker than background by K percent (and by MIN_CONTRAST)
  3. erase very long horizontal runs (ruled paper lines)
  4. connected components (8-connected, run-length + union-find)
  5. drop specks / dashes / huge blobs, cluster boxes into text lines
  Line model : 6. crop each line, scale to LINE_H x LINE_W
  Digit model: 6. merge stacked fragments, split over-wide (touching) boxes,
               7. group digits into numbers by gaps, 8. scale each to DIGIT_SIZE^2
"""
import numpy as np

DIGIT_SIZE = 20      # model input is DIGIT_SIZE x DIGIT_SIZE
DIGIT_BOX = 16       # digit is scaled so its longer side is DIGIT_BOX (MNIST-like)

BG_RADIUS = 15       # background window is (2R+1)^2
K_PERCENT = 15       # ink if gray < bg * (100 - K) / 100
MIN_CONTRAST = 12    # ... and bg - gray > MIN_CONTRAST
MIN_AREA = 6         # components smaller than this are noise
LONG_RUN_FRAC = 4    # horizontal runs longer than width / LONG_RUN_FRAC are ruled lines
NUMBER_GAP = 1.0     # gap > NUMBER_GAP * digit height starts a new number
SPLIT_RATIO = 1.25   # boxes wider than SPLIT_RATIO * height are split into touching digits
DIGIT_ASPECT = 0.75  # typical digit width / height, used to guess how many digits a wide box holds


def background(gray, r=BG_RADIUS):
    """Box mean over a clamped (2r+1)^2 window, integer division like the C code."""
    g = gray.astype(np.int64)
    h, w = g.shape
    ii = np.zeros((h + 1, w + 1), np.int64)
    ii[1:, 1:] = g.cumsum(0).cumsum(1)
    ys, xs = np.arange(h), np.arange(w)
    y0, y1 = np.clip(ys - r, 0, h), np.clip(ys + r + 1, 0, h)
    x0, x1 = np.clip(xs - r, 0, w), np.clip(xs + r + 1, 0, w)
    s = ii[y1][:, x1] - ii[y0][:, x1] - ii[y1][:, x0] + ii[y0][:, x0]
    area = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    return s // area


def ink_mask(gray, bg):
    g = gray.astype(np.int64)
    return (g * 100 < bg * (100 - K_PERCENT)) & (bg - g > MIN_CONTRAST)


def row_runs(row):
    """[(start, end_exclusive), ...] of True runs in a 1-D bool array."""
    padded = np.concatenate(([0], row.astype(np.int8), [0]))
    d = np.diff(padded)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def remove_long_runs(mask):
    max_len = mask.shape[1] // LONG_RUN_FRAC
    out = mask.copy()
    for y in range(mask.shape[0]):
        for s, e in row_runs(mask[y]):
            if e - s > max_len:
                out[y, s:e] = False
    return out


def label(mask):
    """8-connected components. Returns (label image with ids 1..n, list of comps).

    comp = dict(x0, y0, x1, y1 (exclusive), area, id)
    """
    h, w = mask.shape
    parent = []

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    runs = []          # (y, s, e)
    prev = []          # indices of runs in previous row
    for y in range(h):
        cur = []
        j = 0
        for s, e in row_runs(mask[y]):
            idx = len(runs)
            runs.append((y, s, e))
            parent.append(idx)
            # 8-connectivity: previous-row run [ps, pe) touches [s, e) if ps <= e and pe >= s
            while j < len(prev) and runs[prev[j]][2] < s:
                j += 1
            k = j
            while k < len(prev) and runs[prev[k]][1] <= e:
                a, b = find(idx), find(prev[k])
                if a != b:
                    parent[max(a, b)] = min(a, b)
                k += 1
            if k > j:
                j = k - 1   # last overlapping run may also touch the next run
            cur.append(idx)
        prev = cur

    lab = np.zeros((h, w), np.int32)
    comps = {}
    for i, (y, s, e) in enumerate(runs):
        root = find(i)
        c = comps.get(root)
        if c is None:
            c = comps[root] = dict(x0=s, y0=y, x1=e, y1=y + 1, area=0, id=len(comps) + 1)
        c["x0"], c["x1"] = min(c["x0"], s), max(c["x1"], e)
        c["y1"] = y + 1
        c["area"] += e - s
        lab[y, s:e] = c["id"]
    return lab, sorted(comps.values(), key=lambda c: c["id"])


def _h(b):
    return b["y1"] - b["y0"]


def _w(b):
    return b["x1"] - b["x0"]


def text_lines(gray):
    """Shared front half: -> (ink, lab, H, lines) with lines = [[box, ...], ...] top to bottom.

    ink : int64 image, bg - gray clipped at 0 (contrast map used for normalising)
    lab : component label image
    H   : typical digit height in pixels
    box : dict(x0, y0, x1, y1 (exclusive), ids=[component ids]), unsorted within a line
    """
    bg = background(gray)
    mask = remove_long_runs(ink_mask(gray, bg))
    lab, comps = label(mask)
    ink = np.maximum(bg - gray.astype(np.int64), 0)
    comps = [c for c in comps if c["area"] >= MIN_AREA]
    if not comps:
        return ink, lab, 0, []

    tall = sorted(_h(c) for c in comps if _h(c) >= 6)
    H = tall[len(tall) // 2] if tall else 8     # typical digit height

    kept = []
    for c in comps:
        h, w = _h(c), _w(c)
        if h < 0.3 * H and w < 0.3 * H:          # specks
            continue
        if w > 2.5 * h and h < 0.5 * H:          # dashes, underlines
            continue
        if h > 2.5 * H or w > 6 * H:             # borders, scribbles
            continue
        kept.append(dict(x0=c["x0"], y0=c["y0"], x1=c["x1"], y1=c["y1"], ids=[c["id"]]))

    # cluster into text lines by vertical centre
    kept.sort(key=lambda b: (b["y0"] + b["y1"], b["x0"]))
    raw_lines = []   # [cy_sum, n, [boxes]]
    for b in kept:
        cy = (b["y0"] + b["y1"]) / 2
        for ln in raw_lines:
            if abs(cy - ln[0] / ln[1]) < 0.6 * H:
                ln[0] += cy
                ln[1] += 1
                ln[2].append(b)
                break
        else:
            raw_lines.append([cy, 1, [b]])
    raw_lines.sort(key=lambda ln: ln[0] / ln[1])
    return ink, lab, H, [boxes for _, _, boxes in raw_lines]


def find_lines(gray):
    """-> (ink, lab, [line box]) - one box per text line, for the line (CTC) model."""
    ink, lab, _, lines = text_lines(gray)
    out = []
    for boxes in lines:
        out.append(dict(x0=min(b["x0"] for b in boxes), y0=min(b["y0"] for b in boxes),
                        x1=max(b["x1"] for b in boxes), y1=max(b["y1"] for b in boxes),
                        ids=[i for b in boxes for i in b["ids"]]))
    return ink, lab, out


def find_digits(gray):
    """-> (ink, lab, lines) for the per-digit model.

    lines : [[number, ...], ...] top to bottom; number = [box, ...] left to right.
    """
    ink, lab, H, raw_lines = text_lines(gray)
    lines = []
    for boxes in raw_lines:
        boxes.sort(key=lambda b: b["x0"])
        # merge fragments that sit on top of each other (e.g. the flag of a 5)
        merged = []
        for b in boxes:
            if merged:
                m = merged[-1]
                overlap = min(m["x1"], b["x1"]) - max(m["x0"], b["x0"])
                if overlap > 0.5 * min(_w(m), _w(b)):
                    m["x0"], m["x1"] = min(m["x0"], b["x0"]), max(m["x1"], b["x1"])
                    m["y0"], m["y1"] = min(m["y0"], b["y0"]), max(m["y1"], b["y1"])
                    m["ids"] = m["ids"] + b["ids"]
                    continue
            merged.append(dict(b))
        # split boxes that are too wide to be a single digit (touching digits)
        digits = []
        for b in merged:
            digits.extend(_split(b, lab, H))
        if not digits:
            continue
        # group into numbers by gaps
        numbers = [[digits[0]]]
        for a, b in zip(digits, digits[1:]):
            if b["x0"] - a["x1"] > NUMBER_GAP * H:
                numbers.append([])
            numbers[-1].append(b)
        lines.append(numbers)
    return ink, lab, lines


def _split(b, lab, H):
    h, w = _h(b), _w(b)
    ref = max(h, 0.8 * H)
    if w <= SPLIT_RATIO * ref:
        return [b]
    n = int(round(w / (DIGIT_ASPECT * ref)))
    n = max(2, min(n, 4))
    sub = np.isin(lab[b["y0"]:b["y1"], b["x0"]:b["x1"]], b["ids"])
    col = sub.sum(0)
    cuts = [0]
    for i in range(1, n):
        ideal = i * w // n
        lo, hi = max(cuts[-1] + 1, ideal - w // (4 * n)), min(w - 1, ideal + w // (4 * n))
        best = ideal
        for x in range(lo, hi + 1):
            if col[x] < col[best]:
                best = x
        cuts.append(best)
    cuts.append(w)
    out = []
    for s, e in zip(cuts, cuts[1:]):
        if e <= s:
            continue
        rows = np.flatnonzero(sub[:, s:e].any(1))
        if len(rows) == 0:
            continue
        out.append(dict(x0=b["x0"] + s, x1=b["x0"] + e,
                        y0=b["y0"] + rows[0], y1=b["y0"] + rows[-1] + 1, ids=b["ids"]))
    return out


def crop_digit(ink, lab, box):
    """Ink inside a box (digit or line), only pixels of its components, scaled to 0..1 float."""
    region = ink[box["y0"]:box["y1"], box["x0"]:box["x1"]]
    keep = np.isin(lab[box["y0"]:box["y1"], box["x0"]:box["x1"]], box["ids"])
    d = np.where(keep, region, 0).astype(np.float32)
    peak = d.max()
    return d / peak if peak > 0 else d


def normalize(crop):
    """Tight-crop, scale longer side to DIGIT_BOX, centre in DIGIT_SIZE x DIGIT_SIZE."""
    ys, xs = np.nonzero(crop > 0)
    out = np.zeros((DIGIT_SIZE, DIGIT_SIZE), np.float32)
    if len(ys) == 0:
        return out
    crop = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    scale = DIGIT_BOX / max(h, w)
    th, tw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    oy, ox = (DIGIT_SIZE - th) // 2, (DIGIT_SIZE - tw) // 2
    out[oy:oy + th, ox:ox + tw] = resize(crop, th, tw)
    return out


LINE_H, LINE_W = 24, 96   # line model input
LINE_INK_H = 20           # text height inside the line image
LINE_PAD_X = 2


def normalize_line(crop):
    """Tight-crop a line's ink, scale to LINE_INK_H tall (shrinking further if too wide),
    left-align into a LINE_H x LINE_W image."""
    ys, xs = np.nonzero(crop > 0)
    out = np.zeros((LINE_H, LINE_W), np.float32)
    if len(ys) == 0:
        return out
    crop = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    scale = min(LINE_INK_H / h, (LINE_W - 2 * LINE_PAD_X) / w)
    th, tw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    oy = (LINE_H - th) // 2
    out[oy:oy + th, LINE_PAD_X:LINE_PAD_X + tw] = resize(crop, th, tw)
    return out


def resize(crop, th, tw):
    """Supersampled bilinear resize: every output pixel is the mean of SxS bilinear
    samples inside its footprint - cheap area-averaging that ports easily to C."""
    h, w = crop.shape
    S = 3
    sub = (np.arange(S) + 0.5) / S
    fy = np.clip(((np.arange(th)[:, None] + sub).ravel()) * h / th - 0.5, 0, h - 1)
    fx = np.clip(((np.arange(tw)[:, None] + sub).ravel()) * w / tw - 0.5, 0, w - 1)
    y0, x0 = fy.astype(np.int64), fx.astype(np.int64)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    dy, dx = (fy - y0)[:, None], (fx - x0)[None, :]
    c = crop.astype(np.float64)
    top = c[y0][:, x0] * (1 - dx) + c[y0][:, x1] * dx
    bot = c[y1][:, x0] * (1 - dx) + c[y1][:, x1] * dx
    samples = top * (1 - dy) + bot * dy
    return samples.reshape(th, S, tw, S).mean((1, 3)).astype(np.float32)


def extract_digits(gray):
    """gray -> (lines of numbers of boxes, same structure of 20x20 float digit images)."""
    ink, lab, lines = find_digits(gray)
    imgs = [[[normalize(crop_digit(ink, lab, b)) for b in num] for num in ln] for ln in lines]
    return lines, imgs


def extract_lines(gray):
    """gray -> (line boxes, [LINE_H x LINE_W float image per line])."""
    ink, lab, boxes = find_lines(gray)
    return boxes, [normalize_line(crop_digit(ink, lab, b)) for b in boxes]
