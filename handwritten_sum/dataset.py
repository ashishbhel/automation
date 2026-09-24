"""Turn slips (synthetic or real) into labelled 20x20 digit crops.

Crops are produced by the *same* segmentation that runs on the device, then
matched to ground-truth digit boxes by IoU. Detections that match no digit
(a '+', an underline stub, a speck, half of a badly split digit) get label 10
= "not a digit", so the model learns to reject them instead of guessing.
"""
import numpy as np

import segment

JUNK = 10
NUM_CLASSES = 11


def iou(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def crops_from_slip(gray, gt_boxes, min_iou=0.5):
    """-> (X [n,20,20] float32, y [n] int) for one slip."""
    ink, lab, lines = segment.find_digits(gray)
    gts = [(g[2], g[3], g[4], g[5]) for g in gt_boxes]
    X, y = [], []
    for ln in lines:
        for num in ln:
            for b in num:
                box = (b["x0"], b["y0"], b["x1"], b["y1"])
                scores = [iou(box, g) for g in gts]
                best = int(np.argmax(scores)) if scores else -1
                label = gt_boxes[best][0] if best >= 0 and scores[best] >= min_iou else JUNK
                X.append(segment.normalize(segment.crop_digit(ink, lab, b)))
                y.append(label)
    return X, y


def mnist_crops(images, labels):
    """Clean MNIST digits pushed through the same normalisation (extra, well-labelled data)."""
    X = [segment.normalize(img.astype(np.float32) / 255 * (img > 30)) for img in images]
    return X, list(labels)


def build(gen, n_slips, log_every=500):
    X, y = [], []
    for i in range(n_slips):
        gray, _, boxes = gen()
        xs, ys = crops_from_slip(gray, boxes)
        X += xs
        y += ys
        if log_every and (i + 1) % log_every == 0:
            print(f"  {i + 1}/{n_slips} slips, {len(y)} crops", flush=True)
    return np.array(X, np.float32)[..., None], np.array(y, np.int64)


# ---------------------------------------------------------------- line (CTC) model

BLANK = 10          # CTC blank index; classes 0..9 are digits
MAX_LABEL = 6       # longest label we train on (4 digits + slack)


def lines_from_slip(gray, numbers, gt_boxes):
    """-> (X [n, LINE_H, LINE_W], labels [str]) for one slip.

    A detected line is labelled with the number whose digits it fully contains.
    Lines containing no digit (underline stub, '+', specks) get the empty label,
    so the model learns to output nothing for them. Lines that mix digits of two
    numbers or miss some digits can't be labelled and are skipped.
    """
    ink, lab, boxes = segment.find_lines(gray)
    X, labels = [], []
    for b in boxes:
        inside = {}
        for d, li, x0, y0, x1, y1 in gt_boxes:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            hit = b["x0"] <= cx < b["x1"] and b["y0"] <= cy < b["y1"]
            inside.setdefault(li, []).append(hit)
        full = [li for li, hits in inside.items() if all(hits)]
        partial = [li for li, hits in inside.items() if any(hits) and not all(hits)]
        if partial or len(full) > 1:
            continue
        labels.append(str(numbers[full[0]]) if full else "")
        X.append(segment.normalize_line(segment.crop_digit(ink, lab, b)))
    return X, labels


def build_lines(gen, n_slips):
    X, labels = [], []
    for _ in range(n_slips):
        gray, numbers, boxes = gen()
        xs, ls = lines_from_slip(gray, numbers, boxes)
        X += xs
        labels += ls
    return np.array(X, np.float32)[..., None], labels


def encode(labels):
    """strings -> (dense int32 [n, MAX_LABEL] padded with BLANK, lengths [n])."""
    y = np.full((len(labels), MAX_LABEL), BLANK, np.int32)
    n = np.zeros(len(labels), np.int32)
    for i, s in enumerate(labels):
        y[i, :len(s)] = [int(c) for c in s]
        n[i] = len(s)
    return y, n


def ctc_greedy(probs):
    """probs [T, 11] -> (digit string, confidence = min over steps of the winning prob)."""
    best = probs.argmax(1)
    out, prev = [], BLANK
    for k in best:
        if k != prev and k != BLANK:
            out.append(str(k))
        prev = k
    return "".join(out), float(probs.max(1).min())
