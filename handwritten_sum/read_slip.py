"""Read a slip photo and add up the numbers - the exact logic the ESP32 runs.

    python read_slip.py photo.jpg --model build/line_model_int8.tflite

Works on any photo (it is converted to grayscale and resized to 320x240 like
the ESP32 camera frame), so you can test with phone pictures of real slips.
"""
import argparse

import numpy as np
from PIL import Image

import dataset
import export
import segment

MIN_CONF = 0.6   # below this the reading is flagged instead of trusted


def read_lines(gray, model_path):
    """-> [(number or None, confidence, box)] for every text line; None = no digits."""
    boxes, imgs = segment.extract_lines(gray)
    if not imgs:
        return []
    probs = export.predict(model_path, np.array(imgs)[..., None])
    out = []
    for b, p in zip(boxes, probs):
        text, conf = dataset.ctc_greedy(p)
        out.append((int(text) if text else None, conf, b))
    return out


def read_digits(gray, model_path):
    """Per-digit model: -> [(number or None, confidence, boxes)] per number found."""
    lines, imgs = segment.extract_digits(gray)
    flat = [im for ln in imgs for num in ln for im in num]
    if not flat:
        return []
    probs = iter(export.predict(model_path, np.array(flat)[..., None]))
    out = []
    for ln in lines:
        for num in ln:
            ps = [next(probs) for _ in num]
            cls = [int(p.argmax()) for p in ps]
            # a junk box at either end ('+', stray mark) is dropped; junk inside a number is suspicious
            lo, hi = 0, len(cls)
            while lo < hi and cls[lo] == dataset.JUNK:
                lo += 1
            while hi > lo and cls[hi - 1] == dataset.JUNK:
                hi -= 1
            if lo == hi:
                continue
            conf = min(float(p.max()) for p in ps[lo:hi])
            if dataset.JUNK in cls[lo:hi]:
                conf = 0.0
            digits = "".join(str(c) for c in cls[lo:hi] if c != dataset.JUNK)
            out.append((int(digits), conf, num))
    return out


def total(readings, min_conf=MIN_CONF):
    """-> (sum, numbers, trusted) ; trusted is False if any number is below min_conf."""
    nums = [n for n, _, _ in readings if n is not None]
    ok = all(c >= min_conf for n, c, _ in readings if n is not None)
    return sum(nums), nums, ok


def load_gray(path):
    img = Image.open(path).convert("L")
    if img.size != (320, 240):
        img = img.resize((320, 240), Image.BOX)
    return np.asarray(img)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+")
    p.add_argument("--model", default="build/line_model_int8.tflite")
    p.add_argument("--digit-model", action="store_true", help="--model is the per-digit model")
    args = p.parse_args()
    for path in args.images:
        g = load_gray(path)
        r = (read_digits if args.digit_model else read_lines)(g, args.model)
        s, nums, ok = total(r)
        print(f"{path}: {' + '.join(map(str, nums))} = {s}" + ("" if ok else "   (LOW CONFIDENCE - check)"))
