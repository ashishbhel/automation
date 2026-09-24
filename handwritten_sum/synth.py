"""Synthetic "shop slip" generator.

Builds camera-like grayscale photos of a column of handwritten numbers by
pasting real handwritten digits (MNIST, optionally EMNIST or your own scans)
onto simulated paper, then degrading them like a cheap OV2640 would:
perspective, rotation, uneven lighting, blur, sensor noise, JPEG.

Every sample comes with exact ground truth (the numbers, their sum and a box
per digit), so labelling is free. Use MNIST *train* digits for training slips
and MNIST *test* digits for evaluation slips so the writers never overlap.
"""
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT_W, OUT_H = 320, 240      # QVGA, what the ESP32 grabs
SCALE = 2                    # render at 2x then downsample (anti-aliasing)


def load_mnist(split="train"):
    """{digit: [float32 ink crops tightly cropped, 0..1]} from keras' MNIST."""
    import tensorflow as tf
    (xtr, ytr), (xte, yte) = tf.keras.datasets.mnist.load_data()
    x, y = (xtr, ytr) if split == "train" else (xte, yte)
    return digits_by_class(x, y)


def digits_by_class(images, labels):
    out = {d: [] for d in range(10)}
    for img, lab in zip(images, labels):
        ys, xs = np.nonzero(img > 30)
        if len(ys) == 0:
            continue
        out[int(lab)].append(img[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32) / 255)
    return out


def random_numbers(rng, n_min=2, n_max=7):
    nums = []
    for _ in range(rng.integers(n_min, n_max + 1)):
        nd = rng.choice([1, 2, 3, 4], p=[0.15, 0.35, 0.3, 0.2])
        lo = 0 if nd == 1 else 10 ** (nd - 1)
        nums.append(int(rng.integers(lo, 10 ** nd)))
    return nums


class SlipGenerator:
    def __init__(self, digits, seed=0):
        self.digits = digits
        self.rng = np.random.default_rng(seed)

    def __call__(self, numbers=None):
        """-> (gray uint8 [OUT_H, OUT_W], numbers, digit_boxes)

        digit_boxes: [(digit, line_index, x0, y0, x1, y1)] in output pixels.
        """
        rng = self.rng
        numbers = numbers if numbers is not None else random_numbers(rng)
        W, H = OUT_W * SCALE, OUT_H * SCALE

        ink = np.zeros((H, W), np.float32)   # 0..1 pen coverage
        ids = np.zeros((H, W), np.uint8)     # which digit owns each pixel (for boxes)
        info = {}                            # id -> (digit, line)

        dh = rng.uniform(34, 70)             # digit height at render scale
        pitch = dh * rng.uniform(1.35, 2.0)
        n = len(numbers)
        while n * pitch + dh > H * 0.95 and dh > 24:
            dh *= 0.9
            pitch = dh * rng.uniform(1.3, 1.6)
        top = rng.uniform(0.02 * H, max(0.03 * H, H - n * pitch - dh * 0.2))
        right_align = rng.random() < 0.5
        col_x = rng.uniform(0.25 * W, 0.8 * W) if right_align else rng.uniform(0.05 * W, 0.45 * W)
        slant = rng.uniform(-0.15, 0.15)     # baseline drift per line

        next_id = 1
        for li, value in enumerate(numbers):
            glyphs = []
            for ch in str(value):
                d = int(ch)
                g = self.digits[d][rng.integers(len(self.digits[d]))]
                gh = dh * rng.uniform(0.85, 1.1)
                gw = max(2.0, g.shape[1] * gh / g.shape[0] * rng.uniform(0.8, 1.2))
                glyphs.append((d, _resize(g, int(gh), int(gw))))
            gaps = [dh * rng.uniform(-0.04, 0.3) for _ in glyphs]
            total = sum(g.shape[1] for _, g in glyphs) + sum(gaps[1:])
            x = col_x - total if right_align else col_x
            x += rng.normal(0, dh * 0.15)
            base = top + li * pitch + dh + x * slant * 0.1
            if rng.random() < 0.2:           # a "+" in front, like 2nd..nth lines on a bill
                _stroke_plus(ink, x - dh * 0.7, base - dh * 0.5, dh * 0.35, rng)
            for gi, (d, g) in enumerate(glyphs):
                if gi:
                    x += gaps[gi]
                y = base - g.shape[0] + rng.normal(0, dh * 0.04)
                _paste(ink, ids, g, int(x), int(y), next_id)
                info[next_id] = (d, li)
                next_id += 1
                x += g.shape[1]
        if rng.random() < 0.35:              # total line under the column
            y = top + n * pitch + dh * 0.1
            _stroke_line(ink, col_x - (dh * 3 if right_align else 0), y, dh * 3, rng)
        if rng.random() < 0.3:               # stray specks
            for _ in range(rng.integers(1, 6)):
                cx, cy = rng.uniform(0, W), rng.uniform(0, H)
                r = rng.uniform(1, 3)
                yy, xx = np.ogrid[:H, :W]
                ink[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = 1

        # pen thickness
        ink_img = Image.fromarray((ink * 255).astype(np.uint8))
        t = rng.random()
        if t < 0.3:
            ink_img = ink_img.filter(ImageFilter.MaxFilter(3))
        elif t < 0.4:
            ink_img = ink_img.filter(ImageFilter.MaxFilter(5))

        paper = np.asarray(_paper(rng, W, H), np.float32)
        if rng.random() < 0.35:              # ruled notebook paper
            rules = Image.new("L", (W, H), 0)
            d = ImageDraw.Draw(rules)
            sp = pitch * rng.uniform(0.9, 1.1)
            off = (top + dh * rng.uniform(0.9, 1.2)) % sp
            shade = int(rng.uniform(20, 70))
            for k in range(int(H / sp) + 2):
                yy = off + k * sp
                d.line([(0, yy), (W, yy + rng.normal(0, 2))], fill=shade, width=int(rng.integers(1, 4)))
            paper = paper - np.asarray(rules, np.float32)
        pen = rng.uniform(10, 95)            # black to blue/ballpoint in grayscale
        cov = np.asarray(ink_img, np.float32) / 255 * rng.uniform(0.75, 1.0)
        img = paper * (1 - cov) + pen * cov

        # camera geometry: small rotation + perspective, applied to image and id map together
        coeffs = _perspective(rng, W, H)
        img_p = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        img_p = img_p.transform((W, H), Image.PERSPECTIVE, coeffs, Image.BILINEAR, fillcolor=int(paper.mean()))
        ids_p = Image.fromarray(ids).transform((W, H), Image.PERSPECTIVE, coeffs, Image.NEAREST)

        # optics + sensor
        out = img_p.resize((OUT_W, OUT_H), Image.BOX)
        if rng.random() < 0.8:
            out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.1)))
        arr = np.asarray(out, np.float32) * _lighting(rng)
        arr += rng.normal(0, rng.uniform(1, 7), arr.shape)
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        buf = io.BytesIO()
        out.save(buf, "JPEG", quality=int(rng.uniform(30, 90)))
        gray = np.asarray(Image.open(buf).convert("L"))

        ids_small = np.asarray(ids_p)[::SCALE, ::SCALE]
        boxes = []
        for i, (d, li) in info.items():
            ys, xs = np.nonzero(ids_small == i)
            if len(ys):
                boxes.append((d, li, xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        return gray, numbers, boxes


def _resize(g, h, w):
    return np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((max(w, 1), max(h, 1)), Image.BILINEAR),
                      np.float32) / 255


def _paste(ink, ids, g, x, y, gid):
    H, W = ink.shape
    h, w = g.shape
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    sub = g[y0 - y:y1 - y, x0 - x:x1 - x]
    np.maximum(ink[y0:y1, x0:x1], sub, out=ink[y0:y1, x0:x1])
    ids[y0:y1, x0:x1][sub > 0.3] = gid


def _stroke_line(ink, x, y, length, rng):
    img = Image.fromarray((ink * 255).astype(np.uint8))
    ImageDraw.Draw(img).line([(x, y), (x + length, y + rng.normal(0, 3))], fill=255, width=int(rng.integers(2, 5)))
    ink[:] = np.asarray(img, np.float32) / 255


def _stroke_plus(ink, cx, cy, r, rng):
    img = Image.fromarray((ink * 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    wdt = int(rng.integers(2, 5))
    d.line([(cx - r, cy), (cx + r, cy + rng.normal(0, 2))], fill=255, width=wdt)
    d.line([(cx, cy - r), (cx + rng.normal(0, 2), cy + r)], fill=255, width=wdt)
    ink[:] = np.asarray(img, np.float32) / 255


def _paper(rng, W, H):
    base = rng.uniform(150, 245)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    grad = (xx / W - 0.5) * rng.normal(0, 25) + (yy / H - 0.5) * rng.normal(0, 25)
    low = np.asarray(Image.fromarray(rng.uniform(0, 255, (6, 8)).astype(np.uint8)).resize((W, H), Image.BICUBIC),
                     np.float32)
    tex = (low - 128) * rng.uniform(0, 0.12) + rng.normal(0, 3, (H, W))
    return Image.fromarray(np.clip(base + grad + tex, 0, 255).astype(np.uint8))


def _perspective(rng, W, H):
    """Coefficients mapping output -> input for PIL, from jittered corners + rotation."""
    a = np.deg2rad(rng.uniform(-5, 5))
    j = 0.04
    dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float64)
    c = np.array([W / 2, H / 2])
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    src = (dst - c) @ rot.T + c + rng.uniform(-j, j, (4, 2)) * [W, H]
    A, b = [], []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    return np.linalg.solve(np.array(A), np.array(b)).tolist()


def _lighting(rng):
    yy, xx = np.mgrid[0:OUT_H, 0:OUT_W].astype(np.float32)
    cx, cy = rng.uniform(0, OUT_W), rng.uniform(0, OUT_H)
    r2 = ((xx - cx) / OUT_W) ** 2 + ((yy - cy) / OUT_H) ** 2
    light = 1 - rng.uniform(0, 0.45) * r2 * rng.uniform(0.5, 2)
    if rng.random() < 0.25:            # a soft shadow band (hand / phone over the slip)
        nx, ny = rng.normal(size=2)
        side = (xx - cx) * nx + (yy - cy) * ny
        light *= 1 - rng.uniform(0.1, 0.4) / (1 + np.exp(-side / rng.uniform(5, 30)))
    return np.clip(light, 0.3, 1.1) * rng.uniform(0.75, 1.1)


if __name__ == "__main__":
    import argparse
    import os
    p = argparse.ArgumentParser(description="Write a few synthetic slips as PNG for eyeballing.")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--out", default="samples")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    gen = SlipGenerator(load_mnist("test"), seed=args.seed)
    for i in range(args.n):
        gray, nums, _ = gen()
        Image.fromarray(gray).save(os.path.join(args.out, f"slip_{i:02d}_sum{sum(nums)}.png"))
        print(i, nums, "=", sum(nums))
