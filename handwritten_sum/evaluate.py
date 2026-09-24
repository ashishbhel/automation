"""End-to-end evaluation on synthetic slips written by unseen (MNIST test) writers.

    python evaluate.py --slips 1000

Reports, for each model found in --build:
  exact    : slips whose total is exactly right (no rejection)
  accepted : slips the device would show without asking for a re-check
  silent   : slips shown as trusted but WRONG  <- the number that matters for a shop
"""
import argparse
import os

import numpy as np

import read_slip
import synth


def run(reader, model, slips, thresholds):
    res = []
    for gray, numbers in slips:
        readings = reader(gray, model)
        got = [n for n, _, _ in readings if n is not None]
        confs = [c for n, c, _ in readings if n is not None]
        res.append((sum(got) == sum(numbers) and len(got) == len(numbers), min(confs, default=0.0)))
    correct = np.array([c for c, _ in res])
    conf = np.array([m for _, m in res])
    print(f"  exact total: {correct.mean():.3f}")
    for t in thresholds:
        acc = conf >= t
        print(f"  min_conf {t:.2f}: accepted {acc.mean():.3f}, silent errors {(acc & ~correct).mean():.4f}, "
              f"accuracy when accepted {correct[acc].mean() if acc.any() else float('nan'):.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slips", type=int, default=1000)
    p.add_argument("--build", default="build")
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()
    gen = synth.SlipGenerator(synth.load_mnist("test"), seed=args.seed)
    slips = []
    for _ in range(args.slips):
        g, nums, _ = gen()
        slips.append((g, nums))
    n_numbers = sum(len(n) for _, n in slips)
    n_digits = sum(len(str(x)) for _, n in slips for x in n)
    print(f"{args.slips} slips, {n_numbers} numbers, {n_digits} digits")
    th = [0.0, 0.5, 0.7, 0.9]
    line = os.path.join(args.build, "line_model_int8.tflite")
    digit = os.path.join(args.build, "digit_model_int8.tflite")
    if os.path.exists(line):
        print(f"line model ({os.path.getsize(line)} B):")
        run(read_slip.read_lines, line, slips, th)
    if os.path.exists(digit):
        print(f"digit model ({os.path.getsize(digit)} B):")
        run(read_slip.read_digits, digit, slips, th)


if __name__ == "__main__":
    main()
