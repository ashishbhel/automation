"""Turn real labelled slip photos into a fine-tuning set for train_line.py --real.

labels.csv, one row per photo, numbers top to bottom separated by spaces:

    photo_0001.jpg,120 45 7 3050
    photo_0002.jpg,99 1200

    python make_real_dataset.py labels.csv --images photos/ --out real_lines.npz

Photos should come from the SAME camera + lens + lighting as the product
(e.g. frames saved from the ESP32 itself); phone photos help less. A photo is
used only if the number of detected lines equals the number of labels, the
rest are listed so you can see where line finding fails.
"""
import argparse
import csv
import os

import numpy as np

import read_slip
import segment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("labels")
    p.add_argument("--images", default=".")
    p.add_argument("--out", default="real_lines.npz")
    args = p.parse_args()
    X, labels, skipped = [], [], []
    with open(args.labels) as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            name, nums = row[0], row[1].split()
            gray = read_slip.load_gray(os.path.join(args.images, name))
            _, imgs = segment.extract_lines(gray)
            if len(imgs) != len(nums):
                skipped.append((name, len(imgs), len(nums)))
                continue
            X += imgs
            labels += nums
    np.savez_compressed(args.out, X=np.array(X, np.float32)[..., None], labels=np.array(labels))
    print(f"{len(labels)} lines written to {args.out}; {len(skipped)} photos skipped")
    for name, got, want in skipped:
        print(f"  {name}: found {got} lines, label has {want}")


if __name__ == "__main__":
    main()
