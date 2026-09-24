"""Train the tiny digit classifier and export an int8 TFLite model + C array.

    python train_digit.py --slips 6000 --epochs 15 --out build

Outputs in --out:
  digit_model.keras        float model
  digit_model_int8.tflite  full-integer model for TFLite Micro (ESP32)
  digit_model_data.h       the .tflite as a C array for the firmware
  val_crops.npz            held-out crops (writers never seen in training)
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import tensorflow as tf

import dataset
import export
import segment
import synth


def tiny_cnn(width=8):
    """~9k parameters, ~0.2M MACs per digit at width=8."""
    L = tf.keras.layers
    return tf.keras.Sequential([
        L.Input((segment.DIGIT_SIZE, segment.DIGIT_SIZE, 1)),
        L.Conv2D(width, 3, padding="same", activation="relu"),
        L.MaxPooling2D(),                                          # 10x10
        L.Conv2D(width * 2, 3, padding="same", activation="relu"),
        L.MaxPooling2D(),                                          # 5x5
        L.Conv2D(width * 4, 3, activation="relu"),                 # 3x3
        L.Flatten(),
        L.Dropout(0.25),
        L.Dense(dataset.NUM_CLASSES),
    ])


def _worker(args):
    split, n, seed = args
    return dataset.build(synth.SlipGenerator(synth.load_mnist(split), seed=seed), n, log_every=0)


def make_crops(split, n_slips, seed, procs):
    chunks = [(split, n_slips // procs + (i < n_slips % procs), seed * 1000 + i) for i in range(procs)]
    with Pool(procs) as p:
        parts = p.map(_worker, chunks)
    return np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts])


_aug = [tf.keras.layers.RandomTranslation(0.08, 0.08, fill_mode="constant"),
        tf.keras.layers.RandomRotation(0.03, fill_mode="constant"),
        tf.keras.layers.RandomZoom(0.1, fill_mode="constant")]


def augment(x):
    for layer in _aug:
        x = layer(x, training=True)
    return tf.clip_by_value(x * tf.random.uniform([tf.shape(x)[0], 1, 1, 1], 0.7, 1.0), 0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slips", type=int, default=6000, help="synthetic training slips")
    p.add_argument("--val-slips", type=int, default=600)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--no-mnist", action="store_true", help="don't add raw MNIST train digits")
    p.add_argument("--real", help="npz with X [n,20,20,1] and y [n] from real ESP32 captures (fine-tuning)")
    p.add_argument("--procs", type=int, default=os.cpu_count())
    p.add_argument("--out", default="build")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("building synthetic training crops ...", flush=True)
    X, y = make_crops("train", args.slips, 1, args.procs)
    if not args.no_mnist:
        (xm, ym), _ = tf.keras.datasets.mnist.load_data()
        with Pool(args.procs) as pool:
            parts = pool.starmap(dataset.mnist_crops, [(xm[i::args.procs], ym[i::args.procs]) for i in range(args.procs)])
        X = np.concatenate([X] + [np.array(a, np.float32)[..., None] for a, _ in parts])
        y = np.concatenate([y] + [np.array(b) for _, b in parts])
    if args.real:
        r = np.load(args.real)
        reps = max(1, len(y) // (4 * len(r["y"])))    # real data ~20% of each epoch
        X = np.concatenate([X] + [r["X"]] * reps)
        y = np.concatenate([y] + [r["y"]] * reps)
    print("building validation crops (MNIST test writers) ...", flush=True)
    Xv, yv = make_crops("test", args.val_slips, 2, args.procs)
    np.savez_compressed(os.path.join(args.out, "val_crops.npz"), X=Xv, y=yv)
    print(f"train {len(y)} crops, val {len(yv)} crops, class counts {np.bincount(y, minlength=11).tolist()}")

    model = tiny_cnn(args.width)
    model.summary()
    steps = int(np.ceil(len(y) / 256)) * args.epochs
    model.compile(tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(3e-3, steps)),
                  tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    ds = (tf.data.Dataset.from_tensor_slices((X, y)).shuffle(len(y)).batch(256)
          .map(lambda a, b: (augment(a), b), num_parallel_calls=tf.data.AUTOTUNE).prefetch(4))
    model.fit(ds, validation_data=(Xv, yv), epochs=args.epochs, verbose=2)
    model.save(os.path.join(args.out, "digit_model.keras"))

    rng = np.random.default_rng(0)
    path, size = export.int8(model, X[rng.choice(len(X), 500, replace=False)], args.out, "digit_model")
    acc = float((export.predict(path, Xv).argmax(1) == yv).mean())
    print(f"int8 model: {size} bytes, val accuracy {acc:.4f}")


if __name__ == "__main__":
    main()
