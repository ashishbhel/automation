"""Train the line model (reads a whole handwritten number, CTC) and export int8 TFLite.

    python train_line.py --slips 20000 --epochs 20 --out build

The model sees one text line scaled to 24x96 and emits 24 time steps of
11-way scores (digits 0-9 + CTC blank). Touching or overlapping digits are
fine because nothing has to be cut into single digits.

Outputs in --out:
  line_model.keras         float model
  line_model_int8.tflite   full-integer model for TFLite Micro (ESP32)
  line_model_data.h        the .tflite as a C array for the firmware
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

T = segment.LINE_W // 4      # output time steps


def line_cnn(width=12):
    """~30k parameters, ~4M MACs per line at width=12."""
    L = tf.keras.layers

    def block(x, ch, k=3, pad="same"):
        x = L.Conv2D(ch, k, padding=pad, use_bias=False)(x)
        x = L.BatchNormalization()(x)
        return L.ReLU()(x)

    inp = L.Input((segment.LINE_H, segment.LINE_W, 1))
    x = block(inp, width)
    x = L.MaxPooling2D(2)(x)                        # 12 x 48
    x = block(x, width * 2)
    x = L.MaxPooling2D(2)(x)                        # 6 x 24
    x = block(x, width * 8 // 3)
    x = L.MaxPooling2D((2, 1))(x)                   # 3 x 24
    x = block(x, width * 4)
    x = block(x, width * 4, (3, 1), "valid")        # 1 x 24
    x = L.Dropout(0.2)(x)
    x = L.Conv2D(dataset.BLANK + 1, 1)(x)           # 1 x 24 x 11 logits
    out = L.Reshape((T, dataset.BLANK + 1))(x)
    return tf.keras.Model(inp, out)


def _worker(args):
    split, n, seed = args
    return dataset.build_lines(synth.SlipGenerator(synth.load_mnist(split), seed=seed), n)


def make_lines(split, n_slips, seed, procs):
    chunks = [(split, n_slips // procs + (i < n_slips % procs), seed * 1000 + i) for i in range(procs)]
    with Pool(procs) as p:
        parts = p.map(_worker, chunks)
    return np.concatenate([a for a, _ in parts]), [s for _, ls in parts for s in ls]


_shift = tf.keras.layers.RandomTranslation(0.08, 0.03, fill_mode="constant")


def augment(x):
    x = _shift(x, training=True)
    return tf.clip_by_value(x * tf.random.uniform([tf.shape(x)[0], 1, 1, 1], 0.7, 1.0), 0, 1)


class CTCModel(tf.keras.Model):
    """Wraps the line model with a CTC training step (labels padded with BLANK)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def call(self, x, training=False):
        return self.net(x, training=training)

    def _loss(self, x, y, n, training):
        logits = self.net(x, training=training)
        loss = tf.nn.ctc_loss(labels=y, logits=logits, label_length=n,
                              logit_length=tf.fill([tf.shape(x)[0]], T),
                              logits_time_major=False, blank_index=dataset.BLANK)
        return tf.reduce_mean(loss)

    def train_step(self, data):
        x, (y, n) = data
        with tf.GradientTape() as tape:
            loss = self._loss(augment(x), y, n, True)
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {"loss": loss}

    def test_step(self, data):
        x, (y, n) = data
        return {"loss": self._loss(x, y, n, False)}


def line_accuracy(probs, labels):
    return float(np.mean([dataset.ctc_greedy(p)[0] == s for p, s in zip(probs, labels)]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slips", type=int, default=20000, help="synthetic training slips (~4.5 lines each)")
    p.add_argument("--val-slips", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--width", type=int, default=12)
    p.add_argument("--real", help="npz with X [n,24,96,1] and labels [n] (str) from real captures, for fine-tuning")
    p.add_argument("--procs", type=int, default=os.cpu_count())
    p.add_argument("--out", default="build")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("building synthetic training lines ...", flush=True)
    X, labels = make_lines("train", args.slips, 1, args.procs)
    if args.real:
        r = np.load(args.real)
        reps = max(1, len(labels) // (4 * len(r["labels"])))   # real data ~20% of each epoch
        X = np.concatenate([X] + [r["X"]] * reps)
        labels = labels + [str(s) for s in r["labels"]] * reps
    print("building validation lines (MNIST test writers) ...", flush=True)
    Xv, lv = make_lines("test", args.val_slips, 2, args.procs)
    print(f"train {len(labels)} lines, val {len(lv)} lines, empty-label lines {sum(1 for s in labels if not s)}")

    y, n = dataset.encode(labels)
    yv, nv = dataset.encode(lv)
    net = line_cnn(args.width)
    net.summary()
    model = CTCModel(net)
    steps = int(np.ceil(len(labels) / 128)) * args.epochs
    model.compile(tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(3e-3, steps)))
    ds = tf.data.Dataset.from_tensor_slices((X, (y, n))).shuffle(len(y)).batch(128).prefetch(4)
    model.fit(ds, validation_data=(Xv, (yv, nv)), epochs=args.epochs, verbose=2)
    net.save(os.path.join(args.out, "line_model.keras"))

    probs = tf.nn.softmax(net.predict(Xv, verbose=0)).numpy()
    print(f"float model: val line accuracy {line_accuracy(probs, lv):.4f}")
    rng = np.random.default_rng(0)
    path, size = export.int8(net, X[rng.choice(len(X), 500, replace=False)], args.out, "line_model")
    print(f"int8 model: {size} bytes, val line accuracy {line_accuracy(export.predict(path, Xv), lv):.4f}")


if __name__ == "__main__":
    main()
