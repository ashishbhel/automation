"""Check firmware/slip_reader/digit_pipeline.c against segment.py on synthetic slips.

    python test_c_port.py            # compiles with cc, runs 200 slips
"""
import ctypes
import os
import subprocess
import sys
import tempfile

import numpy as np

import dataset
import segment
import synth

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "firmware", "slip_reader", "digit_pipeline.c")


class Box(ctypes.Structure):
    _fields_ = [("x0", ctypes.c_int), ("y0", ctypes.c_int), ("x1", ctypes.c_int), ("y1", ctypes.c_int)]


def load_lib():
    out = os.path.join(tempfile.mkdtemp(), "libdp.so")
    subprocess.check_call(["cc", "-O2", "-std=c99", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                           SRC, "-o", out, "-lm"])
    lib = ctypes.CDLL(out)
    lib.dp_alloc.restype = ctypes.c_void_p
    lib.dp_alloc.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.dp_find_lines.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(Box), ctypes.c_int]
    lib.dp_line_image.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    lib.dp_ctc_decode.restype = ctypes.c_float
    lib.dp_ctc_decode.argtypes = [ctypes.POINTER(ctypes.c_int8), ctypes.c_int, ctypes.c_float, ctypes.c_int,
                                  ctypes.c_char_p, ctypes.c_int]
    return lib


def main(n=200):
    lib = load_lib()
    ws = lib.dp_alloc(synth.OUT_W, synth.OUT_H)
    gen = synth.SlipGenerator(synth.load_mnist("test"), seed=777)
    boxes = (Box * 16)()
    img = (ctypes.c_float * (segment.LINE_H * segment.LINE_W))()
    worst = 0.0
    n_lines = 0
    for i in range(n):
        gray, _, _ = gen()
        buf = np.ascontiguousarray(gray).tobytes()
        py_boxes, py_imgs = segment.extract_lines(gray)
        k = lib.dp_find_lines(ws, buf, boxes, 16)
        c_boxes = [(boxes[j].x0, boxes[j].y0, boxes[j].x1, boxes[j].y1) for j in range(k)]
        want = [(b["x0"], b["y0"], b["x1"], b["y1"]) for b in py_boxes]
        assert c_boxes == want, f"slip {i}: C boxes {c_boxes} != Python {want}"
        for j in range(k):
            lib.dp_line_image(ws, buf, j, img)
            diff = np.abs(np.ctypeslib.as_array(img).reshape(segment.LINE_H, segment.LINE_W) - py_imgs[j]).max()
            worst = max(worst, float(diff))
        n_lines += k
    assert worst < 1e-5, f"line images differ by up to {worst}"

    # CTC decode agrees with dataset.ctc_greedy
    rng = np.random.default_rng(0)
    out = ctypes.create_string_buffer(32)
    for _ in range(500):
        q = rng.integers(-128, 128, (24, 11)).astype(np.int8)
        q[rng.random(24) < 0.5, 10] = 127   # plenty of blanks
        conf = lib.dp_ctc_decode(q.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)), 24, 0.1, -3, out, 32)
        logits = q.astype(np.float64) * 0.1
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        text, pconf = dataset.ctc_greedy(p)
        assert out.value.decode() == text and abs(conf - pconf) < 1e-4, (out.value, text, conf, pconf)
    print(f"OK: {n} slips, {n_lines} lines identical boxes, max pixel diff {worst:.2e}; CTC decode matches")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
