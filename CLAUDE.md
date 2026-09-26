# Project context: handwritten-sum calculator on ESP32

Hand-off notes from the first (cloud) session, for continuing locally with the
ESP32 plugged into this PC. Read this first.

## Goal
A shopkeeper writes a column of whole numbers (1–4 digits, no decimals) on a
small slip. A small camera + ESP32 reads them and shows each number and the
total. The model must be as small as possible and run on a microcontroller.

## How the owner likes to work
- Discuss feasibility and options before writing code; don't start big
  implementations without asking.
- Keep answers short and practical. They can press BOOT/EN on the board when an
  upload needs it.

## Decisions so far
- **Architecture: option B, the line model.** Classical code finds each
  handwritten line → a small CNN reads the whole number (CTC decoding, 11
  classes = digits + blank) → plain code adds the numbers. The network never
  does the arithmetic.
  - Option A (cut out single digits + ~9k-parameter classifier, ~10–15 KB) is
    kept as the "smallest" variant, but finding single digits was weak:
    **~93% digit recall, all digits found on only ~59% of slips**, mostly
    because touching digits ("11", "17") can't be split reliably.
  - Line finding measured **635/636 numbers fully captured** on synthetic slips.
  - Option C (image → total directly) was rejected: needs huge data, no way to
    check or flag errors.
- **Model size:** line CNN ~31.5k parameters, int8 TFLite = **42.8 KB**
  (measured). Estimated ~4M multiply-adds per line; ESP32 speed not yet measured.
- **Hardware:** target an ESP32-S3 camera board with PSRAM (N8R8 / N16R8, e.g.
  XIAO ESP32S3 Sense). An SD card is not needed to run; it's only useful for
  collecting real photos. **The owner currently has an ESP32-WROOM-32 DevKit**
  (no camera, usually no PSRAM). Use it to test inference speed and accuracy
  by sending line images over USB serial, not to test the full camera pipeline.
- **Product advice given:** confirm each number on a screen, refuse to answer
  when confidence is low (errors compound: 99% per digit ≈ 86% of 15-digit
  slips right), use even lighting and a fixed stand, printed slips with
  lines/boxes help a lot, and a phone app is the cheapest first version.

## Data
- **No training data is stored; it is generated on the fly.** `synth.py` pastes
  MNIST digits (MNIST train for training, MNIST test for validation, so the
  writers never overlap) onto simulated paper with ruled lines, "+" signs,
  underlines, perspective, blur, noise, shadows and JPEG. Unlimited and
  perfectly labelled. Preview it with `python synth.py --n 12 --out samples`.
- **Real data still needed** (plan): 30–50+ writers, ~2k–5k digits photographed
  **with the device camera**, plus 200–300 held-out real slips from other writers
  for testing. Label format for `make_real_dataset.py`: `photo.jpg,120 45 7 3050`.
- Handwriting fonts couldn't be downloaded in the cloud container; EMNIST is not
  wired in yet (optional extra source).

## Trained model status: NOT USABLE YET
- A 2-epoch smoke test proved the pipeline runs end to end (train → int8 export
  → 42.8 KB), but its line accuracy was 2.6%, i.e. it did not learn.
- The full run (20k slips ≈ 90k lines, 20 epochs) printed
  `loss: 0.0000e+00 - val_loss: 0.0000e+00` every epoch. **That is a bug, not
  a perfect model.** The run was stopped and no model files were kept.
- **First task: fix `handwritten_sum/train_line.py`.** Suspect the custom
  `CTCModel.train_step`/`test_step` returning a plain dict under Keras 3 (the
  logged loss may not be the computed one), or `tf.nn.ctc_loss` arguments
  (dense labels padded with BLANK=10, `blank_index=10`,
  `logits_time_major=False`). Check it by computing the CTC loss by hand on one
  batch outside `fit()`. If needed, rewrite it as a plain `tf.GradientTape`
  loop. Train small first (`--slips 400 --epochs 2`) and confirm the loss drops
  and line accuracy rises before running the full job (~1 h on 4 CPU cores).

## Code map (branch `claude/handwritten-sum-esp32-model-8ail1r`)
```
handwritten_sum/
  synth.py            synthetic slip generator (320x240 gray, like an OV2640 at QVGA)
  segment.py          classical front end: background/threshold, ruled-line removal,
                      8-connected components, line clustering; normalises lines
                      to 24x96 (line model) or digits to 20x20 (digit model)
  dataset.py          slips -> labelled line crops / digit crops; CTC greedy decode
  train_line.py       line model (CTC) + int8 export     <- has the loss bug
  train_digit.py      per-digit model (11 classes incl. "not a digit") + int8 export
  export.py           int8 TFLite export + writes <name>_data.h C array; reference predict
  evaluate.py         end-to-end: exact-total rate, acceptance rate, silent-error rate
  read_slip.py        run the pipeline on any photo: python read_slip.py img.jpg
  make_real_dataset.py  labelled real photos -> npz for train_line.py --real
  test_c_port.py      compiles the C front end and checks it against segment.py
firmware/slip_reader/
  digit_pipeline.c/h  C port of the line front end + CTC decode, no ESP deps
  slip_reader.ino     ESP32 camera + TFLite Micro sketch (AI-Thinker pin map,
                      QVGA grayscale, button on GPIO13) - NOT hardware-tested
```
- Setup: `pip install -r handwritten_sum/requirements.txt` (numpy, pillow,
  tensorflow-cpu). Run the scripts from inside `handwritten_sum/`.
- **Verified:** `python test_c_port.py` → the C and Python front ends give
  identical line boxes on 445 lines, max pixel difference 6e-8, and CTC decode
  matches.
- The pipeline uses ~270 KB of working buffers (PSRAM on ESP32-CAM/S3). A WROOM
  without PSRAM needs a slimmer version for full frames; model-only tests are fine.

## Suggested next steps (local)
1. Hardware smoke test: install `arduino-cli` + the ESP32 core, find the port,
   flash a blink + chip-info sketch (GPIO2 LED, print flash/heap/PSRAM at 115200).
2. Fix the `train_line.py` loss bug, retrain, run `evaluate.py`, then commit
   `build/line_model_int8.tflite` and `build/line_model_data.h` (.gitignore allows these).
3. WROOM model test: a sketch with the model + a serial protocol (the PC sends
   24x96 int8 line images, the ESP32 returns digits, confidence and ms per line),
   with a PC script that sends synthetic lines and compares against `export.predict`.
4. Test on real phone photos with `read_slip.py`, then collect real device data.

## Open questions for the owner
- Numerals: Western 0–9, Devanagari/Gujarati, or both?
- Free-form paper, or can shops get printed slips with lines or boxes?
- Final board: ESP32-S3 camera board (recommended) vs ESP32-CAM?
