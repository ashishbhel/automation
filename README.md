# Handwritten-sum calculator on an ESP32 camera

A shopkeeper writes a column of whole numbers (1–4 digits each) on a small
slip, holds it under an ESP32 camera and presses a button. The device reads
the numbers and shows the total.

```
handwritten_sum/     Python: synthetic data, training, evaluation, int8 export
  synth.py           synthetic "slip photo" generator (unlimited labelled data)
  segment.py         classical front end: find text lines / digits (reference for the C port)
  dataset.py         turns slips into labelled line crops (and digit crops)
  train_line.py      line model (CTC) - recommended           -> build/line_model_int8.tflite
  train_digit.py     per-digit model - the absolute smallest   -> build/digit_model_int8.tflite
  evaluate.py        end-to-end "is the total right?" on unseen writers
  read_slip.py       run the full pipeline on any photo (phone pictures work)
  make_real_dataset.py  real labelled photos -> fine-tuning set
  test_c_port.py     checks the C front end is bit-exact with segment.py
firmware/slip_reader/
  slip_reader.ino    ESP32 sketch: camera -> lines -> TFLite Micro -> total
  digit_pipeline.c/h C port of the front end (no ESP deps, unit-tested on a PC)
```

RESULTS_PLACEHOLDER
