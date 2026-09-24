// Classical front end for the handwritten-sum calculator (C port of
// handwritten_sum/segment.py, line-model path). Plain C99, no ESP-IDF
// dependencies, so it is unit-tested on a PC against the Python reference
// (see handwritten_sum/test_c_port.py).
//
// Usage:
//   dp_workspace_t *ws = dp_alloc(320, 240);           // once (~170 KB, PSRAM is fine)
//   int n = dp_find_lines(ws, gray, lines, DP_MAX_LINES);
//   for (i = 0; i < n; i++) dp_line_image(ws, gray, &lines[i], img);  // 24x96 floats in 0..1
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DP_LINE_H 24
#define DP_LINE_W 96
#define DP_LINE_INK_H 20
#define DP_LINE_PAD_X 2
#define DP_MAX_LINES 16

typedef struct {
  int x0, y0, x1, y1;  // x1, y1 exclusive
} dp_box_t;

typedef struct dp_workspace dp_workspace_t;

dp_workspace_t *dp_alloc(int width, int height);
void dp_free(dp_workspace_t *ws);

// Finds text lines in a grayscale frame (width*height bytes, row-major).
// Returns the number of lines written to `lines` (top to bottom).
int dp_find_lines(dp_workspace_t *ws, const uint8_t *gray, dp_box_t *lines, int max_lines);

// Renders line `index` (as returned by the last dp_find_lines) into a
// DP_LINE_H x DP_LINE_W float image with ink in 0..1 (model input before quantisation).
void dp_line_image(dp_workspace_t *ws, const uint8_t *gray, int index, float *out);

// Greedy CTC decode of [steps x 11] int8 logits (class 10 = blank).
// Writes the digits as a NUL-terminated string, returns the confidence
// (min over steps of the winning softmax probability).
float dp_ctc_decode(const int8_t *logits, int steps, float scale, int zero_point, char *out, int out_size);

#ifdef __cplusplus
}
#endif
