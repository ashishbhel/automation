// C port of handwritten_sum/segment.py (line-model path). Keep the two in sync:
// handwritten_sum/test_c_port.py checks this file against the Python reference.
#include "digit_pipeline.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define BG_RADIUS 15
#define K_PERCENT 15
#define MIN_CONTRAST 12
#define MIN_AREA 6
#define LONG_RUN_FRAC 4
#define MAX_COMPS 1024

typedef struct {
  int16_t y, s, e;  // run [s, e) on row y
} run_t;

typedef struct {
  int x0, y0, x1, y1, area, line;
} comp_t;

struct dp_workspace {
  int w, h;
  uint8_t *bg;       // background estimate
  uint8_t *crop;     // scratch for one line's ink
  run_t *runs;
  int32_t *parent;   // union-find over runs
  int16_t *run_comp; // run -> component index (-1 = none)
  int max_runs, n_runs;
  comp_t comps[MAX_COMPS];
  int n_comps;
  int kept[MAX_COMPS];
  dp_box_t lines[DP_MAX_LINES];
  int n_lines;
};

dp_workspace_t *dp_alloc(int w, int h) {
  dp_workspace_t *ws = (dp_workspace_t *)calloc(1, sizeof(dp_workspace_t));
  if (!ws) return NULL;
  ws->w = w;
  ws->h = h;
  ws->max_runs = w * h / 8;
  ws->bg = (uint8_t *)malloc((size_t)w * h);
  ws->crop = (uint8_t *)malloc((size_t)w * h);
  ws->runs = (run_t *)malloc(sizeof(run_t) * ws->max_runs);
  ws->parent = (int32_t *)malloc(sizeof(int32_t) * ws->max_runs);
  ws->run_comp = (int16_t *)malloc(sizeof(int16_t) * ws->max_runs);
  if (!ws->bg || !ws->crop || !ws->runs || !ws->parent || !ws->run_comp) {
    dp_free(ws);
    return NULL;
  }
  return ws;
}

void dp_free(dp_workspace_t *ws) {
  if (!ws) return;
  free(ws->bg);
  free(ws->crop);
  free(ws->runs);
  free(ws->parent);
  free(ws->run_comp);
  free(ws);
}

// Box mean over a clamped (2R+1)^2 window, floor division (== segment.background).
static void background(dp_workspace_t *ws, const uint8_t *g) {
  const int w = ws->w, h = ws->h, r = BG_RADIUS;
  uint32_t *col = (uint32_t *)calloc(w, sizeof(uint32_t));  // vertical window sums
  if (!col) return;
  int top = 0, bot = 0;  // rows [top, bot) are in col
  for (int y = 0; y < h; y++) {
    int y0 = y - r < 0 ? 0 : y - r, y1 = y + r + 1 > h ? h : y + r + 1;
    while (bot < y1) {
      for (int x = 0; x < w; x++) col[x] += g[bot * w + x];
      bot++;
    }
    while (top < y0) {
      for (int x = 0; x < w; x++) col[x] -= g[top * w + x];
      top++;
    }
    uint32_t s = 0;
    int left = 0, right = 0;
    for (int x = 0; x < w; x++) {
      int x0 = x - r < 0 ? 0 : x - r, x1 = x + r + 1 > w ? w : x + r + 1;
      while (right < x1) s += col[right++];
      while (left < x0) s -= col[left++];
      ws->bg[y * w + x] = (uint8_t)(s / (uint32_t)((y1 - y0) * (x1 - x0)));
    }
  }
  free(col);
}

static inline int is_ink(int g, int bg) {
  return g * 100 < bg * (100 - K_PERCENT) && bg - g > MIN_CONTRAST;
}

static int32_t find(int32_t *p, int32_t a) {
  while (p[a] != a) {
    p[a] = p[p[a]];
    a = p[a];
  }
  return a;
}

// Ink runs per row (long ruled-line runs dropped) + 8-connected union-find.
static void label(dp_workspace_t *ws, const uint8_t *g) {
  const int w = ws->w, h = ws->h, max_len = w / LONG_RUN_FRAC;
  int n = 0, prev_start = 0, prev_end = 0;
  for (int y = 0; y < h; y++) {
    const uint8_t *row = g + y * w, *bgr = ws->bg + y * w;
    int cur_start = n, j = prev_start;
    int x = 0;
    while (x < w) {
      if (!is_ink(row[x], bgr[x])) {
        x++;
        continue;
      }
      int s = x;
      while (x < w && is_ink(row[x], bgr[x])) x++;
      int e = x;
      if (e - s > max_len || n >= ws->max_runs) continue;
      ws->runs[n].y = (int16_t)y;
      ws->runs[n].s = (int16_t)s;
      ws->runs[n].e = (int16_t)e;
      ws->parent[n] = n;
      // previous-row run [ps, pe) touches [s, e) if ps <= e && pe >= s
      while (j < prev_end && ws->runs[j].e < s) j++;
      int k = j;
      while (k < prev_end && ws->runs[k].s <= e) {
        int32_t a = find(ws->parent, n), b = find(ws->parent, k);
        if (a != b) ws->parent[a > b ? a : b] = a < b ? a : b;
        k++;
      }
      if (k > j) j = k - 1;
      n++;
    }
    prev_start = cur_start;
    prev_end = n;
  }
  ws->n_runs = n;

  // components in order of their root run (== order of first appearance)
  ws->n_comps = 0;
  for (int i = 0; i < n; i++) {
    int32_t root = find(ws->parent, i);
    const run_t *r = &ws->runs[i];
    if (root == i) {
      if (ws->n_comps >= MAX_COMPS) {
        ws->run_comp[i] = -1;
        continue;
      }
      comp_t *c = &ws->comps[ws->n_comps];
      c->x0 = r->s;
      c->x1 = r->e;
      c->y0 = r->y;
      c->y1 = r->y + 1;
      c->area = 0;
      c->line = -1;
      ws->run_comp[i] = (int16_t)ws->n_comps++;
    } else {
      ws->run_comp[i] = ws->run_comp[root];
    }
    int ci = ws->run_comp[i];
    if (ci < 0) continue;
    comp_t *c = &ws->comps[ci];
    if (r->s < c->x0) c->x0 = r->s;
    if (r->e > c->x1) c->x1 = r->e;
    c->y1 = r->y + 1;
    c->area += r->e - r->s;
  }
}

static int cmp_int(const void *a, const void *b) {
  return *(const int *)a - *(const int *)b;
}

static const comp_t *g_sort_comps;  // qsort context (single-threaded)

static int cmp_kept(const void *a, const void *b) {
  const comp_t *p = &g_sort_comps[*(const int *)a], *q = &g_sort_comps[*(const int *)b];
  int ka = p->y0 + p->y1, kb = q->y0 + q->y1;
  if (ka != kb) return ka - kb;
  if (p->x0 != q->x0) return p->x0 - q->x0;
  return *(const int *)a - *(const int *)b;  // stable, like Python's sort
}

int dp_find_lines(dp_workspace_t *ws, const uint8_t *gray, dp_box_t *out, int max_lines) {
  background(ws, gray);
  label(ws, gray);
  ws->n_lines = 0;

  // typical digit height = median height of components >= MIN_AREA and >= 6 px tall
  static int heights[MAX_COMPS];
  int nh = 0, any = 0;
  for (int i = 0; i < ws->n_comps; i++) {
    comp_t *c = &ws->comps[i];
    if (c->area < MIN_AREA) continue;
    any = 1;
    if (c->y1 - c->y0 >= 6) heights[nh++] = c->y1 - c->y0;
  }
  if (!any) return 0;
  qsort(heights, nh, sizeof(int), cmp_int);
  const int H = nh ? heights[nh / 2] : 8;

  int nk = 0;
  for (int i = 0; i < ws->n_comps; i++) {
    comp_t *c = &ws->comps[i];
    int h = c->y1 - c->y0, w = c->x1 - c->x0;
    if (c->area < MIN_AREA) continue;
    if (h < 0.3 * H && w < 0.3 * H) continue;  // specks
    if (w > 2.5 * h && h < 0.5 * H) continue;  // dashes, underlines
    if (h > 2.5 * H || w > 6 * H) continue;    // borders, scribbles
    ws->kept[nk++] = i;
  }
  g_sort_comps = ws->comps;
  qsort(ws->kept, nk, sizeof(int), cmp_kept);

  // cluster into lines by vertical centre (first line whose running mean is within 0.6 H)
  double sum[DP_MAX_LINES];
  int cnt[DP_MAX_LINES], nl = 0;
  for (int k = 0; k < nk; k++) {
    comp_t *c = &ws->comps[ws->kept[k]];
    double cy = (c->y0 + c->y1) / 2.0;
    int l = 0;
    for (; l < nl; l++)
      if (fabs(cy - sum[l] / cnt[l]) < 0.6 * H) break;
    if (l == nl) {
      if (nl == DP_MAX_LINES) continue;
      sum[nl] = 0;
      cnt[nl] = 0;
      ws->lines[nl].x0 = c->x0;
      ws->lines[nl].y0 = c->y0;
      ws->lines[nl].x1 = c->x1;
      ws->lines[nl].y1 = c->y1;
      nl++;
    }
    sum[l] += cy;
    cnt[l]++;
    c->line = l;
    dp_box_t *b = &ws->lines[l];
    if (c->x0 < b->x0) b->x0 = c->x0;
    if (c->y0 < b->y0) b->y0 = c->y0;
    if (c->x1 > b->x1) b->x1 = c->x1;
    if (c->y1 > b->y1) b->y1 = c->y1;
  }

  // sort lines top to bottom (stable insertion sort), remap component line ids
  int order[DP_MAX_LINES], rank[DP_MAX_LINES];
  for (int i = 0; i < nl; i++) order[i] = i;
  for (int i = 1; i < nl; i++) {
    int v = order[i], j = i - 1;
    while (j >= 0 && sum[order[j]] / cnt[order[j]] > sum[v] / cnt[v]) {
      order[j + 1] = order[j];
      j--;
    }
    order[j + 1] = v;
  }
  dp_box_t sorted[DP_MAX_LINES];
  for (int i = 0; i < nl; i++) {
    rank[order[i]] = i;
    sorted[i] = ws->lines[order[i]];
  }
  for (int i = 0; i < ws->n_comps; i++)
    if (ws->comps[i].line >= 0) ws->comps[i].line = rank[ws->comps[i].line];
  memcpy(ws->lines, sorted, sizeof(dp_box_t) * nl);
  ws->n_lines = nl;

  int n = nl < max_lines ? nl : max_lines;
  memcpy(out, ws->lines, sizeof(dp_box_t) * n);
  return n;
}

void dp_line_image(dp_workspace_t *ws, const uint8_t *gray, int index, float *out) {
  memset(out, 0, sizeof(float) * DP_LINE_H * DP_LINE_W);
  if (index < 0 || index >= ws->n_lines) return;
  const dp_box_t *b = &ws->lines[index];
  const int w = b->x1 - b->x0, h = b->y1 - b->y0;
  uint8_t *crop = ws->crop;
  memset(crop, 0, (size_t)w * h);

  // ink (bg - gray) of this line's components only
  int peak = 0;
  for (int i = 0; i < ws->n_runs; i++) {
    int ci = ws->run_comp[i];
    if (ci < 0 || ws->comps[ci].line != index) continue;
    const run_t *r = &ws->runs[i];
    for (int x = r->s; x < r->e; x++) {
      int p = r->y * ws->w + x;
      int d = ws->bg[p] - gray[p];
      if (d < 0) d = 0;
      crop[(r->y - b->y0) * w + (x - b->x0)] = (uint8_t)d;
      if (d > peak) peak = d;
    }
  }
  if (peak == 0) return;

  const double scale = fmin((double)DP_LINE_INK_H / h, (double)(DP_LINE_W - 2 * DP_LINE_PAD_X) / w);
  int th = (int)rint(h * scale), tw = (int)rint(w * scale);
  if (th < 1) th = 1;
  if (tw < 1) tw = 1;
  const int oy = (DP_LINE_H - th) / 2;
  const float fpeak = (float)peak;

  // supersampled (3x3) bilinear resize == segment.resize
  enum { S = 3 };
  for (int y = 0; y < th; y++) {
    for (int x = 0; x < tw; x++) {
      double acc = 0;
      for (int sy = 0; sy < S; sy++) {
        double fy = ((y + (sy + 0.5) / S) * h) / th - 0.5;
        fy = fy < 0 ? 0 : (fy > h - 1 ? h - 1 : fy);
        int y0 = (int)fy, y1 = y0 + 1 < h ? y0 + 1 : h - 1;
        double dy = fy - y0;
        for (int sx = 0; sx < S; sx++) {
          double fx = ((x + (sx + 0.5) / S) * w) / tw - 0.5;
          fx = fx < 0 ? 0 : (fx > w - 1 ? w - 1 : fx);
          int x0 = (int)fx, x1 = x0 + 1 < w ? x0 + 1 : w - 1;
          double dx = fx - x0;
          double c00 = (float)crop[y0 * w + x0] / fpeak, c01 = (float)crop[y0 * w + x1] / fpeak;
          double c10 = (float)crop[y1 * w + x0] / fpeak, c11 = (float)crop[y1 * w + x1] / fpeak;
          double top = c00 * (1 - dx) + c01 * dx, bot = c10 * (1 - dx) + c11 * dx;
          acc += top * (1 - dy) + bot * dy;
        }
      }
      out[(oy + y) * DP_LINE_W + DP_LINE_PAD_X + x] = (float)(acc / (S * S));
    }
  }
}

float dp_ctc_decode(const int8_t *logits, int steps, float scale, int zero_point, char *out, int out_size) {
  const int C = 11, BLANK = 10;
  int prev = BLANK, n = 0;
  float conf = 1.0f;
  for (int t = 0; t < steps; t++) {
    const int8_t *l = logits + t * C;
    int best = 0;
    for (int c = 1; c < C; c++)
      if (l[c] > l[best]) best = c;
    float denom = 0;
    for (int c = 0; c < C; c++) denom += expf((l[c] - l[best]) * scale);
    float p = 1.0f / denom;
    if (p < conf) conf = p;
    if (best != prev && best != BLANK && n < out_size - 1) out[n++] = (char)('0' + best);
    prev = best;
  }
  out[n] = 0;
  (void)zero_point;  // cancels out in the softmax
  return conf;
}
