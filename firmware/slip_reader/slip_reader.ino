// Handwritten-sum calculator: press the button, the camera reads a column of
// handwritten numbers (one number per line, up to 4 digits each) and prints
// the total. Written for Arduino-ESP32 + the TensorFlow Lite Micro port
// (Library Manager: "TensorFlowLite_ESP32", or Espressif's esp-tflite-micro,
// which uses ESP-NN kernels - much faster on ESP32-S3).
//
// Board: AI-Thinker ESP32-CAM (default pins below) or an ESP32-S3 camera
// board (XIAO ESP32S3 Sense, ESP32-S3-EYE - change the pin map). PSRAM must
// be enabled (Tools > PSRAM > Enabled).
//
// Copy build/line_model_data.h (from handwritten_sum/train_line.py) next to
// this file before compiling.
//
// NOTE: this sketch is a starting point; the digit_pipeline.c front end is
// tested bit-exact against the Python reference, the camera/TFLM glue is not
// hardware-tested - TFLM API details differ slightly between library versions.
#include <Arduino.h>

#include "esp_camera.h"
#include "digit_pipeline.h"
#include "line_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ---- AI-Thinker ESP32-CAM pin map ----
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22
#define BUTTON_PIN 13     // to GND, internal pull-up
#define FLASH_LED_PIN 4   // on-board LED: even light helps a lot

static const float MIN_CONF = 0.6f;  // below this, ask the shopkeeper to check / retake

constexpr int kArenaSize = 60 * 1024;
static uint8_t *tensor_arena;
static tflite::MicroInterpreter *interpreter;
static dp_workspace_t *ws;
static float line_img[DP_LINE_H * DP_LINE_W];

static bool init_camera() {
  camera_config_t c = {};
  c.pin_pwdn = PWDN_GPIO_NUM;
  c.pin_reset = RESET_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM;
  c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_d7 = Y9_GPIO_NUM;
  c.pin_d6 = Y8_GPIO_NUM;
  c.pin_d5 = Y7_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM;
  c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d2 = Y4_GPIO_NUM;
  c.pin_d1 = Y3_GPIO_NUM;
  c.pin_d0 = Y2_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM;
  c.pin_href = HREF_GPIO_NUM;
  c.pin_pclk = PCLK_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.ledc_timer = LEDC_TIMER_0;
  c.ledc_channel = LEDC_CHANNEL_0;
  c.pixel_format = PIXFORMAT_GRAYSCALE;  // the pipeline works on 8-bit gray
  c.frame_size = FRAMESIZE_QVGA;         // 320x240, same as training
  c.fb_count = 1;
  c.fb_location = CAMERA_FB_IN_PSRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;
  return esp_camera_init(&c) == ESP_OK;
}

static bool init_model() {
  const tflite::Model *model = tflite::GetModel(g_line_model);
  static tflite::MicroMutableOpResolver<4> resolver;
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddReshape();
  tensor_arena = (uint8_t *)heap_caps_malloc(kArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!tensor_arena) return false;
  interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, kArenaSize);
  if (interpreter->AllocateTensors() != kTfLiteOk) return false;
  Serial.printf("model %u bytes, arena used %u bytes\n", g_line_model_len,
                (unsigned)interpreter->arena_used_bytes());
  return true;
}

// Reads one frame; returns false if any line is uncertain.
static bool read_slip(long *total) {
  digitalWrite(FLASH_LED_PIN, HIGH);
  delay(150);                        // let exposure settle
  esp_camera_fb_return(esp_camera_fb_get());  // drop a stale frame
  camera_fb_t *fb = esp_camera_fb_get();
  digitalWrite(FLASH_LED_PIN, LOW);
  if (!fb) return false;

  uint32_t t0 = millis();
  dp_box_t lines[DP_MAX_LINES];
  int n = dp_find_lines(ws, fb->buf, lines, DP_MAX_LINES);
  TfLiteTensor *in = interpreter->input(0), *out = interpreter->output(0);
  const float in_scale = in->params.scale;
  const int in_zp = in->params.zero_point;
  bool trusted = true;
  *total = 0;
  for (int i = 0; i < n; i++) {
    dp_line_image(ws, fb->buf, i, line_img);
    for (int p = 0; p < DP_LINE_H * DP_LINE_W; p++) {
      int q = (int)lroundf(line_img[p] / in_scale) + in_zp;
      in->data.int8[p] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
    }
    if (interpreter->Invoke() != kTfLiteOk) continue;
    char digits[8];
    float conf = dp_ctc_decode(out->data.int8, out->dims->data[1], out->params.scale,
                               out->params.zero_point, digits, sizeof(digits));
    if (!digits[0]) continue;        // '+', underline, smudge: no digits
    long v = atol(digits);
    *total += v;
    if (conf < MIN_CONF) trusted = false;
    Serial.printf("  %c %6ld   (conf %.2f)\n", i ? '+' : ' ', v, conf);
  }
  esp_camera_fb_return(fb);
  Serial.printf("  = %ld   [%lu ms]%s\n", *total, (unsigned long)(millis() - t0),
                trusted ? "" : "  CHECK: low confidence");
  return trusted;
}

void setup() {
  Serial.begin(115200);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(FLASH_LED_PIN, OUTPUT);
  ws = dp_alloc(320, 240);
  if (!ws || !init_camera() || !init_model()) {
    Serial.println("init failed");
    while (true) delay(1000);
  }
  Serial.println("ready - press the button");
}

void loop() {
  if (digitalRead(BUTTON_PIN) == LOW) {
    long total;
    read_slip(&total);
    while (digitalRead(BUTTON_PIN) == LOW) delay(10);
  }
  delay(10);
}
