
/*// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cmath>

#include "libs/base/filesystem.h"
#include "libs/base/led.h"
#include "libs/camera/camera.h"
#include "libs/tensorflow/posenet.h"
#include "libs/tensorflow/posenet_decoder_op.h"
#include "libs/tpu/edgetpu_manager.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/all_ops_resolver.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"

// Performs pose estimation with camera images (using the PoseNet model),
// running on the Edge TPU. Scores and keypoint data is printed to the serial
// console.
//
// To build and flash from coralmicro root:
//    bash build.sh
//    python3 scripts/flashtool.py -e tracker

// [start-sphinx-snippet:tracker]
namespace coralmicro {
namespace {

  struct PersonLocation {
    float angle_deg;    // horizontal angle from center (- = left, + = right)
    float distance_m;   // estimated distance in meters (approximate)
  };
  PersonLocation EstimatePersonLocation(const tensorflow::Pose& pose,
                                         int image_width, int image_height) {
    // Centroid from keypoints with score > 0.2
    float cx = 0, cy = 0;
    int n = 0;
    for (int j = 0; j < tensorflow::kKeypoints; ++j) {
      if (pose.keypoints[j].score > 0.2f) {
        cx += pose.keypoints[j].x;
        cy += pose.keypoints[j].y;
        ++n;
      }
    }
    if (n == 0) return {0, 0};
    cx /= n; cy /= n;
    // Angle (horizontal)
    float norm_x = (cx - image_width / 2.0f) / (image_width / 2.0f);
    float angle_deg = norm_x * 50.0f;  // ~100° FOV → ±50° from center
    // Distance from pose height (NOSE=0, LEFT_ANKLE=15, RIGHT_ANKLE=16)
    float head_y = pose.keypoints[0].y;
    float foot_y = pose.keypoints[15].score > 0.2f ? pose.keypoints[15].y
                : pose.keypoints[16].y;
    float height_px = fabs(foot_y - head_y);
    float focal_px = image_width / (2.0f * tanf(50.0f * 3.14159f / 180.0f));
    float distance_m = (1.7f * focal_px) / fmaxf(height_px, 10.0f);
    return {angle_deg, distance_m};
  }
  
constexpr int kModelArenaSize = 1 * 1024 * 1024;
constexpr int kExtraArenaSize = 1 * 1024 * 1024;
constexpr int kTensorArenaSize = kModelArenaSize + kExtraArenaSize;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);
constexpr char kModelPath[] =
    "/models/posenet_mobilenet_v1_075_324_324_16_quant_decoder_edgetpu.tflite";
constexpr char kTestInputPath[] = "/models/posenet_test_input_324.bin";

void Main() {
  printf("Posenet Example!\r\n");
  // Turn on Status LED to show the board is on.
  LedSet(Led::kStatus, true);

  tflite::MicroErrorReporter error_reporter;
  TF_LITE_REPORT_ERROR(&error_reporter, "Posenet!");
  // Turn on the TPU and get it's context.
  auto tpu_context =
      EdgeTpuManager::GetSingleton()->OpenDevice(PerformanceMode::kMax);
  if (!tpu_context) {
    printf("ERROR: Failed to get EdgeTpu context\r\n");
    vTaskSuspend(nullptr);
  }
  // Reads the model and checks version.
  std::vector<uint8_t> posenet_tflite;
  if (!LfsReadFile(kModelPath, &posenet_tflite)) {
    TF_LITE_REPORT_ERROR(&error_reporter, "Failed to load model!");
    vTaskSuspend(nullptr);
  }
  auto* model = tflite::GetModel(posenet_tflite.data());
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(&error_reporter,
                         "Model schema version is %d, supported is %d",
                         model->version(), TFLITE_SCHEMA_VERSION);
    vTaskSuspend(nullptr);
  }
  // Creates a micro interpreter.
  tflite::MicroMutableOpResolver<2> resolver;
  resolver.AddCustom(kCustomOp, RegisterCustomOp());
  resolver.AddCustom(kPosenetDecoderOp, RegisterPosenetDecoderOp());
  auto interpreter = tflite::MicroInterpreter{
      model, resolver, tensor_arena, kTensorArenaSize, &error_reporter};
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(&error_reporter, "AllocateTensors failed.");
    vTaskSuspend(nullptr);
  }
  auto* posenet_input = interpreter.input(0);
  // Runs posenet on a test image.
  printf("Getting outputs for posenet test input\r\n");
  std::vector<uint8_t> posenet_test_input_bin;
  if (!LfsReadFile(kTestInputPath, &posenet_test_input_bin)) {
    TF_LITE_REPORT_ERROR(&error_reporter, "Failed to load test input!");
    vTaskSuspend(nullptr);
  }
  if (posenet_input->bytes != posenet_test_input_bin.size()) {
    TF_LITE_REPORT_ERROR(&error_reporter,
                         "Input tensor length doesn't match canned input\r\n");
    vTaskSuspend(nullptr);
  }
  memcpy(tflite::GetTensorData<uint8_t>(posenet_input),
         posenet_test_input_bin.data(), posenet_test_input_bin.size());
  if (interpreter.Invoke() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(&error_reporter, "Invoke failed.");
    vTaskSuspend(nullptr);
  }
  auto test_image_output =
      tensorflow::GetPosenetOutput(&interpreter, 0.5);
  printf("%s\r\n", tensorflow::FormatPosenetOutput(test_image_output).c_str());
  // Starts the camera for live poses.
  CameraTask::GetSingleton()->SetPower(true);
  CameraTask::GetSingleton()->Enable(CameraMode::kStreaming);
  printf("Starting live posenet\r\n");
  auto model_height = posenet_input->dims->data[1];
  auto model_width = posenet_input->dims->data[2];
  
  for (;;) {
    CameraFrameFormat fmt{
        CameraFormat::kRgb,
        CameraFilterMethod::kBilinear,
        CameraRotation::k270,
        model_width,
        model_height,
        false,
        tflite::GetTensorData<uint8_t>(posenet_input)};
    if (!CameraTask::GetSingleton()->GetFrame({fmt})) {
      TF_LITE_REPORT_ERROR(&error_reporter, "Failed to get image from camera.");
      break;
    }
    if (interpreter.Invoke() != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(&error_reporter, "Invoke failed.");
      break;
    }
    auto output = tensorflow::GetPosenetOutput(&interpreter,
                                               0.5);
    //printf("%s\r\n", tensorflow::FormatPosenetOutput(output).c_str());

    for (const auto& pose : output) {
      PersonLocation location = EstimatePersonLocation(pose, model_width, model_height);
      printf("Location: angle_deg=%f, distance_m=%f\r\n", location.angle_deg, location.distance_m);
    }
    printf("\r\n");
    vTaskDelay(pdMS_TO_TICKS(100));
  }
  CameraTask::GetSingleton()->SetPower(false);
}

}  // namespace
}  // namespace coralmicro


extern "C" void app_main(void* param) {
  (void)param;
  coralmicro::Main();
  vTaskSuspend(nullptr);
}
// [end-sphinx-snippet:posenet]
*/

// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Serves a minimal HTML page and JPEG snapshots over the USB serial console,
// matching the framing produced by examples/tracker/uart-http-proxy (HTTP/1.1
// request in, HTTP/1.1 response out with Content-Length).
//
// On the host: run uart-http-proxy with a large --idle-ms and --max-ms, then
// open http://127.0.0.1:8787/ (tune idle timeout for JPEG size and baud rate).
//
// Flash from coralmicro root:
//   bash build.sh
//   python3 scripts/flashtool.py -e uart_camera_http

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "libs/base/console_m7.h"
#include "libs/base/led.h"
#include "libs/base/tasks.h"
#include "libs/camera/camera.h"
#include "libs/libjpeg/jpeg.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"

namespace coralmicro {
namespace {

constexpr size_t kMaxHeader = 16384;
constexpr size_t kMaxBody = 65536;

constexpr char kIndexHtml[] =
    "<!DOCTYPE html><html><head><meta charset=\"utf-8\"/><title>coral "
    "camera</title></head><body>"
    "<p>UART proxy camera (poll /frame)</p>"
    "<img id=\"f\" style=\"max-width:100%;height:auto\" alt=\"frame\"/>"
    "<script>"
    "function t(){var e=document.getElementById('f');"
    "e.src='/frame?'+Date.now();}"
    "setInterval(t,250);t();"
    "</script></body></html>";

struct ParsedRequest {
  char method[8]{};
  char path[192]{};
  size_t content_length = 0;
  std::vector<uint8_t> body;
};

bool StartsWithIgnoreCase(const char* s, const char* prefix) {
  while (*prefix) {
    if (std::tolower(static_cast<unsigned char>(*s)) !=
        std::tolower(static_cast<unsigned char>(*prefix))) {
      return false;
    }
    ++s;
    ++prefix;
  }
  return true;
}

// Normalizes request-target to origin-form path only: "/foo". Mutates `target`.
// Handles absolute-form URIs some proxies send: "http://host:port/path?query".
void CanonicalizeRequestTarget(char* target) {
  if (strncmp(target, "http://", 7) == 0) {
    const char* host = target + 7;
    const char* slash = strchr(host, '/');
    if (slash) {
      memmove(target, slash, strlen(slash) + 1);
    } else {
      target[0] = '/';
      target[1] = '\0';
    }
  } else if (strncmp(target, "https://", 8) == 0) {
    const char* host = target + 8;
    const char* slash = strchr(host, '/');
    if (slash) {
      memmove(target, slash, strlen(slash) + 1);
    } else {
      target[0] = '/';
      target[1] = '\0';
    }
  }
  char* q = strchr(target, '?');
  if (q) {
    *q = '\0';
  }
}

bool HeadersComplete(const std::vector<uint8_t>& header) {
  size_t n = header.size();
  if (n >= 4 && header[n - 4] == '\r' && header[n - 3] == '\n' &&
      header[n - 2] == '\r' && header[n - 1] == '\n') {
    return true;
  }
  if (n >= 2 && header[n - 2] == '\n' && header[n - 1] == '\n') {
    return true;
  }
  return false;
}

void AppendFmt(std::vector<uint8_t>* out, const char* fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  if (n > 0) {
    size_t len = static_cast<size_t>(
        std::min(n, static_cast<int>(sizeof(buf) - 1)));
    out->insert(out->end(), buf, buf + len);
  }
}

void SendRawResponse(const std::vector<uint8_t>& response) {
  if (response.empty()) {
    return;
  }
  auto* console = ConsoleM7::GetSingleton();
  const char* p = reinterpret_cast<const char*>(response.data());
  size_t remaining = response.size();
  // ConsoleM7::Write takes int length; chunk if ever needed.
  constexpr size_t kChunk = static_cast<size_t>(INT_MAX / 2);
  while (remaining > 0) {
    int n = static_cast<int>(std::min(remaining, kChunk));
    console->Write(const_cast<char*>(p), n);
    p += n;
    remaining -= static_cast<size_t>(n);
  }
}

void SendError(int code, const char* phrase, const char* message) {
  std::vector<uint8_t> r;
  AppendFmt(&r, "HTTP/1.1 %d %s\r\nContent-Type: text/plain; "
                 "charset=utf-8\r\nConnection: close\r\n",
            code, phrase);
  AppendFmt(&r, "Content-Length: %u\r\n\r\n%s",
            static_cast<unsigned>(std::strlen(message)), message);
  SendRawResponse(r);
}

bool ReadHttpRequest(ParsedRequest* req) {
  std::vector<uint8_t> header;
  header.reserve(512);
  while (header.size() < kMaxHeader) {
    char c;
    int n = ConsoleM7::GetSingleton()->Read(&c, 1);
    if (n != 1) {
      taskYIELD();
      continue;
    }
    header.push_back(static_cast<uint8_t>(c));
    if (HeadersComplete(header)) {
      break;
    }
  }
  if (header.size() >= kMaxHeader) {
    return false;
  }

  std::string h(header.begin(), header.end());
  size_t line_end = h.find("\r\n");
  size_t line_end_skip = 2;
  if (line_end == std::string::npos) {
    line_end = h.find('\n');
    line_end_skip = 1;
    if (line_end == std::string::npos) {
      return false;
    }
  }
  std::string req_line = h.substr(0, line_end);
  char target[256];
  if (std::sscanf(req_line.c_str(), "%7s %255s", req->method, target) != 2) {
    return false;
  }
  CanonicalizeRequestTarget(target);
  std::strncpy(req->path, target, sizeof(req->path) - 1);
  req->path[sizeof(req->path) - 1] = '\0';

  size_t pos = line_end + line_end_skip;
  while (pos < h.size()) {
    size_t e = h.find("\r\n", pos);
    if (e == std::string::npos) {
      break;
    }
    if (e == pos) {
      break;
    }
    std::string hdr = h.substr(pos, e - pos);
    if (StartsWithIgnoreCase(hdr.c_str(), "content-length:")) {
      const char* v = std::strchr(hdr.c_str(), ':');
      if (v) {
        ++v;
        while (*v == ' ' || *v == '\t') {
          ++v;
        }
        req->content_length = static_cast<size_t>(std::strtoul(v, nullptr, 10));
      }
    }
    pos = e + 2;
  }

  if (req->content_length > kMaxBody) {
    return false;
  }
  req->body.resize(req->content_length);
  for (size_t i = 0; i < req->content_length; ++i) {
    char b;
    int got = 0;
    while (got != 1) {
      got = ConsoleM7::GetSingleton()->Read(&b, 1);
      if (got != 1) {
        taskYIELD();
      }
    }
    req->body[i] = static_cast<uint8_t>(b);
  }
  return true;
}

void HandleFrame() {
  std::vector<uint8_t> rgb(CameraTask::kWidth * CameraTask::kHeight *
                           CameraFormatBpp(CameraFormat::kRgb));
  auto fmt = CameraFrameFormat{
      CameraFormat::kRgb,
      CameraFilterMethod::kBilinear,
      CameraRotation::k0,
      CameraTask::kWidth,
      CameraTask::kHeight,
      /*preserve_ratio=*/false,
      rgb.data(),
      /*white_balance=*/true,
  };
  if (!CameraTask::GetSingleton()->GetFrame({fmt})) {
    SendError(503, "Service Unavailable", "camera frame");
    return;
  }
  std::vector<uint8_t> jpeg;
  JpegCompressRgb(rgb.data(), fmt.width, fmt.height, /*quality=*/75, &jpeg);
  if (jpeg.empty()) {
    SendError(500, "Internal Server Error", "jpeg");
    return;
  }
  std::vector<uint8_t> r;
  AppendFmt(&r,
            "HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\nConnection: close\r\nContent-Length: %u\r\n\r\n",
            static_cast<unsigned>(jpeg.size()));
  r.insert(r.end(), jpeg.begin(), jpeg.end());
  SendRawResponse(r);
}

void Dispatch(const ParsedRequest& req) {
  if (std::strcmp(req.method, "GET") != 0) {
    SendError(405, "Method Not Allowed", "use GET");
    return;
  }
  if (std::strncmp(req.path, "/frame", 6) == 0 &&
      (req.path[6] == '\0' || req.path[6] == '/')) {
    HandleFrame();
    return;
  }
  SendError(404, "Not Found", "path");
}

[[noreturn]] void ServerTask(void* /*param*/) {
  vTaskDelay(pdMS_TO_TICKS(500));
  printf("uart_camera_http: waiting for HTTP on serial (use uart-http-proxy on "
         "host)\r\n");

  while (true) {
    printf("HandleFrame\r\n");
    ParsedRequest req;
    if (!ReadHttpRequest(&req)) {
      SendError(400, "Bad Request", "header");
      continue;
    }
    //Dispatch(req);
  }
}

}  // namespace
}  // namespace coralmicro

extern "C" void app_main(void* param) {
  (void)param;
  coralmicro::LedSet(coralmicro::Led::kStatus, true);
  coralmicro::CameraTask::GetSingleton()->SetPower(true);
  coralmicro::CameraTask::GetSingleton()->Enable(
  coralmicro::CameraMode::kStreaming);

  xTaskCreate(coralmicro::ServerTask, "uart_http_cam",
              configMINIMAL_STACK_SIZE * 20, nullptr,
              coralmicro::kAppTaskPriority, nullptr);
  vTaskSuspend(nullptr);
}
