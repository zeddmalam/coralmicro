
// Copyright 2022 Google LLC
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
#include <cstdio>
#include <string>
#include <vector>
#include <unistd.h>

#include "libs/base/filesystem.h"
#include "libs/base/led.h"
#include "libs/camera/camera.h"
#include "libs/libjpeg/jpeg.h"
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
// Auto white-balance can over-correct in some scenes and produce unstable casts.
// Keep this explicit so color behavior is predictable and easy to tune.
constexpr bool kAutoWhiteBalance = false;
// When true, serial JSON sends a tiny fixed grey placeholder (1×1 WebP as
// base64, ~60 chars) instead of a camera frame. Monitoring stack upscales to
// 324×324 for pose overlay.
constexpr bool kNoImage = true;

std::string Base64Encode(const std::vector<uint8_t>& data) {
  static const char kTable[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const uint8_t* p = data.data();
  const size_t len = data.size();
  std::string out;
  out.reserve((len + 2) / 3 * 4);
  for (size_t i = 0; i < len; i += 3) {
    const unsigned a = p[i];
    const unsigned b = i + 1 < len ? p[i + 1] : 0;
    const unsigned c = i + 2 < len ? p[i + 2] : 0;
    out.push_back(kTable[a >> 2]);
    out.push_back(kTable[((a & 3) << 4) | (b >> 4)]);
    out.push_back(i + 1 < len ? kTable[((b & 15) << 2) | (c >> 6)] : '=');
    out.push_back(i + 2 < len ? kTable[c & 63] : '=');
  }
  return out;
}

void WriteBase64ToStdout(const char* s, size_t n) {
  constexpr size_t kChunk = 512;
  for (size_t i = 0; i < n; i += kChunk) {
    size_t chunk = n - i;
    if (chunk > kChunk) {
      chunk = kChunk;
    }
    (void)write(STDOUT_FILENO, s + i, chunk);
  }
  (void)fflush(stdout);
}

void WriteBase64ToStdout(const std::string& b64) {
  WriteBase64ToStdout(b64.c_str(), b64.size());
}

void HandleFrame() {
  std::vector<uint8_t> rgb(CameraTask::kWidth * CameraTask::kHeight *
                           CameraFormatBpp(CameraFormat::kRgb));
  auto fmt = CameraFrameFormat{
    CameraFormat::kRgb,
    CameraFilterMethod::kBilinear,
    CameraRotation::k270,
    CameraTask::kWidth,
    CameraTask::kHeight,
      /*preserve_ratio=*/false,
      rgb.data(),
      /*white_balance=*/kAutoWhiteBalance,
  };
  if (!CameraTask::GetSingleton()->GetFrame({fmt})) {
    printf("HandleFrame: GetFrame failed\r\n");
    return;
  }
  std::vector<uint8_t> jpeg;
  JpegCompressRgb(rgb.data(), fmt.width, fmt.height, /*quality=*/75, &jpeg);
  if (jpeg.empty()) {
    printf("HandleFrame: JpegCompressRgb failed\r\n");
    return;
  }
  WriteBase64ToStdout(Base64Encode(jpeg));
}

// 1×1 mid-grey WebP, base64-encoded (44 bytes raw).
constexpr char kNoImagePlaceholderWebPBase64[] =
    "UklGRiQAAABXRUJQVlA4IBgAAABQAQCdASoBAAEADMDOJaQABHQAAAAAAAA=";

void EmitNoImagePlaceholderBase64() {
  WriteBase64ToStdout(kNoImagePlaceholderWebPBase64,
                      sizeof(kNoImagePlaceholderWebPBase64) - 1);
}

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
        tflite::GetTensorData<uint8_t>(posenet_input),
        kAutoWhiteBalance};
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
    printf("{\"poses\":[");

    bool first_pose = true;
    (void)fflush(stdout);
    for (const auto& pose : output) {
      if (!first_pose) {
        printf(",");
      }
      first_pose = false;

      printf("{\"score\":%.4f,\"keypoints\":[", pose.score);
      (void)fflush(stdout);

      for (int j = 0; j < tensorflow::kKeypoints; ++j) {
        if (j > 0) {
          printf(",");
        }
        printf(
            "{\"name\":\"%s\",\"x\":%.4f,\"y\":%.4f,\"score\":%.4f}",
            tensorflow::KeypointTypes[j], pose.keypoints[j].x,
            pose.keypoints[j].y, pose.keypoints[j].score);
        (void)fflush(stdout);
      }
      printf("]}");
      (void)fflush(stdout);
      //printf("{angle_deg:%f, distance_m:%f}\n", location.angle_deg, location.distance_m);
    }
    printf("],\"imageData\":\"");
    // write() bypasses the stdio buffer; without fflush, base64 can appear on
    // the wire before this printf's tail ("imageData:\"") is flushed.
    (void)fflush(stdout);
    if (kNoImage) {
      EmitNoImagePlaceholderBase64();
    } else {
      HandleFrame();
    }
//    vTaskDelay(pdMS_TO_TICKS(500));
    printf("\"");
    printf("}\n\n");
    (void)fflush(stdout);
    // ~4k base64 @ 115200 8N1 needs ~350ms on the wire; shorter delay overflows
    // the USB/UART path and drops the payload (headers still print).
    vTaskDelay(pdMS_TO_TICKS(10));
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
