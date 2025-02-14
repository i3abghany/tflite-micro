/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/examples/anomaly_detection/ad_model_settings.h"
#include "tensorflow/lite/micro/examples/anomaly_detection/ad_model.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/examples/anomaly_detection/detection_responder.h"
#include "tensorflow/lite/micro/examples/anomaly_detection/util/quantization_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/examples/anomaly_detection/dataset.h"
#include <iostream>
#include <fstream>
#include <chrono>

struct TestSample
{
  std::string name;
  const float *data;
  size_t size;
};

TestSample GetTestSample(const char *dataset_path, const char* filename, int idx)
{
    std::ifstream in(std::string(dataset_path) + "/" + std::string(filename), std::ifstream::ate | std::ifstream::binary);
    size_t size = in.tellg();
    assert(size % 4 == 0);
    size /= 4;
    const int INPUT_SAMPLE_SIZE = 640;
    int seek = INPUT_SAMPLE_SIZE * idx * sizeof(float);
    in.seekg(seek);
    float *data = new float[INPUT_SAMPLE_SIZE];
    float tmp;
    for (int j = 0; j < INPUT_SAMPLE_SIZE; j++) {
      in.read(reinterpret_cast<char*>(&tmp), sizeof(float));
      data[j] = tmp;
    }
    return { std::to_string(idx), data, INPUT_SAMPLE_SIZE };
}

constexpr int tensor_arena_size = 200 * 1024;
uint8_t tensor_arena[tensor_arena_size];
const char *dataset_path = "REV_PARSE_PATH_PLACEHOLDER/tflite-micro/tensorflow/lite/micro/examples/anomaly_detection/dataset";

TF_LITE_MICRO_TESTS_BEGIN
TF_LITE_MICRO_TEST(TestInvoke) {
  const tflite::Model* model = ::tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    std::cout << "Model provided is schema version not equal to supported version" << std::endl;
  }

  tflite::MicroMutableOpResolver<3> micro_op_resolver;
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddQuantize();
  micro_op_resolver.AddDequantize();

  tflite::MicroInterpreter interpreter(model, micro_op_resolver, tensor_arena,
                                       tensor_arena_size);
  interpreter.AllocateTensors();
  TfLiteTensor* input = interpreter.input(0);

  float input_scale = input->params.scale;
  int input_zero_point = input->params.zero_point;
  int i = 0;
  auto t1 = std::chrono::high_resolution_clock::now();
  for (const char *name : test_sample_file_paths)
  {
    if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0)
      continue;
    for (int j = 0; j < 40; j++) {
      auto datum = GetTestSample(dataset_path, name, j);
      // std::cout << "Starting inference: " << i << " (AD)" << std::endl;
      i++;
      int8_t *quant_input = new int8_t[datum.size];
      for (size_t k = 0; k < datum.size; k++) {
        quant_input[k] = QuantizeFloatToInt8(datum.data[k], input_scale, input_zero_point);
      }
      TFLITE_DCHECK_EQ(input->bytes, static_cast<size_t>(datum.size));
      memcpy(input->data.int8, quant_input, input->bytes);
      TfLiteStatus invoke_status = interpreter.Invoke();
      if (invoke_status != kTfLiteOk) {
        std::cout << "Invoke failed\n";
      }
      TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);
      TfLiteTensor* output = interpreter.output(0);
      float diffsum = 0;

      for (size_t t = 0; t < kFeatureElementCount; t++) {
        float converted = DequantizeInt8ToFloat(output->data.int8[t], interpreter.output(0)->params.scale,
                                                interpreter.output(0)->params.zero_point);
        float diff = converted - datum.data[t];
        diffsum += diff * diff;
      }
      diffsum /= kFeatureElementCount;
      std::cout << "diffsum: " << diffsum << std::endl;
    }
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto sz = sizeof(test_sample_file_paths) / sizeof(test_sample_file_paths[0]);
  std::cout << "Time in ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / sz << std::endl;
}

TF_LITE_MICRO_TESTS_END
