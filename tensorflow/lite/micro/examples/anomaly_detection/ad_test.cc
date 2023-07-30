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

struct TestSample
{
  std::string name;
  const float *data;
  size_t size;
};

std::vector<TestSample> GetTestSample(const char *dataset_path, const char* filename)
{
    std::ifstream in(std::string(dataset_path) + "/" + std::string(filename), std::ifstream::ate | std::ifstream::binary);
    size_t size = in.tellg();
    in.seekg(0);
    assert(size % 4 == 0);
    size /= 4;
    const int INPUT_SAMPLE_SIZE = 640;
    float **data = new float*[size / INPUT_SAMPLE_SIZE];
    std::vector<TestSample> ret;
    float tmp;
    for (int i = 0; i < (int)(size / INPUT_SAMPLE_SIZE); i++) {
      data[i] = new float[INPUT_SAMPLE_SIZE];
      for (int j = 0; j < INPUT_SAMPLE_SIZE; j++) {
        in.read(reinterpret_cast<char*>(&tmp), sizeof(float));
        data[i][j] = tmp;
      }
      ret.push_back({ std::to_string(i), data[i], INPUT_SAMPLE_SIZE });
    }
    return ret;
}

std::vector<TestSample> load_test_data()
{
  const char *dataset_path = "/localhome/mam47/research/microscale/tflite-micro/tensorflow/lite/micro/examples/anomaly_detection/dataset";
  for (const char *name : test_sample_file_paths)
  {
    if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0)
      continue;
    return GetTestSample(dataset_path, name);
  }
  assert(false);
}

constexpr int tensor_arena_size = 200 * 1024;
uint8_t tensor_arena[tensor_arena_size];

TF_LITE_MICRO_TESTS_BEGIN
TF_LITE_MICRO_TEST(TestInvoke) {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = ::tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    std::cout << "Model provided is schema version not equal to supported version" << std::endl;
  }

  tflite::MicroMutableOpResolver<3> micro_op_resolver;
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddQuantize();
  micro_op_resolver.AddDequantize();

  // Build an interpreter to run the model with.
  tflite::MicroInterpreter interpreter(model, micro_op_resolver, tensor_arena,
                                       tensor_arena_size);
  interpreter.AllocateTensors();

  // Get information about the memory area to use for the model's input.
  TfLiteTensor* input = interpreter.input(0);

  // Copy an image with a person into the memory area used for the input.
  auto test_data = load_test_data();
  float input_scale = input->params.scale;
  int input_zero_point = input->params.zero_point;
  int i = 0;
  for (auto &datum : test_data)
  {
    std::cout << "Starting inference: " << i << std::endl;
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

TF_LITE_MICRO_TESTS_END
