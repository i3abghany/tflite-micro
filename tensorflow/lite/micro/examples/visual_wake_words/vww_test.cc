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
#include "tensorflow/lite/micro/examples/visual_wake_words/vww_model_settings.h"
#include "tensorflow/lite/micro/examples/visual_wake_words/vww_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/examples/visual_wake_words/detection_responder.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/examples/visual_wake_words/dataset.h"
#include <iostream>
#include <fstream>
#include <chrono>

struct TestSample
{
  std::string name;
  int8_t *data;
  size_t size;
};

TestSample GetTestSample(const char *dataset_path, const char* filename)
{
    std::string full_path = std::string(dataset_path) + "/" + std::string(filename);
    std::ifstream in(full_path, std::ifstream::ate | std::ifstream::binary);
    assert(in.is_open());
    size_t size = in.tellg();
    in.seekg(0);
    char *data = new char[size];
    in.read(data, size);
    return TestSample{ std::string(filename), (int8_t *) data, size };
}

constexpr int tensor_arena_size = 300 * 1024;
uint8_t tensor_arena[tensor_arena_size];
const char *dataset_path = "REV_PARSE_PATH_PLACEHOLDER/tflite-micro/tensorflow/lite/micro/examples/visual_wake_words/dataset";

TF_LITE_MICRO_TESTS_BEGIN
TF_LITE_MICRO_TEST(TestInvoke) {
  const tflite::Model* model = ::tflite::GetModel(__vww_96_int8_mod_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    std::cout << "Model provided is schema version not equal to supported version" << std::endl;
  }
  tflite::MicroMutableOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddAveragePool2D(tflite::Register_AVERAGE_POOL_2D_INT8());
  micro_op_resolver.AddConv2D(tflite::Register_CONV_2D_INT8());
  micro_op_resolver.AddDepthwiseConv2D(tflite::Register_DEPTHWISE_CONV_2D_INT8());
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax(tflite::Register_SOFTMAX_INT8());
  micro_op_resolver.AddFullyConnected(tflite::Register_FULLY_CONNECTED_INT8());

  tflite::MicroInterpreter interpreter(model, micro_op_resolver, tensor_arena, tensor_arena_size);
  interpreter.AllocateTensors();
  int i = 0;
  auto t1 = std::chrono::high_resolution_clock::now();
  for (const char *name : test_sample_file_paths)
  {
    if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0)
      continue;
    auto datum = GetTestSample(dataset_path, name);
    // std::cout << "Starting inference: " << i << " (VWW)" << std::endl;
    TfLiteTensor* input = interpreter.input(0);
    TFLITE_DCHECK_EQ(input->bytes, static_cast<size_t>(datum.size));
    memcpy(input->data.int8, datum.data, input->bytes);
    TfLiteStatus invoke_status = interpreter.Invoke();

    if (invoke_status != kTfLiteOk) {
      std::cout << "Invoke failed\n";
    }
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

    TfLiteTensor* output = interpreter.output(0);
    int non_person_score = output->data.int8[0];
    int is_person_score = output->data.int8[1];

    bool predicted_person = is_person_score > non_person_score;
    bool is_correct = RespondToDetection(predicted_person, datum.name.c_str());

    std::cout << "is_correct: " << is_correct << std::endl;
    i++;
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto sz = sizeof(test_sample_file_paths) / sizeof(test_sample_file_paths[0]);
  std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / sz << std::endl;
}

TF_LITE_MICRO_TESTS_END
