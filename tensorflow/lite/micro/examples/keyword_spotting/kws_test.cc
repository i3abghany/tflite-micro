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
#include "tensorflow/lite/micro/examples/keyword_spotting/kws_model_settings.h"
#include "tensorflow/lite/micro/examples/keyword_spotting/dataset/tst_000595_On_5_test_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/examples/keyword_spotting/kws_data.h"
#include "tensorflow/lite/micro/examples/keyword_spotting/detection_responder.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/examples/keyword_spotting/dataset.h"
#include <iostream>
#include <fstream>

struct TestSample
{
  std::string name;
  const int8_t *data;
  size_t size;
};

TestSample GetTestSample(const char *dataset_path, const char* filename)
{
    std::ifstream in(std::string(dataset_path) + "/" + std::string(filename), std::ifstream::ate | std::ifstream::binary);
    size_t size = in.tellg();
    in.seekg(0);
    char *data = new char[size];
    in.read(data, size);
    return { filename, (int8_t *) data, size };
}

std::vector<TestSample> load_test_data()
{
  const char *dataset_path = "/localhome/mam47/libs/tflite-micro/tensorflow/lite/micro/examples/keyword_spotting/dataset";
  std::vector<TestSample> ret;

#if 0
  DIR *dir = opendir(dataset_path);
  struct dirent *ent = readdir(dir);
  while (ent != NULL)
  {
    if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
    {
      ent = readdir(dir);
      continue;
    }
    ret.push_back(GetTestSample(dataset_path, ent->d_name));
    ent = readdir(dir);
  }
#else
  for (const char *name : test_sample_file_paths)
  {
    if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0)
      continue;
    ret.push_back(GetTestSample(dataset_path, name));
  }
#endif
  return ret;
}

constexpr int tensor_arena_size = 200 * 1024;
uint8_t tensor_arena[tensor_arena_size];

TF_LITE_MICRO_TESTS_BEGIN
extern int8_t g_kws_ref_model_model_data[];
TF_LITE_MICRO_TEST(TestInvoke) {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = ::tflite::GetModel(g_kws_ref_model_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    std::cout << "Model provided is schema version not equal to supported version" << std::endl;
  }

  tflite::MicroMutableOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddConv2D(tflite::Register_CONV_2D_INT8());
  micro_op_resolver.AddDepthwiseConv2D(tflite::Register_DEPTHWISE_CONV_2D_INT8());
  micro_op_resolver.AddAveragePool2D(tflite::Register_AVERAGE_POOL_2D_INT8());
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddFullyConnected(tflite::Register_FULLY_CONNECTED_INT8());
  micro_op_resolver.AddSoftmax(tflite::Register_SOFTMAX_INT8());

  // Build an interpreter to run the model with.
  tflite::MicroInterpreter interpreter(model, micro_op_resolver, tensor_arena,
                                       tensor_arena_size);
  interpreter.AllocateTensors();

  // Get information about the memory area to use for the model's input.
  TfLiteTensor* input = interpreter.input(0);

  // Make sure the input has the properties we expect.
  // TF_LITE_MICRO_EXPECT(input != nullptr);
  // TF_LITE_MICRO_EXPECT_EQ(4, input->dims->size);
  // TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  // TF_LITE_MICRO_EXPECT_EQ(kNumRows, input->dims->data[1]);
  // TF_LITE_MICRO_EXPECT_EQ(kNumCols, input->dims->data[2]);
  // TF_LITE_MICRO_EXPECT_EQ(kNumChannels, input->dims->data[3]);
  // TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt8, input->type);

  // Copy an image with a person into the memory area used for the input.
  auto test_data = load_test_data();
  int correct = 0;
  int i = 0;
  std::cout << "Starting inference\n";
  for (auto &datum : test_data)
  {
    std::cout << "Starting inference: " << i << std::endl;
    TFLITE_DCHECK_EQ(input->bytes, static_cast<size_t>(datum.size));
    memcpy(input->data.int8, datum.data, input->bytes);
    TfLiteStatus invoke_status = interpreter.Invoke();
    if (invoke_status != kTfLiteOk) {
      std::cout << "Invoke failed\n";
    }
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);
    TfLiteTensor* output = interpreter.output(0);
    bool is_correct = RespondToDetection(output->data.int8, datum.name.c_str());
    correct += is_correct == true;
    std::cout << i << "/" << test_data.size() << ": predicted correctly: " << (is_correct ? "true" : "false") << std::endl;
    i++;
  }

  std::cout << "Testing accuracy: " << ((float) correct / test_data.size()) << std::endl;
}

TF_LITE_MICRO_TESTS_END
