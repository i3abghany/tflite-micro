/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/examples/mnist_lenet/detection_responder.h"
#include "tensorflow/lite/micro/examples/mnist_lenet/lenet_model_settings.h"

#include "tensorflow/lite/micro/micro_log.h"
#include <iostream>
#include <cstring>

int extract_ground_truth(const char *test_sample_name)
{
  static char cp[64];
  strncpy(cp, test_sample_name, sizeof("000_0"));
  char *p = strtok(cp, "_");
  p = strtok(NULL, "_");
  return stoi(std::string(p));
}

bool RespondToDetection(int8_t *probs, const char *filename)
{
  int y_true = extract_ground_truth(filename);
  int8_t argmax = 0;
  int8_t max = probs[0];
  for (int i = 1; i < kCategoryCount; i++) {
    if (probs[i] > max) {
      max = probs[i];
      argmax = i;
    }
  }
  return argmax == y_true;
}
