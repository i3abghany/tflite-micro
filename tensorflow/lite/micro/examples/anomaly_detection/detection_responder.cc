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

#include "tensorflow/lite/micro/examples/anomaly_detection/detection_responder.h"
#include "tensorflow/lite/micro/examples/anomaly_detection/ad_model_settings.h"

#include "tensorflow/lite/micro/micro_log.h"
#include <iostream>
#include <cstring>

// This dummy implementation writes person and no person scores to the error
// console. Real applications will want to take some custom action instead, and
// should implement their own versions of this function.

int extract_ground_truth(const char *test_sample_name)
{
  static char cp[64];
  strcpy(cp, test_sample_name);
  char *p = strtok(cp, "_");
  int i = 0;
  while (p) {
      if (i == 3) {
        return stoi(std::string(p));
      }
      else p = strtok(NULL, "_");
      i++;
  }
  return 0;
}

bool RespondToDetection(int8_t *accuracies, const char *test_sample_name) {
  int max_idx = 0;
  int max_value = accuracies[0];

  for (int i = 1; i < 2; i++) {
    if (max_value < accuracies[i]) {
      max_idx = i; max_value = accuracies[i];
    }
  }
  return extract_ground_truth(test_sample_name) == max_idx;
}
