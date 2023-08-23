/* Copyright 2020 The MLPerf Authors. All Rights Reserved.
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
/// \file
/// \brief MNIST LSTM model settings.

#ifndef V0_1_IC_MODEL_SETTINGS_H_
#define V0_1_IC_MODEL_SETTINGS_H_

constexpr int kNumCols = 28;
constexpr int kNumRows = 28;
constexpr int kNumChannels = 1;

constexpr int kIcInputSize = kNumCols * kNumRows * kNumChannels;

constexpr int kCategoryCount = 10;

#endif  // V0_1_IC_MODEL_SETTINGS_H_