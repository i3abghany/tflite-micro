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
/// \brief Visual wakewords serialized flatbuffer model.

/// \detail This is a TensorFlow Lite model file that has been converted into a
/// C data array using the tensorflow.lite.util.convert_bytes_to_c_source()
/// function. This form is useful for compiling into a binary for devices that
/// don't have a file system.

// Keep model aligned to 8 bytes to guarantee aligned 64-bit accesses.