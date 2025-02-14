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

#include "tensorflow/lite/micro/kernels/conv.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "xxhash.h"

namespace tflite {
namespace {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataConv));
}

// uint32_t GetBiasHash(int32_t* bias_data, size_t n_params) {
//   return XXH32(bias_data, n_params * sizeof(int32_t), 0x1337);
// }
// 
// TfLiteStatus PopulateHashesForBias(TfLiteContext* context, TfLiteNode* node, size_t n_params) {
//   if (node->bias_hash != 0xFFFFFFFF)
//     return kTfLiteOk;
// 
//   auto redundancy = tflite::micro::GetEvalRedundancyTensors(context, node, kConvBiasTensor);
//   int32_t *d_r2 = tflite::micro::GetTensorData<int32_t>(redundancy.t2);
// 
//   node->bias_hash = GetBiasHash(d_r2, n_params);
// 
//   return kTfLiteOk;
// }

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);

  // int32_t *d_orig = tflite::micro::GetTensorData<int32_t>(const_cast<TfLiteEvalTensor*>(bias));
  // size_t n_params = tflite::micro::GetTensorShape(bias).FlatSize();
  // PopulateHashesForBias(context, node, n_params);
  // uint32_t calculated_hash = GetBiasHash(d_orig, n_params);

  // if (node->bias_hash != calculated_hash) {
  //   auto redundancy = tflite::micro::GetEvalRedundancyTensors(context, node, kConvBiasTensor);
  //   int32_t *d_r1 = tflite::micro::GetTensorData<int32_t>(redundancy.t1);
  //   MicroPrintf("Warning: found an integrity violation: %d, %d", node->bias_hash, calculated_hash);
  //   for (size_t i = 0; i < n_params; i++)
  //     d_orig[i] = d_r1[i];
  // }

  // auto redundancy = tflite::micro::GetEvalRedundancyTensors(context, node, kConvBiasTensor);

  // int32_t *d_orig = tflite::micro::GetTensorData<int32_t>(const_cast<TfLiteEvalTensor*>(bias));
  // int32_t *d_r1 = tflite::micro::GetTensorData<int32_t>(redundancy.t1);
  // int32_t *d_r2 = tflite::micro::GetTensorData<int32_t>(redundancy.t2);

  // size_t n_params = tflite::micro::GetTensorShape(bias).FlatSize();
  // for (size_t i = 0; i < n_params; i++) {
  //   auto& d = d_orig[i], &r1 = d_r1[i], &r2 = d_r2[i];
  //   if (d != r1 || r1 != r2 || d != r2) {
  //     MicroPrintf("Warning: found an integrity violation: %d, %d, %d", d, r1, r2);
  //     auto voted_value = (d & r1) | (r1 & r2) | (d & r2);
  //     d = voted_value;
  //   }
  // }

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  TFLITE_DCHECK(node->user_data != nullptr);
  const auto& data = *(static_cast<const OpDataConv*>(node->user_data));

  TF_LITE_ENSURE_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_MSG(
      context,
      input->type == filter->type ||
          (input->type == kTfLiteInt16 && filter->type == kTfLiteInt8) ||
          (input->type == kTfLiteInt8 && filter->type == kTfLiteInt4),
      "Hybrid models are not supported on TFLite Micro.");

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32: {
      tflite::reference_ops::Conv(
          ConvParamsFloat(params, data), tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(filter),
          tflite::micro::GetTensorData<float>(filter),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetOptionalTensorData<float>(bias),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output),
          tflite::micro::GetTensorShape(nullptr), nullptr);
      break;
    }
    case kTfLiteInt16: {
      switch (bias->type) {
        case kTfLiteInt32: {
          reference_integer_ops::ConvPerChannel(
              ConvParamsQuantized(params, data),
              data.per_channel_output_multiplier, data.per_channel_output_shift,
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int16_t>(input),
              tflite::micro::GetTensorShape(filter),
              tflite::micro::GetTensorData<int8_t>(filter),
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<std::int32_t>(bias),
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int16_t>(output));
          break;
        }
        case kTfLiteInt64: {
          reference_integer_ops::ConvPerChannel(
              ConvParamsQuantized(params, data),
              data.per_channel_output_multiplier, data.per_channel_output_shift,
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int16_t>(input),
              tflite::micro::GetTensorShape(filter),
              tflite::micro::GetTensorData<int8_t>(filter),
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<std::int64_t>(bias),
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int16_t>(output));
          break;
        }
        default:
          MicroPrintf("Bias type %s (%d) not supported.",
                      TfLiteTypeGetName(bias->type), bias->type);
          return kTfLiteError;
      }
      break;
    }
    case kTfLiteInt8: {
      switch (filter->type) {
        case kTfLiteInt4: {
          int8_t* unpacked_filter_data = static_cast<int8_t*>(
              context->GetScratchBuffer(context, data.filter_buffer_index));
          tflite::tensor_utils::UnpackDenseInt4IntoInt8(
              tflite::micro::GetTensorData<int8_t>(filter),
              tflite::micro::GetTensorShape(filter).FlatSize(),
              unpacked_filter_data);
          reference_integer_ops::ConvPerChannel(
              ConvParamsQuantized(params, data),
              data.per_channel_output_multiplier, data.per_channel_output_shift,
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int8_t>(input),
              tflite::micro::GetTensorShape(filter), unpacked_filter_data,
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<int32_t>(bias),
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int8_t>(output));
          break;
        }
        case kTfLiteInt8: {
          reference_integer_ops::ConvPerChannel(
              ConvParamsQuantized(params, data),
              data.per_channel_output_multiplier, data.per_channel_output_shift,
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int8_t>(input),
              tflite::micro::GetTensorShape(filter),
              tflite::micro::GetTensorData<int8_t>(filter),
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<int32_t>(bias),
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int8_t>(output));
          break;
        }
        default:
          MicroPrintf("Weight type %s (%d) not supported.",
                      TfLiteTypeGetName(filter->type), filter->type);
          return kTfLiteError;
      }
      break;
    }
    default:
      MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                  input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_CONV_2D() {
  return tflite::micro::RegisterOp(Init, ConvPrepare, Eval);
}

}  // namespace tflite
