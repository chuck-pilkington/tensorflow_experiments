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

#ifndef TENSORFLOW_LITE_DELEGATES_JINTERC_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_JINTERC_DELEGATE_H_

#include <stdint.h>

#include "tensorflow/lite/c/c_api_internal.h"

#ifdef SWIG
#define TFL_CAPI_EXPORT
#else
#if defined(_WIN32)
#ifdef TFL_COMPILE_LIBRARY
#define TFL_CAPI_EXPORT __declspec(dllexport)
#else
#define TFL_CAPI_EXPORT __declspec(dllimport)
#endif  // TFL_COMPILE_LIBRARY
#else
#define TFL_CAPI_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32
#endif  // SWIG

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Encapsulated compilation/runtime tradeoffs.
enum TfLiteJintercInferenceUsage {
  // Delegate will be used only once, therefore, bootstrap/init time should
  // be taken into account.
  TFLITE_JINTERC_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER = 0,

  // Prefer maximizing the throughput. Same delegate will be used repeatedly on
  // multiple inputs.
  TFLITE_JINTERC_INFERENCE_PREFERENCE_SUSTAINED_SPEED = 1,
};

enum TfLiteJintercInferencePriority {
  TFLITE_JINTERC_INFERENCE_PRIORITY_MAX_PRECISION = 0,
  TFLITE_JINTERC_INFERENCE_PRIORITY_MIN_LATENCY = 1,
  TFLITE_JINTERC_INFERENCE_PRIORITY_MIN_MEMORY_USAGE = 2,
};

// IMPORTANT: Always use TfLiteJintercDelegateOptionsDefault() method to create
// new instance of TfLiteJintercDelegateOptions, otherwise every new added option
// may break inference.
typedef struct {
  // When set to zero, computations are carried out in maximal possible
  // precision. Otherwise, the JINTERC may quantify tensors, downcast values,
  // process in FP16 to increase performance. For most models precision loss is
  // warranted.
  // [OBSOLETE]: to be removed
  int32_t is_precision_loss_allowed;

  // Preference is defined in TfLiteJintercInferencePreference.
  int32_t inference_preference;

  // Ordered priorities provide better control over desired semantics,
  // where priority(n) is more important than priority(n+1), therefore,
  // each time inference engine needs to make a decision, it uses
  // ordered priorities to do so.
  // For example:
  //   MAX_PRECISION at priority1 would not allow to decrease presision,
  //   but moving it to priority2 or priority3 would result in F16 calculation.
  //
  // Priority is defined in TfLiteJintercInferencePriority.
  int32_t inference_priority1;
  int32_t inference_priority2;
  int32_t inference_priority3;
} TfLiteJintercDelegateOptions;

// Populates TfLiteJintercDelegateOptions as follows:
//   is_precision_loss_allowed = false
//   inference_preference = TFLITE_JINTERC_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER
//   priority1 = TFLITE_JINTERC_INFERENCE_PRIORITY_MAX_PRECISION
//   priority2 = TFLITE_JINTERC_INFERENCE_PRIORITY_MIN_LATENCY
//   priority3 = TFLITE_JINTERC_INFERENCE_PRIORITY_MIN_MEMORY_USAGE
TFL_CAPI_EXPORT TfLiteJintercDelegateOptions TfLiteJintercDelegateOptionsDefault();

// Creates a new delegate instance that need to be destroyed with
// TfLiteJintercDelegateDelete when delegate is no longer used by TFLite.
//
// This delegate encapsulates multiple JINTERC-acceleration APIs under the hood to
// make use of the fastest available on a device.
//
// When `options` is set to `nullptr`, then default options are used.
TFL_CAPI_EXPORT TfLiteDelegate* TfLiteJintercCreate(
    const TfLiteJintercDelegateOptions* options);

// Destroys a delegate created with `TfLiteJintercCreate` call.
TFL_CAPI_EXPORT void TfLiteJintercDelegateDelete(TfLiteDelegate* delegate);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_DELEGATES_JINTERC_DELEGATE_H_
