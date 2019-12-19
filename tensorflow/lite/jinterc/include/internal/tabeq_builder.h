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

#ifndef TABEQ_MODEL_BUILDER_H_

#include <cstdint>
#include <string>

#include <tabeq/model.h>
#include <tabeq/operations.h>

#include "tensorflow/lite/context.h"

#include <sstream>

// -- Helper defs for line and file
//
#define LFSTRM(msg)         \
    std::ostringstream msg; \
    msg << __FILE__ << ":" << __LINE__ << " -> "

#define LFCHAR(msg) msg.str().c_str()
#define LFSTR(msg) msg.str()

namespace tflite {
namespace tabeq {

using Status = ::tabeq::Status;
using TabeqGraph = ::tabeq::TabeqGraph;
using TensorRef = ::tabeq::TensorRef;
using Node = ::tabeq::Node;

// Validates which operations are supported and returns array of operations to
// replace with GPU kernels. The caller must free the pointer on TfLiteIntArray.
TfLiteIntArray* GetOpsToReplace(TfLiteContext* context);

// Extracts TFLite delegate execution plan from the input TFLite context and
// converts it into generic graph format.
Status BuildModel(TfLiteContext* context,
                  const TfLiteDelegateParams* delegate_params,
                  TabeqGraph* graph);

// Module-internal converter, exposed for unit testing purpose only.
Status ConvertTfLiteTensorToTensorRef(const TfLiteTensor& tflite_tensor,
                                      TensorRef* tensor_ref);

}  // namespace tabeq
}  // namespace tflite

#endif  // TABEQ_MODEL_BUILDER_H_
