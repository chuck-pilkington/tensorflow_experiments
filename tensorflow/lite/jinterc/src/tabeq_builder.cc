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

#include <internal/tabeq_builder.h>

#include <stddef.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <jinterc.h>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/context.h"

#if 0
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#endif

#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace tabeq {

using namespace ::tabeq::model;
using namespace ::tabeq;

Status IsSupported(const TfLiteContext* context, TfLiteNode* node,
                   const TfLiteRegistration* registration) {
#if 1
    throw JintercException("IsSupported is not supported");
#else
    return NewOperationParser(registration)
        ->IsSupported(context, node, registration);
#endif
}

bool IsAllFloatTensors(const TfLiteContext* context,
                       const TfLiteIntArray* array) {
    for (int i = 0; i < array->size; ++i) {
        const TfLiteTensor* t = context->tensors + array->data[i];
        bool const type_supported =
            (t->type == kTfLiteFloat32 || t->type == kTfLiteFloat16);
        if (t->allocation_type == kTfLiteArenaRw && !type_supported) {
            return false;
        }
    }
    return true;
}

std::string GetOpNameByRegistration(const TfLiteRegistration* registration) {
    auto op = registration->builtin_code;
    std::string result =
        EnumNameBuiltinOperator(static_cast<BuiltinOperator>(op));
    if (op == kTfLiteBuiltinCustom) {
        result += " " + std::string(registration->custom_name);
    }
    return result;
}

Status GetNodeAndRegistration(TfLiteContext* context, int node_id,
                              TfLiteNode** tflite_node,
                              TfLiteRegistration** registration) {
    if (context->GetNodeAndRegistration(context, node_id, tflite_node,
                                        registration) != kTfLiteOk) {
        return InvalidArgumentError(absl::StrCat(
            "Couldn't get node and registration info for op: ", node_id));
    }
    return OkStatus();
}

// TODO(impjdi): Check number of input/output tensors and their dimensions.
// TODO(impjdi): Check ops' parameters.
TfLiteIntArray* GetOpsToReplace(TfLiteContext* context) {

    TfLiteIntArray* execution_plan = nullptr;

    if (context->GetExecutionPlan(context, &execution_plan) != kTfLiteOk) {
        context->ReportError(context, "Unable to get graph execution plan.");
        return nullptr;
    }

    // Iterate through graph and find ops to replace.
    TfLiteIntArray* subgraph = TfLiteIntArrayCreate(execution_plan->size);
    subgraph->size = 0;
    std::set<std::string> errors;
    for (int i = 0; i < execution_plan->size; ++i) {
        const int node_id = execution_plan->data[i];
        TfLiteNode* node;
        TfLiteRegistration* registration;
        auto status =
            GetNodeAndRegistration(context, node_id, &node, &registration);
        if (!status.ok()) {
            context->ReportError(context, status.error_message().c_str());
            return nullptr;
        }
        status = IsSupported(context, node, registration);
        if (status.ok() &&
            // TODO(eignasheva): resolve sub operation support for metal
            // delegate registration->builtin_code != kTfLiteBuiltinSub &&
            IsAllFloatTensors(context, node->inputs) &&
            IsAllFloatTensors(context, node->outputs)) {
            if (errors.empty()) subgraph->data[subgraph->size++] = node_id;
        } else {
            errors.insert(absl::StrCat(GetOpNameByRegistration(registration),
                                       ": ", status.error_message()));
        }
    }
    if (!errors.empty()) {
        std::string unsupported = absl::StrJoin(errors, "\n");
        std::string error_message =
            "Next operations are not supported by GPU delegate:\n" +
            unsupported + "\nFirst " + std::to_string(subgraph->size) +
            " operations will run on the GPU, and the remaining " +
            std::to_string(execution_plan->size - subgraph->size) +
            " on the CPU.";
        context->ReportError(context, error_message.c_str());
    }
    return subgraph;
}

#if 0
Status BuildModel(TfLiteContext* context,
                  const TfLiteDelegateParams* delegate_params,
                  TabeqGraph* graph) {
    std::vector<std::unique_ptr<TFLiteOperationParser>> operations;
    std::vector<int> tflite_nodes;
    for (int i = 0; i < delegate_params->nodes_to_replace->size; ++i) {
        TfLiteNode* tflite_node = nullptr;
        TfLiteRegistration* registration = nullptr;
        RETURN_IF_ERROR(GetNodeAndRegistration(
            context, delegate_params->nodes_to_replace->data[i], &tflite_node,
            &registration));
        if (registration->builtin_code == kTfLiteBuiltinDequantize) {
            // Ignore Dequantize nodes.
            continue;
        }
        auto op_parser = NewOperationParser(registration);
        if (!op_parser) {
            return UnimplementedError(
                absl::StrCat("Operation ", registration->builtin_code, "(",
                             registration->custom_name,
                             ") is not supported by TFLite GPU Delegate."));
        }
        operations.push_back(std::move(op_parser));
        tflite_nodes.push_back(i);
    }
  std::vector<Value<TensorRef>*> tensor_to_value(context->tensors_size,
  std::vector<Value<TensorRef<BHWC>>*> tensor_to_value(context->tensors_size,
                                                       nullptr);
  for (int i = 0; i < operations.size(); ++i) {
        TfLiteNode* tflite_node;
        TfLiteRegistration* registration;
        RETURN_IF_ERROR(GetNodeAndRegistration(
            context, delegate_params->nodes_to_replace->data[tflite_nodes[i]],
            &tflite_node, &registration));
        ObjectReader reader(graph, context, tflite_node, &tensor_to_value);
        const auto status =
            operations[i]->Parse(tflite_node, registration, graph, &reader);
        if (!status.ok()) {
            return InternalError(
                absl::StrCat(GetOpNameByRegistration(registration), ": ",
                             status.error_message()));
        }
  }
  return OkStatus();
}
#endif

}  // namespace tabeq
}  // namespace tflite
