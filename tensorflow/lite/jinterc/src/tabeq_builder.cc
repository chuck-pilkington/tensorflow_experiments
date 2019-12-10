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

int GetNumberOfRuntimeInputsForNode(const TfLiteContext* context,
                                    const TfLiteNode* tflite_node) {
    int number_of_runtime_inputs = 0;
    for (int i = 0; i < tflite_node->inputs->size; i++) {
        if (!IsConstantTensor(
                &context->tensors[tflite_node->inputs->data[i]])) {
            number_of_runtime_inputs++;
        }
    }
    return number_of_runtime_inputs;
}

int GetNumberOfRuntimeOutputsForNode(const TfLiteContext* context,
                                     const TfLiteNode* tflite_node) {
    int number_of_runtime_outputs = 0;
    for (int i = 0; i < tflite_node->outputs->size; i++) {
        if (!IsConstantTensor(
                &context->tensors[tflite_node->outputs->data[i]])) {
            number_of_runtime_outputs++;
        }
    }
    return number_of_runtime_outputs;
}

Status CheckTensorIsAvailable(const TfLiteContext* context,
                              const TfLiteNode* tflite_node, int idx) {
    // If tensor id is in range, it's guaranteed that it'll be available.
    if (idx >= tflite_node->inputs->size) {
        return OutOfRangeError(absl::StrFormat(
            "Requested index goes beyond array size (%d vs %d).", idx,
            tflite_node->inputs->data[idx]));
    }
    return OkStatus();
}

Status CheckMaxSupportedOpVersion(const TfLiteRegistration* registration,
                                  int max_version) {
    const int op_version = registration->version;
    if (op_version > max_version) {
        return UnimplementedError(
            absl::StrFormat("Max version supported: %d. Requested version %d.",
                            max_version, op_version));
    }
    return OkStatus();
}

Status ConvertTfLiteTensorToTensorRef(const TfLiteTensor& tflite_tensor,
                                      TensorRef* tensor_ref) {

    throw JintercException("ConvertTfLiteTensorToTensorRef not implemented");
    //   tensor_ref->type = ToDataType(tflite_tensor.type);
    //   return ExtractTensorShape(tflite_tensor, &tensor_ref->shape);
}

class ObjectReader {
   public:
    ObjectReader(TabeqGraph* graph, TfLiteContext* context,
                 const TfLiteNode* tflite_node,
                 std::vector<Value<TensorRef>*>* tensor_to_value)
        : graph_(graph),
          context_(context),
          tflite_node_(tflite_node),
          tensor_to_value_(tensor_to_value) {}

    Status ReadValue(uint32_t idx, Value<TensorRef>** value) const {
        if (idx >= tflite_node_->inputs->size) {
            return OutOfRangeError(
                absl::StrCat("ReadValue: input tensor index: ", idx));
        }
        return ReadValueByTensorIdx(tflite_node_->inputs->data[idx], value);
    }

    int GetNumberOfRuntimeInputs() const {
        return GetNumberOfRuntimeInputsForNode(context_, tflite_node_);
    }

    Status GetTensorDims(uint32_t idx, TfLiteIntArray* dimensions) const {
        if (idx >= tflite_node_->inputs->size) {
            return OutOfRangeError(absl::StrCat("Input tensor index: ", idx));
        }
        const int tensor_idx = tflite_node_->inputs->data[idx];
        if (tensor_idx < 0 || tensor_idx > context_->tensors_size) {
            return OutOfRangeError(absl::StrCat("Tensor index: ", tensor_idx));
        }
        const TfLiteTensor& tflite_tensor = context_->tensors[tensor_idx];
        *dimensions = *tflite_tensor.dims;
        return OkStatus();
    }

    template <typename TensorT>
    Status ReadTensor(uint32_t idx, TensorT* t) const {
        RETURN_IF_ERROR(CheckTensorIsAvailable(context_, tflite_node_, idx));
        const int32_t tensor_idx = tflite_node_->inputs->data[idx];
        const TfLiteTensor* tflite_tensor = context_->tensors + tensor_idx;
        t->data.resize(NumElements(tflite_tensor));
        RETURN_IF_ERROR(CreateVectorCopyData(*tflite_tensor, &t->data[0]));

        // Axis and data layout depend on operation this tensor is used in. So,
        // postpone resolutions until operations are parsed.
        t->id = tensor_idx;
        return SetAllDimensions(tflite_tensor->dims, &t->shape);
    }

    Status AddOutput(const Node* node, int id) {
        if (tflite_node_->outputs->size <= id) {
            return InvalidArgumentError(absl::StrCat(
                "Data id ", id, " must be less than tflite node outputs size ",
                tflite_node_->outputs->size));
        }
        int output_tensor_idx = tflite_node_->outputs->data[id];
        Value<TensorRef>* value;
        RETURN_IF_ERROR(ReadValueByTensorIdx(output_tensor_idx, &value));
        RETURN_IF_ERROR(graph_->SetProducer(node->id, value->id));
        return OkStatus();
    }

    Status AddOutputs(const Node* node) {
        for (int i = 0; i < tflite_node_->outputs->size; ++i) {
            RETURN_IF_ERROR(AddOutput(node, i));
        }
        return OkStatus();
    }

    Status AddInput(const Node* node, uint32_t idx) {
        Value<TensorRef>* input;
        RETURN_IF_ERROR(ReadValue(idx, &input));
        return graph_->AddConsumer(node->id, input->id);
    }

    Status ReadValueByTensorIdx(uint32_t tensor_idx,
                                Value<TensorRef>** value) const {
        if (tensor_idx >= tensor_to_value_->size()) {
            return OutOfRangeError(
                absl::StrCat("ReadValue: input tensor index: ", tensor_idx));
        }
        if ((*tensor_to_value_)[tensor_idx] == nullptr) {
            const TfLiteTensor& tflite_tensor = context_->tensors[tensor_idx];
            if (tflite::IsConstantTensor(&tflite_tensor)) {
                return NotFoundError(absl::StrCat(
                    "ReadValue: value is a constant tensor: ", tensor_idx));
            }
            Value<TensorRef>* value = graph_->NewValue();
            RETURN_IF_ERROR(
                ConvertTfLiteTensorToTensorRef(tflite_tensor, &value->tensor));
            value->tensor.ref = tensor_idx;
            (*tensor_to_value_)[tensor_idx] = value;
        }
        *value = (*tensor_to_value_)[tensor_idx];
        return OkStatus();
    }

    TfLiteTensor* GetInputTensor(int index) const {
        return index >= 0 && index < tflite_node_->inputs->size
                   ? context_->tensors + tflite_node_->inputs->data[index]
                   : nullptr;
    }

    TfLiteTensor* GetOutputTensor(int index) const {
        return index >= 0 && index < tflite_node_->outputs->size
                   ? context_->tensors + tflite_node_->outputs->data[index]
                   : nullptr;
    }

   private:
    TabeqGraph* graph_ = nullptr;
    const TfLiteContext* context_ = nullptr;
    const TfLiteNode* tflite_node_ = nullptr;
    std::vector<Value<TensorRef>*>* tensor_to_value_;
};

typedef int FullyConnectedAttributes;

Status GetFullyConnectedAttributes(int weights_tensor_id, int bias_tensor_id,
                                   ObjectReader* reader,
                                   FullyConnectedAttributes* attr) {

#if 1
    throw JintercException("GetFullyConnectedAttributes not supported");
#else
    Tensor<HW, DataType::FLOAT32> weights;
    RETURN_IF_ERROR(reader->ReadTensor(weights_tensor_id, &weights));
    attr->weights.data = std::move(weights.data);
    attr->weights.id = weights.id;
    attr->weights.shape.h = 1;
    attr->weights.shape.w = 1;
    attr->weights.shape.o = weights.shape.h;
    attr->weights.shape.i = weights.shape.w;
    reader->ReadTensor(bias_tensor_id, &attr->bias).IgnoreError();  // optional

    return OkStatus();
#endif
}

template <typename ParamsT>
Status RetrieveBuiltinData(const TfLiteNode* tflite_node,
                           ParamsT** tf_options) {
    const auto* params =
        reinterpret_cast<const ParamsT*>(tflite_node->builtin_data);
    if (!params) {
        return InternalError("Unable to retrieve builtin_data.");
    }
    *tf_options = const_cast<ParamsT*>(params);
    return OkStatus();
}

// A parser responsible for parsing TFLite operation and adding it to a graph.
class TFLiteOperationParser {
   public:
    virtual ~TFLiteOperationParser() = default;

    // Parses TFLite operation. This method allows expanding fused operations
    // into more than one node.
    virtual Status Parse(const TfLiteNode* tflite_node,
                         const TfLiteRegistration* registration,
                         TabeqGraph* graph, ObjectReader* reader) = 0;

    // Verifies whether passed tflite node may be built by GPU delegate or not.
    virtual Status IsSupported(const TfLiteContext* context,
                               const TfLiteNode* tflite_node,
                               const TfLiteRegistration* registration) = 0;
};

class FullyConnectedOperationParser : public TFLiteOperationParser {
   public:
    Status IsSupported(const TfLiteContext* context,
                       const TfLiteNode* tflite_node,
                       const TfLiteRegistration* registration) final {
        RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 1));
        TfLiteFullyConnectedParams* tf_options = nullptr;
        RETURN_IF_ERROR(RetrieveBuiltinData(tflite_node, &tf_options));
        if (tf_options->weights_format !=
            kTfLiteFullyConnectedWeightsFormatDefault) {
            return UnimplementedError(
                "Unsupported FullyConnected weights format.");
        }
        // TODO(eignasheva): check input shape
        return OkStatus();
    }

    Status Parse(const TfLiteNode* tflite_node,
                 const TfLiteRegistration* registration, TabeqGraph* graph,
                 ObjectReader* reader) final {
        Node* node = graph->NewNode();
        RETURN_IF_ERROR(reader->AddInput(node, 0));

        const auto* tf_options =
            reinterpret_cast<const TfLiteFullyConnectedParams*>(
                tflite_node->builtin_data);
        if (tf_options->weights_format !=
            kTfLiteFullyConnectedWeightsFormatDefault) {
            return UnimplementedError(
                "Unsupported FullyConnected weights format.");
        }

        FullyConnectedAttributes attr;
        RETURN_IF_ERROR(GetFullyConnectedAttributes(1, 2, reader, &attr));

#if 1
        throw JintercException("FC Parse is WIP");
#else
        // Tensor<HW, DataType::FLOAT32> weights;
        TensorRef weights;
        RETURN_IF_ERROR(reader->ReadTensor(1, &weights));
        auto input = graph->FindInputs(node->id)[0];
        int batch_size = input->tensor.shape.b;
        if (input->tensor.shape.DimensionsProduct() / batch_size !=
            weights.shape.w) {
            return UnimplementedError(
                "Amount of input data should match weights width");
        }

        Node* conv = node;
        if (input->tensor.shape.h != 1 || input->tensor.shape.w != 1) {
            auto& reshape = node;
            conv = graph->NewNode();  // reset conv pointer!
            Value<TensorRef>* reshaped_value = graph->NewValue();
            reshaped_value->tensor.shape = BHWC(1, 1, 1, weights.shape.w);
            RETURN_IF_ERROR(
                graph->SetProducer(reshape->id, reshaped_value->id));
            reshape->operation.type = ToString(OperationType::RESHAPE);
            ReshapeAttributes attr;
            attr.new_shape = reshaped_value->tensor.shape;
            reshape->operation.attributes = attr;
            RETURN_IF_ERROR(graph->AddConsumer(conv->id, reshaped_value->id));
        }

        conv->operation.type = ToString(OperationType::FULLY_CONNECTED);
        conv->operation.attributes = std::move(attr);
        Status result = reader->AddOutputs(conv);
        RETURN_IF_ERROR(MaybeFuseActivationToTheSingleOutput(
            tf_options->activation, graph, conv));

        return result;
#endif
    }
};

class UnsupportedOperationParser : public TFLiteOperationParser {
   public:
    Status IsSupported(const TfLiteContext* context,
                       const TfLiteNode* tflite_node,
                       const TfLiteRegistration* registration) final {
        return UnimplementedError("Operation is not supported.");
    }

    Status Parse(const TfLiteNode* tflite_node,
                 const TfLiteRegistration* registration, TabeqGraph* graph,
                 ObjectReader* reader) final {
        return UnimplementedError("Operation is not supported.");
    }
};

std::unique_ptr<TFLiteOperationParser> NewOperationParser(
    const TfLiteRegistration* registration) {
    const auto builtin_code = registration->builtin_code;
    const absl::string_view custom_name = registration->custom_name;
    switch (builtin_code) {

        case kTfLiteBuiltinFullyConnected:
            return absl::make_unique<FullyConnectedOperationParser>();

#if 0  // -- TODO: Implement these!
       //
        case kTfLiteBuiltinAbs:
            return absl::make_unique<ElementwiseOperationParser>(
                OperationType::ABS);
        case kTfLiteBuiltinAdd:
            return absl::make_unique<AddOperationParser>();
        case kTfLiteBuiltinAveragePool2d:
            return absl::make_unique<Pooling2DOperationParser>(
                PoolingType::AVERAGE);
        case kTfLiteBuiltinConcatenation:
            return absl::make_unique<ConcatenationOperationParser>();
        case kTfLiteBuiltinConv2d:
            return absl::make_unique<Conv2DOperationParser>();
        case kTfLiteBuiltinCos:
            return absl::make_unique<ElementwiseOperationParser>(
                OperationType::COS);
        case kTfLiteBuiltinDepthwiseConv2d:
            return absl::make_unique<DepthwiseConvolutionOperationParser>();
        case kTfLiteBuiltinDiv:
            return absl::make_unique<ElementwiseOperationParser>(
                OperationType::DIV);
        case kTfLiteBuiltinFullyConnected:
            return absl::make_unique<FullyConnectedOperationParser>();
        case kTfLiteBuiltinHardSwish:
            return absl::make_unique<HardSwishOperationParser>();
        case kTfLiteBuiltinLogistic:
            return absl::make_unique<ElementwiseOperationParser>(
                OperationType::SIGMOID);
        case kTfLiteBuiltinLog:
            return absl::make_unique<ElementwiseOperationParser>(
                OperationType::LOG);
        case kTfLiteBuiltinLstm:
            return absl::make_unique<LSTMOperationParser>();
        case kTfLiteBuiltinMaxPool2d:
            return absl::make_unique<Pooling2DOperationParser>(
                PoolingType::MAX);
        case kTfLiteBuiltinMul:
            return absl::make_unique<MulOperationParser>();
        case kTfLiteBuiltinPad:
            return absl::make_unique<PadOperationParser>();
        case kTfLiteBuiltinPow:
            return absl::make_unique<ElementwiseOperationParser>(
                OperationType::POW);
        case kTfLiteBuiltinRelu:
            return absl::make_unique<ReLUOperationParser>(0);
        case kTfLiteBuiltinRelu6:
            return absl::make_unique<ReLUOperationParser>(6);
        case kTfLiteBuiltinLeakyRelu:
            return absl::make_unique<ReLUOperationParser>(0);
        case kTfLiteBuiltinPrelu:
            return absl::make_unique<PReLUOperationParser>();
        case kTfLiteBuiltinReshape:
            return absl::make_unique<ReshapeOperationParser>();
        case kTfLiteBuiltinResizeBilinear:
            return absl::make_unique<ResizeBilinearOperationParser>();
        case kTfLiteBuiltinRsqrt:
            return absl::make_unique<ElementwiseOperationParser>(
                OperationType::RSQRT);
        case kTfLiteBuiltinSin:
            return absl::make_unique<ElementwiseOperationParser>(
                OperationType::SIN);
        case kTfLiteBuiltinSoftmax:
            return absl::make_unique<SoftmaxOperationParser>();
        case kTfLiteBuiltinSlice:
            return absl::make_unique<SliceOperationParser>();
        case kTfLiteBuiltinStridedSlice:
            return absl::make_unique<StridedSliceOperationParser>();
        case kTfLiteBuiltinSqrt:
            return absl::make_unique<ElementwiseOperationParser>(
                OperationType::SQRT);
        case kTfLiteBuiltinSquare:
            return absl::make_unique<ElementwiseOperationParser>(
                OperationType::SQUARE);
        case kTfLiteBuiltinSquaredDifference:
            return absl::make_unique<ElementwiseOperationParser>(
                OperationType::SQUARED_DIFF);
        case kTfLiteBuiltinSub:
            return absl::make_unique<ElementwiseOperationParser>(
                OperationType::SUB);
        case kTfLiteBuiltinTanh:
            return absl::make_unique<ElementwiseOperationParser>(
                OperationType::TANH);
        case kTfLiteBuiltinTranspose:
            return absl::make_unique<TransposeOperationParser>();
        case kTfLiteBuiltinTransposeConv:
            return absl::make_unique<TransposeConvOperationParser>();

        case kTfLiteBuiltinCustom:
            if (custom_name == "Convolution2DTransposeBias") {
                return absl::make_unique<Convolution2DTransposeBiasParser>();
            }
            if (custom_name == "MaxPoolingWithArgmax2D") {
                return absl::make_unique<Pooling2DOperationParser>(
                    PoolingType::MAX);
            }
            if (custom_name == "MaxUnpooling2D") {
                return absl::make_unique<Unpooling2DOperationParser>();
            }
            break;
#endif
    }
    return absl::make_unique<UnsupportedOperationParser>();
}

Status IsSupported(const TfLiteContext* context, TfLiteNode* node,
                   const TfLiteRegistration* registration) {

    return NewOperationParser(registration)
        ->IsSupported(context, node, registration);
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
