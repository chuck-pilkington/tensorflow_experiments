/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/internal/optimized/integer_ops/fully_connected.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <tabeq/tabeq.h>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace jinterc {
namespace fully_connected {

// This file has only the reference version of the jinterc FullyConnected
enum KernelType {
    kReference,
    kJinterc,
};

struct OpData {
    // The scaling factor from input to output (aka the 'real multiplier') can
    // be represented as a fixed point multiplier plus a left shift.
    int32_t output_multiplier;
    int output_shift;
    // The range of the fused activation layer. For example for kNone and
    // uint8_t these would be 0 and 255.
    int32_t output_activation_min;
    int32_t output_activation_max;
    // The index of the temporary tensor where the quantized inputs are cached.
    int scratch_tensor_index;

    tabeq::runtime::SubGraph* hwImpl = nullptr;
};

constexpr int kInputTensor = 0;
constexpr int kWeightsTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;
constexpr int kShuffledInputWorkspaceTensor = 1;

inline TfLiteStatus CheckTypes(TfLiteContext* context,
                               const TfLiteTensor* input,
                               const TfLiteTensor* filter,
                               const TfLiteTensor* bias, TfLiteTensor* output,
                               TfLiteFullyConnectedParams* params) {
    const bool is_quantized =
        ((filter->type == kTfLiteUInt8) || (filter->type == kTfLiteInt8));
    const bool is_hybrid = is_quantized && (input->type == kTfLiteFloat32);
    const bool is_shuffled =
        is_quantized && (params->weights_format ==
                         kTfLiteFullyConnectedWeightsFormatShuffled4x16Int8);

    // optional bias tensor.
    const bool is_optional_bias_float = !bias || (bias->type == kTfLiteFloat32);
    const bool is_optional_bias_int = !bias || (bias->type == kTfLiteInt32);

    if (is_quantized) {
        if (is_shuffled) {
            TF_LITE_ENSURE_EQ(context, input->type, kTfLiteUInt8);
            TF_LITE_ENSURE_EQ(context, filter->type, kTfLiteUInt8);
            TF_LITE_ENSURE_EQ(context, output->type, kTfLiteInt16);
            TF_LITE_ENSURE_EQ(context, is_optional_bias_int, true);
        } else if (is_hybrid) {
            TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
            TF_LITE_ENSURE_EQ(context, output->type, kTfLiteFloat32);
            TF_LITE_ENSURE_EQ(context, is_optional_bias_float, true);
        } else {
            TF_LITE_ENSURE(context, input->type == kTfLiteUInt8 ||
                                        input->type == kTfLiteInt8);
            TF_LITE_ENSURE(context, output->type == kTfLiteUInt8 ||
                                        output->type == kTfLiteInt8 ||
                                        output->type == kTfLiteInt16);
            TF_LITE_ENSURE_EQ(context, is_optional_bias_int, true);
        }
    } else {
        // Only float32 is supported currently
        TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
        TF_LITE_ENSURE_EQ(context, output->type, kTfLiteFloat32);
        TF_LITE_ENSURE_EQ(context, filter->type, kTfLiteFloat32);
        TF_LITE_ENSURE_EQ(context, is_optional_bias_float, true);
    }

    return kTfLiteOk;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
    // This is a builtin op, so we don't use the contents in 'buffer', if any.
    // Instead, we allocate a new object to carry information from Prepare() to
    // Eval().
    auto* op_data = new OpData();
    context->AddTensors(context, /*tensors_to_add=*/2,
                        &op_data->scratch_tensor_index);
    return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
    delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {

    printf("JINTERC FC: Running prepare stage...\n");

    auto* params =
        reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
    OpData* data = reinterpret_cast<OpData*>(node->user_data);

    // Check we have all the inputs and outputs we need.
    TF_LITE_ENSURE(context, node->inputs->size == 2 || node->inputs->size == 3);
    // Shuffled formats need a workspace to store the shuffled input
    // activations.
    const int expected_outputs_count =
        params->weights_format == kTfLiteFullyConnectedWeightsFormatDefault ? 1
                                                                            : 2;
    TF_LITE_ENSURE_EQ(context, node->outputs->size, expected_outputs_count);

    const TfLiteTensor* input = GetInput(context, node, kInputTensor);
    const TfLiteTensor* filter = GetInput(context, node, kWeightsTensor);
    const TfLiteTensor* bias =
        (node->inputs->size == 3)
            ? GetOptionalInputTensor(context, node, kBiasTensor)
            : nullptr;
    TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

    // Check proper datatype match among all Input Tensors
    TF_LITE_ENSURE_STATUS(
        CheckTypes(context, input, filter, bias, output, params));

    // Check all the parameters of tensor match within themselves and match the
    // input configuration.
    int input_size = 1;
    for (int i = 0; i < input->dims->size; i++) {
        input_size *= input->dims->data[i];
    }

    TF_LITE_ENSURE_EQ(context, NumDimensions(filter), 2);
    const int batch_size = input_size / filter->dims->data[1];
    const int num_units = filter->dims->data[0];

    if (bias) {
        TF_LITE_ENSURE_EQ(context, NumElements(bias),
                          SizeOfDimension(filter, 0));
    }

    // Note that quantized inference requires that all tensors have their
    // parameters set. This is usually done during quantized training.
    if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8) {
        double real_multiplier = 0.0;
        TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
            context, input, filter, bias, output, &real_multiplier));
        int exponent;
        QuantizeMultiplier(real_multiplier, &data->output_multiplier,
                           &exponent);
        data->output_shift = exponent;
        TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
            context, params->activation, output, &data->output_activation_min,
            &data->output_activation_max));
    }

    // If we have to perform on-the-fly quantization (with quantized weights and
    // float inputs) first we need to quantize the inputs. Allocate a temporary
    // buffer to store the intermediate quantized values.
    if (input->type == kTfLiteFloat32 &&
        (filter->type == kTfLiteUInt8 || filter->type == kTfLiteInt8)) {
        TfLiteIntArrayFree(node->temporaries);
        node->temporaries = TfLiteIntArrayCreate(2);
        node->temporaries->data[0] = data->scratch_tensor_index;

        TfLiteTensor* input_quantized =
            GetTemporary(context, node, /*index=*/0);
        input_quantized->type = filter->type;
        input_quantized->allocation_type = kTfLiteArenaRw;

        TfLiteIntArray* input_quantized_size = TfLiteIntArrayCopy(input->dims);
        TF_LITE_ENSURE_OK(context,
                          context->ResizeTensor(context, input_quantized,
                                                input_quantized_size));

        node->temporaries->data[1] = data->scratch_tensor_index + 1;
        TfLiteTensor* scaling_factors =
            GetTemporary(context, node, /*index=*/1);
        scaling_factors->type = kTfLiteFloat32;
        scaling_factors->allocation_type = kTfLiteArenaRw;
        int scaling_dims[1] = {batch_size};
        if (!TfLiteIntArrayEqualsArray(scaling_factors->dims, 1,
                                       scaling_dims)) {
            TfLiteIntArray* scaling_factors_size = TfLiteIntArrayCreate(1);
            scaling_factors_size->data[0] = batch_size;
            TF_LITE_ENSURE_OK(context,
                              context->ResizeTensor(context, scaling_factors,
                                                    scaling_factors_size));
        }
    }

    // Resize output.
    TfLiteIntArray* output_size_array = nullptr;
    if (params->keep_num_dims) {
        // When number of dimensions are kept the filter operates along the last
        // dimensions. In other words, for an input tensor with shape
        // [batch_size, ..., n_inputs] and a filter of shape [n_inputs, n_units]
        // this Op produces an output of shape [batch_size, ..., n_units].
        TF_LITE_ENSURE_EQ(context, input->dims->data[input->dims->size - 1],
                          SizeOfDimension(filter, 1));
        output_size_array = TfLiteIntArrayCopy(input->dims);
        output_size_array->data[output_size_array->size - 1] = num_units;
    } else {
        // Otherwise, the output is (potentially flattened to) a 2-D matrix.
        output_size_array = TfLiteIntArrayCreate(2);
        output_size_array->data[0] = batch_size;
        output_size_array->data[1] = num_units;
    }
    TF_LITE_ENSURE_OK(
        context, context->ResizeTensor(context, output, output_size_array));

    /**
     * First quick hook into tabeq
     */
    auto& g = *new tabeq::runtime::SubGraph();

    tabeq::runtime::Operator& o = g.allocOp();
    o.code.builtin_code = tabeq::runtime::FULLY_CONNECTED;

    return kTfLiteOk;
}

namespace {
template <KernelType kernel_type>
void FullyConnectedInt8(const OpData* data, const TfLiteTensor* input,
                        const TfLiteTensor* filter, const TfLiteTensor* bias,
                        TfLiteTensor* output,
                        CpuBackendContext* cpu_backend_context) {
    FullyConnectedParams op_params;
    op_params.input_offset = -input->params.zero_point;
    op_params.weights_offset = -filter->params.zero_point;
    op_params.output_offset = output->params.zero_point;
    op_params.output_multiplier = data->output_multiplier;
    op_params.output_shift = data->output_shift;
    op_params.quantized_activation_min = data->output_activation_min;
    op_params.quantized_activation_max = data->output_activation_max;
    if (kernel_type == kReference) {
        reference_integer_ops::FullyConnected(
            op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
            GetTensorShape(filter), GetTensorData<int8_t>(filter),
            GetTensorShape(bias), GetTensorData<int32_t>(bias),
            GetTensorShape(output), GetTensorData<int8_t>(output));
    } else {
        optimized_integer_ops::FullyConnected(
            op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
            GetTensorShape(filter), GetTensorData<int8_t>(filter),
            GetTensorShape(bias), GetTensorData<int32_t>(bias),
            GetTensorShape(output), GetTensorData<int8_t>(output),
            cpu_backend_context);
    }
}
}  // namespace

template <KernelType kernel_type>
TfLiteStatus EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                           TfLiteFullyConnectedParams* params, OpData* data,
                           const TfLiteTensor* input,
                           const TfLiteTensor* filter, const TfLiteTensor* bias,
                           TfLiteTensor* output) {
    int32_t input_offset = -input->params.zero_point;
    int32_t filter_offset = -filter->params.zero_point;
    int32_t output_offset = output->params.zero_point;

    // Only the Pie path supports quantized models and float inputs/outputs.
    TF_LITE_ENSURE(context, input->type != kTfLiteFloat32);

    FullyConnectedParams op_params;
    op_params.input_offset = input_offset;
    op_params.weights_offset = filter_offset;
    op_params.output_offset = output_offset;
    op_params.output_multiplier = data->output_multiplier;
    op_params.output_shift = data->output_shift;
    op_params.quantized_activation_min = data->output_activation_min;
    op_params.quantized_activation_max = data->output_activation_max;
    switch (output->type) {
        case kTfLiteUInt8:
            if (kernel_type == kReference) {
                reference_ops::FullyConnected(
                    op_params, GetTensorShape(input),
                    GetTensorData<uint8_t>(input), GetTensorShape(filter),
                    GetTensorData<uint8_t>(filter), GetTensorShape(bias),
                    GetTensorData<int32_t>(bias), GetTensorShape(output),
                    GetTensorData<uint8_t>(output));
            } else {
                optimized_ops::FullyConnected(
                    op_params, GetTensorShape(input),
                    GetTensorData<uint8_t>(input), GetTensorShape(filter),
                    GetTensorData<uint8_t>(filter), GetTensorShape(bias),
                    GetTensorData<int32_t>(bias), GetTensorShape(output),
                    GetTensorData<uint8_t>(output),
                    CpuBackendContext::GetFromContext(context));
            }
            break;
        case kTfLiteInt8:
            FullyConnectedInt8<kernel_type>(
                data, input, filter, bias, output,
                CpuBackendContext::GetFromContext(context));
            break;
        case kTfLiteInt16:
            if (kernel_type == kReference) {
                reference_ops::FullyConnected(
                    op_params, GetTensorShape(input),
                    GetTensorData<uint8_t>(input), GetTensorShape(filter),
                    GetTensorData<uint8_t>(filter), GetTensorShape(bias),
                    GetTensorData<int32_t>(bias), GetTensorShape(output),
                    GetTensorData<int16_t>(output));
            } else {
                optimized_ops::FullyConnected(
                    op_params, GetTensorShape(input),
                    GetTensorData<uint8_t>(input), GetTensorShape(filter),
                    GetTensorData<uint8_t>(filter), GetTensorShape(bias),
                    GetTensorData<int32_t>(bias), GetTensorShape(output),
                    GetTensorData<int16_t>(output),
                    CpuBackendContext::GetFromContext(context));
            }
            break;
        default:
            context->ReportError(context,
                                 "Quantized FullyConnected expects output data "
                                 "type uint8, int8 or int16");
            return kTfLiteError;
    }

    return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus EvalShuffledQuantized(TfLiteContext* context, TfLiteNode* node,
                                   TfLiteFullyConnectedParams* params,
                                   OpData* data, const TfLiteTensor* input,
                                   const TfLiteTensor* filter,
                                   const TfLiteTensor* bias,
                                   TfLiteTensor* output,
                                   TfLiteTensor* shuffled_input_workspace) {
    // TODO(b/110697972) decide more consistently if / how / where we want
    // to perform this kind of runtime data type checks.
    if (shuffled_input_workspace->type != kTfLiteUInt8) {
        context->ReportError(context, "Unexpected data type");
        return kTfLiteError;
    }

#define TF_LITE_SHUFFLED_FULLY_CONNECTED(type)                               \
    {                                                                        \
        type::ShuffledFullyConnected(                                        \
            op_params, GetTensorShape(input), GetTensorData<uint8_t>(input), \
            GetTensorShape(filter), GetTensorData<uint8_t>(filter),          \
            GetTensorShape(bias), GetTensorData<int32_t>(bias),              \
            GetTensorShape(output), GetTensorData<int16_t>(output),          \
            GetTensorData<uint8_t>(shuffled_input_workspace),                \
            CpuBackendContext::GetFromContext(context));                     \
    }
    FullyConnectedParams op_params;
    op_params.output_multiplier = data->output_multiplier;
    op_params.output_shift = data->output_shift;
    op_params.quantized_activation_min = data->output_activation_min;
    op_params.quantized_activation_max = data->output_activation_max;
    if (kernel_type == kReference) {
        reference_ops::ShuffledFullyConnected(
            op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
            GetTensorShape(filter), GetTensorData<uint8_t>(filter),
            GetTensorShape(bias), GetTensorData<int32_t>(bias),
            GetTensorShape(output), GetTensorData<int16_t>(output),
            GetTensorData<uint8_t>(shuffled_input_workspace));
    } else {
        optimized_ops::ShuffledFullyConnected(
            op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
            GetTensorShape(filter), GetTensorData<uint8_t>(filter),
            GetTensorShape(bias), GetTensorData<int32_t>(bias),
            GetTensorShape(output), GetTensorData<int16_t>(output),
            GetTensorData<uint8_t>(shuffled_input_workspace),
            CpuBackendContext::GetFromContext(context));
    }
#undef TF_LITE_SHUFFLED_FULLY_CONNECTED

    return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus EvalFloat(TfLiteContext* context, TfLiteNode* node,
                       TfLiteFullyConnectedParams* params, OpData* data,
                       const TfLiteTensor* input, const TfLiteTensor* filter,
                       const TfLiteTensor* bias, TfLiteTensor* output) {
    float output_activation_min, output_activation_max;
    CalculateActivationRange(params->activation, &output_activation_min,
                             &output_activation_max);
    if (kernel_type == kReference) {
        FullyConnectedParams op_params;
        op_params.float_activation_min = output_activation_min;
        op_params.float_activation_max = output_activation_max;
        reference_ops::FullyConnected(
            op_params, GetTensorShape(input), GetTensorData<float>(input),
            GetTensorShape(filter), GetTensorData<float>(filter),
            GetTensorShape(bias), GetTensorData<float>(bias),
            GetTensorShape(output), GetTensorData<float>(output));
    } else {
        FullyConnectedParams op_params;
        op_params.float_activation_min = output_activation_min;
        op_params.float_activation_max = output_activation_max;
        optimized_ops::FullyConnected(
            op_params, GetTensorShape(input), GetTensorData<float>(input),
            GetTensorShape(filter), GetTensorData<float>(filter),
            GetTensorShape(bias), GetTensorData<float>(bias),
            GetTensorShape(output), GetTensorData<float>(output),
            CpuBackendContext::GetFromContext(context));
    }

    return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {

    printf("Hello, in the JINTERC fully connect eval function...\n");

    auto* params =
        reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
    OpData* data = reinterpret_cast<OpData*>(node->user_data);

    const TfLiteTensor* input = GetInput(context, node, kInputTensor);
    const TfLiteTensor* filter = GetInput(context, node, kWeightsTensor);
    const TfLiteTensor* bias =
        (node->inputs->size == 3)
            ? GetOptionalInputTensor(context, node, kBiasTensor)
            : nullptr;
    TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

    switch (filter->type) {
        case kTfLiteFloat32:
            return EvalFloat<kernel_type>(context, node, params, data, input,
                                          filter, bias, output);
        case kTfLiteUInt8:
            if (params->weights_format ==
                kTfLiteFullyConnectedWeightsFormatShuffled4x16Int8) {
                TfLiteTensor* shuffled_input_workspace =
                    GetOutput(context, node, kShuffledInputWorkspaceTensor);
                return EvalShuffledQuantized<kernel_type>(
                    context, node, params, data, input, filter, bias, output,
                    shuffled_input_workspace);
            } else if (params->weights_format ==
                       kTfLiteFullyConnectedWeightsFormatDefault) {
                return EvalQuantized<kernel_type>(context, node, params, data,
                                                  input, filter, bias, output);
            } else {
                context->ReportError(
                    context, "Unhandled fully-connected weights format");
                return kTfLiteError;
            }
        case kTfLiteInt8:
            if (params->weights_format ==
                kTfLiteFullyConnectedWeightsFormatDefault) {
                return EvalQuantized<kernel_type>(context, node, params, data,
                                                  input, filter, bias, output);
            } else {
                context->ReportError(
                    context, "Unhandled fully-connected weights format");
                return kTfLiteError;
            }
        default:
            context->ReportError(context,
                                 "Filter data type %s currently not supported.",
                                 TfLiteTypeGetName(filter->type));
            return kTfLiteError;
    }
    return kTfLiteOk;
}

}  // namespace fully_connected

TfLiteRegistration* Register_FULLY_CONNECTED_JINTERC() {
    static TfLiteRegistration r = {
        fully_connected::Init, fully_connected::Free, fully_connected::Prepare,
        fully_connected::Eval<fully_connected::kReference>};
    return &r;
}

}  // namespace jinterc
}  // namespace ops
}  // namespace tflite
