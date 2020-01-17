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

#include "tensorflow/lite/jinterc/include/jinterc_delegate.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <tabeq/model_transformer.h>
#include <tabeq/status.h>
#include <tabeq/transformations.h>

#include "tensorflow/lite/builtin_ops.h"

#include "tensorflow/lite/jinterc/include/jinterc.h"

#if 0
#include "tensorflow/lite/delegates/jinterc/api.h"
#include "tensorflow/lite/delegates/jinterc/cl/api.h"
#include "tensorflow/lite/delegates/jinterc/cl/opencl_wrapper.h"
#include "tensorflow/lite/delegates/jinterc/cl/tensor_type_util.h"
#include "tensorflow/lite/delegates/jinterc/common/model.h"
#include "tensorflow/lite/delegates/jinterc/common/model_builder.h"
#include "tensorflow/lite/delegates/jinterc/common/model_transformer.h"
#include "tensorflow/lite/delegates/jinterc/common/status.h"
#include "tensorflow/lite/delegates/jinterc/common/transformations/general_transformations.h"
#include "tensorflow/lite/delegates/jinterc/gl/api2.h"
#endif

#include "tensorflow/lite/minimal_logging.h"

#include <internal/tabeq_builder.h>

namespace tflite {
namespace jinterc {

using namespace ::tflite::tabeq;
using namespace ::tabeq;

InferencePriority ToPriority(int32_t priority) {
    switch (priority) {
        case TFLITE_JINTERC_INFERENCE_PRIORITY_MAX_PRECISION:
            return InferencePriority::MAX_PRECISION;
        case TFLITE_JINTERC_INFERENCE_PRIORITY_MIN_LATENCY:
            return InferencePriority::MIN_LATENCY;
        case TFLITE_JINTERC_INFERENCE_PRIORITY_MIN_MEMORY_USAGE:
            return InferencePriority::MIN_MEMORY_USAGE;
    }
    return InferencePriority::UNKNOWN;
}

InferenceUsage ToUsage(int32_t usage) {
    switch (usage) {
        case TFLITE_JINTERC_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER:
            return InferenceUsage::FAST_SINGLE_ANSWER;
        case TFLITE_JINTERC_INFERENCE_PREFERENCE_SUSTAINED_SPEED:
            return InferenceUsage::SUSTAINED_SPEED;
    }
    return InferenceUsage::UNKNOWN;
}

int GetPriorityPosition(const TfLiteJintercDelegateOptions& options,
                        InferencePriority p) {
    if (ToPriority(options.inference_priority1) == p) return 1;
    if (ToPriority(options.inference_priority2) == p) return 2;
    if (ToPriority(options.inference_priority3) == p) return 3;
    return 4;  // least important
}

// Forward declarations.
TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate);

class Delegate {
   public:
    explicit Delegate(const TfLiteJintercDelegateOptions* options) {
        options_ = options ? *options : TfLiteJintercDelegateOptionsDefault();
    }

    Status Prepare(TfLiteContext* context,
                   const TfLiteDelegateParams* delegate_params) {

        // Extract TFLite delegate execution plan from the context and convert
        // it into a TabeqGraph.
        TabeqGraph graph;
        RETURN_IF_ERROR(BuildModel(context, delegate_params, &graph));

        // First apply general transformations on the graph.
        NullTransformationReporter reporter;
        ModelTransformer transformer(&graph, &reporter);

        if (!ApplyGeneralTransformations(&transformer)) {
            return InternalError("Graph general transformations failed");
        }

        // -- Next do Tabeq-specific transformations
        //
        DebugTransformationReporter debugReporter;
        ModelTransformer tabeqTransformer(&graph, &debugReporter);

        if (!ApplyTabeqTransformations(&tabeqTransformer)) {
            return InternalError("Graph Tabeq transformations failed");
        }

        std::vector<uint32_t> input_refs;
        {
            const auto& inputs = graph.inputs();
            input_refs.reserve(inputs.size());
            for (auto input : inputs) {
                input_refs.push_back(input->tensor.ref);
            }
        }
        std::vector<uint32_t> output_refs;
        {
            const auto& outputs = graph.outputs();
            output_refs.reserve(outputs.size());
            for (auto output : outputs) {
                output_refs.push_back(output->tensor.ref);
            }
        }

        std::unique_ptr<::tabeq::InferenceBuilder> builder;

        RETURN_IF_ERROR(::tabeq::InitializeBuilder(std::move(graph), &builder));

#if 0
        Status status = InitializeOpenClApi(&graph, &builder);
        if (!status.ok()) {
            context->ReportError(context, "%s", status.error_message().c_str());
            context->ReportError(context, "Falling back to OpenGL");
            RETURN_IF_ERROR(InitializeOpenGlApi(&graph, &builder));
        }
#endif

        // At this point tflite didn't allocate tensors yet, therefore, collect
        // indices and set all input and output tensors from tflite later.
        input_indices_.reserve(input_refs.size());
        for (uint32_t tensor_index : input_refs) {
            const int64_t object_index = input_indices_.size();
            input_indices_.push_back(tensor_index);
            RETURN_IF_ERROR(builder->SetInputObjectDef(
                object_index, GetObjectDef(tensor_index)));
        }
        output_indices_.reserve(output_refs.size());
        for (uint32_t tensor_index : output_refs) {
            const int64_t object_index = output_indices_.size();
            output_indices_.push_back(tensor_index);
            RETURN_IF_ERROR(builder->SetOutputObjectDef(
                object_index, GetObjectDef(tensor_index)));
        }

        return builder->Build(&runner_);
    }

    Status SetInputsAndOutputs(TfLiteContext* context) {

        int i = 0;
        for (auto index : input_indices_) {
            RETURN_IF_ERROR(
                runner_->SetInputObject(i++, GetTensorObject(index, context)));
        }
        i = 0;
        for (auto index : output_indices_) {
            RETURN_IF_ERROR(
                runner_->SetOutputObject(i++, GetTensorObject(index, context)));
        }

        return OkStatus();
    }

    Status Invoke(TfLiteContext* context) {
        RETURN_IF_ERROR(SetInputsAndOutputs(context));

        return runner_->Run();
    }

    ObjectDef GetObjectDef(int index) const {
        ObjectDef default_object_def;
        default_object_def.data_type = DataType::FLOAT32;
        default_object_def.data_layout = DataLayout::BHWC;
        default_object_def.object_type = ObjectType::CPU_MEMORY;
        default_object_def.user_provided = true;
        return default_object_def;
    }

    TensorObject GetTensorObject(int index, TfLiteContext* context) const {
        auto& tensor = context->tensors[index];

        return TensorObject(tensor.data.raw, tensor.bytes);
    }

    TfLiteDelegate* tflite_delegate() { return &delegate_; }

   private:
    TfLiteDelegate delegate_ = {
        reinterpret_cast<void*>(this),  // .data_
        DelegatePrepare,                // .Prepare
        nullptr,                        // .CopyFromBufferHandle
        nullptr,                        // .CopyToBufferHandle
        nullptr,                        // .FreeBufferHandle
        kTfLiteDelegateFlagsNone,       // .flags
    };

    TfLiteJintercDelegateOptions options_;

    std::unique_ptr<InferenceRunner> runner_;

    std::vector<int64_t> input_indices_;
    std::vector<int64_t> output_indices_;
};

inline Delegate* GetDelegate(TfLiteNode* node) {
    return reinterpret_cast<Delegate*>(node->user_data);
}

inline Delegate* GetDelegate(TfLiteDelegate* delegate) {
    return reinterpret_cast<Delegate*>(delegate->data_);
}

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
    const TfLiteRegistration kRegistration = {
        // .init
        [](TfLiteContext* context, const char* buffer, size_t) -> void* {
            const auto* params =
                reinterpret_cast<const TfLiteDelegateParams*>(buffer);
            auto* jinterc_delegate = GetDelegate(params->delegate);
            // Everything below should happen in prepare function call, but
            // TFLite for whatever reason forbids that.
            const auto status = jinterc_delegate->Prepare(context, params);
            if (!status.ok()) {
                context->ReportError(context, "TfLiteJintercDelegate Init: %s",
                                     status.error_message().c_str());
                return nullptr;
            }
            return jinterc_delegate;
        },
        // .free
        [](TfLiteContext*, void* buffer) -> void {},
        // .prepare
        [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
            if (!node->user_data) {
                context->ReportError(context,
                                     "TfLiteJintercDelegate Prepare: delegate "
                                     "is not initialized");
                return kTfLiteError;
            }
            // TODO(akulik): tflite tensors are not allocated here either. It
            // would be good to set inputs and outputs only once here instead of
            // setting them every time in .invoke.
            return kTfLiteOk;
        },
        // .invoke
        [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
            const auto status = GetDelegate(node)->Invoke(context);
            if (!status.ok()) {
                context->ReportError(context,
                                     "TfLiteJintercDelegate Invoke: %s",
                                     status.error_message().c_str());
                return kTfLiteError;
            }
            return kTfLiteOk;
        },
        nullptr,                  // .profiling_string
        0,                        // .builtin_code
        "TfLiteJintercDelegate",  // .custom_name
        1,                        // .version
    };
    TfLiteIntArray* ops_to_replace = GetOpsToReplace(context);
    const auto status = context->ReplaceNodeSubsetsWithDelegateKernels(
        context, kRegistration, ops_to_replace, delegate);
    TfLiteIntArrayFree(ops_to_replace);
    return status;
}

}  // namespace jinterc
}  // namespace tflite

TfLiteJintercDelegateOptions TfLiteJintercDelegateOptionsDefault() {
    TfLiteJintercDelegateOptions options;
    // set it to -1 to detect whether it was later adjusted.
    options.is_precision_loss_allowed = -1;
    options.inference_preference =
        TFLITE_JINTERC_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
    options.inference_priority1 =
        TFLITE_JINTERC_INFERENCE_PRIORITY_MAX_PRECISION;
    options.inference_priority2 = TFLITE_JINTERC_INFERENCE_PRIORITY_MIN_LATENCY;
    options.inference_priority3 =
        TFLITE_JINTERC_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
    return options;
}

TfLiteDelegate* TfLiteJintercCreate(
    const TfLiteJintercDelegateOptions* options) {
    auto* jinterc_delegate = new tflite::jinterc::Delegate(options);
    TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                         "Created TensorFlow Lite delegate for JINTERC.");
    return jinterc_delegate ? jinterc_delegate->tflite_delegate() : nullptr;
}

void TfLiteJintercDelegateDelete(TfLiteDelegate* delegate) {
    delete tflite::jinterc::GetDelegate(delegate);
}
