

#ifndef JINTERC_DELEGATE_H_
#define JINTERC_DELEGATE_H_

#include <stdint.h>
#include <vector>

#include "jinterc/jinterc.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/context_util.h"

using TfLiteIntArrayView = tflite::TfLiteIntArrayView;

// This is where the execution of the operations or whole graph happens.
// The class below has an empty implementation just as a guideline
// on the structure.
class JintercDelegate {
   public:
    // Returns true if my delegate can handle this type of op.
    static bool SupportedOp(const TfLiteRegistration* registration) {
        switch (registration->builtin_code) {
            case kTfLiteBuiltinFullyConnected:
                return true;
            default:
                return false;
        }
    }

    // Any initialization code needed
    TfLiteStatus Init(TfLiteContext* context,
                      const TfLiteDelegateParams* delegate_params) {

        printf("Warning: Unsupported 'Init' method called\n");

        // throw JintercException("Unsupported 'Init' method called");

        return kTfLiteOk;
    }

    // Any preparation work needed (e.g. allocate buffers)
    TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
        // throw JintercException("Unsupported 'Prepare' method called");
        printf("Warning: Unsupported 'Prepare' method called\n");
        return kTfLiteOk;
    }

    // Actual running of the delegate subgraph.
    TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node) {
        // throw JintercException("Unsupported 'Invoke' method called");
        printf("Warning: Unsupported 'Invoke' method called\n");
        return kTfLiteOk;
    }

    // ... Add any other methods needed.

    // -- testing destructor...
    //
    ~JintercDelegate() { printf("Destructing Jinterc delegate...\n"); }
};

// the subgraph in the main TfLite graph.
TfLiteRegistration GetJintercDelegateNodeRegistration() {
    // This is the registration for the Delegate Node that gets
    // added to the TFLite graph instead of the subGraph it
    // replaces. It is treated as a an OP node. But in our case Init
    // will initialize the delegate Invoke will run the delegate
    // graph. Prepare for preparing the delegate. Free for any
    // cleaning needed by the delegate.
    TfLiteRegistration kernel_registration;
    kernel_registration.builtin_code = kTfLiteBuiltinDelegate;
    kernel_registration.custom_name = "JintercDelegate";
    kernel_registration.free = [](TfLiteContext* context,
                                  void* buffer) -> void {
        delete reinterpret_cast<JintercDelegate*>(buffer);
    };
    kernel_registration.init = [](TfLiteContext* context, const char* buffer,
                                  size_t) -> void* {
        // In the node init phase, initialize JintercDelegate
        // instance
        const TfLiteDelegateParams* delegate_params =
            reinterpret_cast<const TfLiteDelegateParams*>(buffer);
        JintercDelegate* my_delegate = new JintercDelegate;
        if (!my_delegate->Init(context, delegate_params)) {
            return nullptr;
        }
        return my_delegate;
    };
    kernel_registration.invoke = [](TfLiteContext* context,
                                    TfLiteNode* node) -> TfLiteStatus {
        JintercDelegate* kernel =
            reinterpret_cast<JintercDelegate*>(node->user_data);
        return kernel->Invoke(context, node);
    };
    kernel_registration.prepare = [](TfLiteContext* context,
                                     TfLiteNode* node) -> TfLiteStatus {
        JintercDelegate* kernel =
            reinterpret_cast<JintercDelegate*>(node->user_data);
        return kernel->Prepare(context, node);
    };

    return kernel_registration;
}

// TfLiteDelegate methods

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
    // Claim all nodes that can be evaluated by the delegate and ask
    // the framework to update the graph with delegate kernel
    // instead. Reserve 1 element, since we need first element to be
    // size.
    std::vector<int> supported_nodes(1);
    TfLiteIntArray* plan;
    TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &plan));
    TfLiteNode* node;
    TfLiteRegistration* registration;
    for (int node_index : TfLiteIntArrayView(plan)) {
        TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
            context, node_index, &node, &registration));
        if (JintercDelegate::SupportedOp(registration)) {
            supported_nodes.push_back(node_index);
        }
    }
    // Set first element to the number of nodes to replace.
    supported_nodes[0] = supported_nodes.size() - 1;
    TfLiteRegistration my_delegate_kernel_registration =
        GetJintercDelegateNodeRegistration();

    // This call split the graphs into subgraphs, for subgraphs that
    // can be handled by the delegate, it will replace it with a
    // 'my_delegate_kernel_registration'
    return context->ReplaceNodeSubsetsWithDelegateKernels(
        context, my_delegate_kernel_registration,
        reinterpret_cast<TfLiteIntArray*>(supported_nodes.data()), delegate);
}

void FreeBufferHandle(TfLiteContext* context, TfLiteDelegate* delegate,
                      TfLiteBufferHandle* handle) {
    // Do any cleanups.
}

TfLiteStatus CopyToBufferHandle(TfLiteContext* context,
                                TfLiteDelegate* delegate,
                                TfLiteBufferHandle buffer_handle,
                                TfLiteTensor* tensor) {
    // Copies data from tensor to delegate buffer if needed.
    return kTfLiteOk;
}

TfLiteStatus CopyFromBufferHandle(TfLiteContext* context,
                                  TfLiteDelegate* delegate,
                                  TfLiteBufferHandle buffer_handle,
                                  TfLiteTensor* tensor) {
    // Copies the data from delegate buffer into the tensor raw
    // memory.
    return kTfLiteOk;
}

// Caller takes ownership of the returned pointer.
TfLiteDelegate* CreateJintercDelegate() {
    TfLiteDelegate* delegate = new TfLiteDelegate;

    delegate->data_ = nullptr;
    delegate->flags = kTfLiteDelegateFlagsNone;
    delegate->Prepare = &DelegatePrepare;
    // This cannot be null.
    delegate->CopyFromBufferHandle = &CopyFromBufferHandle;
    // This can be null.
    delegate->CopyToBufferHandle = &CopyToBufferHandle;
    // This can be null.
    delegate->FreeBufferHandle = &FreeBufferHandle;

    return delegate;
}

#if 0
// To add the delegate you need to call

auto* my_delegate = CreateJintercDelegate();
if (interpreter->ModifyGraphWithDelegate(my_delegate) != kTfLiteOk) {
    // Handle error
} else {
    interpreter->Invoke();
}
...
    // Don't forget to delete your delegate
    delete my_delegate;
#endif

#endif
