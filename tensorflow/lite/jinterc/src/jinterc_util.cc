#include <jinterc_util.h>

void copyTensorDims(TfLiteTensor *tflTensor,
                    tabeq::runtime::Tensor *tbqTensor) {
    if (tflTensor == nullptr || tbqTensor == nullptr) return;
    // -- WIP
    throw new JintercException("copyTensorDims not yet implemented");
}
void copyTensorPointer(TfLiteTensor *tflTensor,
                       tabeq::runtime::Tensor *tbqTensor) {
    if (tflTensor == nullptr || tbqTensor == nullptr) return;
    // -- WIP
    throw new JintercException("copyTensorDims not yet implemented");
}