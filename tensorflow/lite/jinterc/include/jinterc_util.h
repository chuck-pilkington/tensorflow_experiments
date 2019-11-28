#ifndef _jinterc_util_h_
#define _jinterc_util_h_
#endif

#include <jinterc.h>

namespace tflite {
namespace jinterc {

void copyTensorDims(TfLiteTensor *tflTensor, tabeq::runtime::Tensor *tbqTensor);
void copyTensorPointer(TfLiteTensor *tflTensor,
                       tabeq::runtime::Tensor *tbqTensor);

}  // namespace jinterc
}  // namespace tflite