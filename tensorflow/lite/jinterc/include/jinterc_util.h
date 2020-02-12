#ifndef _jinterc_util_h_
#define _jinterc_util_h_
#endif

#include <jinterc/jinterc.h>

/**
 * Debugging routines
 */
extern "C" {
extern void cepName(const char *newName); 
extern void cepFloats(void *p, int nbytes);
}

namespace tflite {
namespace jinterc {

#if 0
// -- WIP
//
void copyTensorDims(TfLiteTensor *tflTensor, tabeq::runtime::Tensor *tbqTensor);

void copyTensorPointer(TfLiteTensor *tflTensor,
                       tabeq::runtime::Tensor *tbqTensor);
#endif

}  // namespace jinterc
}  // namespace tflite
