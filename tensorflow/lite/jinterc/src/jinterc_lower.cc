#include "tensorflow/lite/jinterc/include/jinterc_lower.h"

namespace tflite {
namespace jinterc {

TfLiteIntArray* GetOpsToReplace(TfLiteContext* context) {
    throw JintercException("Jinterc 'GetOpsToReplace' not implemented");
}
}  // namespace jinterc
}  // namespace tflite