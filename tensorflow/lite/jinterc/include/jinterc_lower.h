#ifndef jinterc_lower_h_
#define jinterc_lower_h_

#include "jinterc.h"

namespace tflite {
namespace jinterc {

TfLiteIntArray* GetOpsToReplace(TfLiteContext* context);
}
}  // namespace tflite

#endif