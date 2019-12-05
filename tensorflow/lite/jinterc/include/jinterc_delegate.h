
#ifndef JINTERC_DELEGATE_H_
#define JINTERC_DELEGATE_H_

#include <stdint.h>
#include <vector>

#include "tensorflow/lite/c/c_api_internal.h"

// Caller takes ownership of the returned pointer.
TfLiteDelegate* CreateJintercDelegate();

#endif
