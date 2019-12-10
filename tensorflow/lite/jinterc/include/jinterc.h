#ifndef _jinterc_h__
#define _jinterc_h__

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <stdexcept>

#include <tabeq/tabeq.h>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"

#include "jinterc_util.h"

class JintercException : public std::exception {

    std::string msg;

   public:
    const char *what() const noexcept override { return msg.c_str(); }

    JintercException(std::string m) { msg = m; }
};

namespace tflite {
namespace jinterc {

// Encapsulated compilation/runtime tradeoffs.
enum class InferenceUsage {
    UNKNOWN,

    // InferenceRunner will be used only once. Therefore, it is important to
    // minimize bootstrap time as well.
    FAST_SINGLE_ANSWER,

    // Prefer maximizing the throughput. Same inference runner will be used
    // repeatedly on different inputs.
    SUSTAINED_SPEED,
};

// Defines aspects to control while instantiating a runner.
enum class InferencePriority {
    UNKNOWN,

    MIN_LATENCY,

    MAX_PRECISION,

    MIN_MEMORY_USAGE,
};

}  // namespace jinterc
}  // namespace tflite

/**
 * Initial build flow testing
 */
extern void jintercTest();

#endif
