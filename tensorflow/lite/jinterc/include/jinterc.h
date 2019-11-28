#ifndef _jinterc_h__
#define _jinterc_h__

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <stdexcept>

#include <tabeq/tabeq.h>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"

#include <jinterc_util.h>

class JintercException : public std::exception {

    std::string msg;

   public:
    const char *what() const noexcept override { return msg.c_str(); }

    JintercException(std::string m) { msg = m; }
};


/**
 * Initial build flow testing
 */
extern void jintercTest();

#endif