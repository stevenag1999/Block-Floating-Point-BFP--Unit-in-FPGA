/**
 * Common definitions for BFP HW/SW interface
 * Shared between host application and FPGA kernel
 */

#ifndef COMMON_BFP_H
#define COMMON_BFP_H

#include <cstdint>

// BFP Configuration - Must match HW kernel
#define WE 5
#define WM 7
#define N  16

// Operation codes - Must match bfp_kernel.cpp enum
typedef enum : unsigned int {
    OP_ENCODE = 0,
    OP_DECODE = 1,
    OP_ADD    = 2,
    OP_SUB    = 3,
    OP_MUL    = 4,
    OP_DIV    = 5,
    OP_RCP    = 6
} bfp_op_t;

// Operation names for display
static const char* OP_NAMES[] = {
    "ENCODE",
    "DECODE",
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "RCP"
};

#endif // COMMON_BFP_H
