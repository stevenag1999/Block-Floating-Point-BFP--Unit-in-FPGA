#include <ap_int.h>
#include "fp32_ops_hls.h"

// Configuration
#define N 16

using blk_t = float[N];

// Operation codes
typedef enum : unsigned int {
    OP_ADD = 2,
    OP_SUB = 3, 
    OP_MUL = 4,
    OP_DIV = 5,
    OP_RCP = 6
} fp32_op_t;

//=============================================================================
// MAIN FP32 KERNEL
//=============================================================================
extern "C" {

void fp32_kernel(
    // Control
    const unsigned int operation,
    const unsigned int n_blocks,
    // Input FP32 arrays
    const float* in_fp32_a,     // Input array A
    const float* in_fp32_b,     // Input array B  
    // Output FP32
    float* out_fp32
) {
    // Interface pragmas
    #pragma HLS INTERFACE m_axi port=in_fp32_a offset=slave bundle=gmem0 \
        max_read_burst_length=16 num_read_outstanding=4
    
    #pragma HLS INTERFACE m_axi port=in_fp32_b offset=slave bundle=gmem1 \
        max_read_burst_length=16 num_read_outstanding=4
        
    #pragma HLS INTERFACE m_axi port=out_fp32 offset=slave bundle=gmem2 \
        max_write_burst_length=16 num_write_outstanding=4

    #pragma HLS INTERFACE s_axilite port=operation
    #pragma HLS INTERFACE s_axilite port=n_blocks  
    #pragma HLS INTERFACE s_axilite port=return

    // Main processing loop
    process_blocks: for (unsigned int blk_idx = 0; blk_idx < n_blocks; blk_idx++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=128 avg=32
       
        const unsigned int offset = blk_idx * N;
        blk_t A, B, Z;
        
        //=====================================================================
        // PHASE 1: LOAD DATA
        //=====================================================================
    LOAD_A:
        for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
            A[i] = in_fp32_a[offset + i];
        }
        
        // For RCP, only need B, for others need both A and B
        if (operation != OP_RCP) {
    LOAD_B:
            for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
                B[i] = in_fp32_b[offset + i];
            }
        } else {
    LOAD_B_RCP:
            for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
                B[i] = in_fp32_b[offset + i];
            }
        }
        
        //=====================================================================
        // PHASE 2: COMPUTE
        //=====================================================================
        switch (operation) {
            case OP_ADD:
                fp32_add_blocks<N>(A, B, Z);
                break;
                
            case OP_SUB:
                fp32_sub_blocks<N>(A, B, Z);
                break;
                
            case OP_MUL:
                fp32_mul_blocks<N>(A, B, Z);
                break;
                
            case OP_DIV:
                fp32_div_blocks<N>(A, B, Z);
                break;
                
            case OP_RCP:
                fp32_rcp_blocks<N>(B, Z);
                break;
                
            default:
                // Default to addition
                fp32_add_blocks<N>(A, B, Z);
                break;
        }
        
        //=====================================================================
        // PHASE 3: STORE RESULTS
        //=====================================================================
    STORE_RESULT:
        for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
            out_fp32[offset + i] = Z[i];
        }
    }
}

} // extern "C"
