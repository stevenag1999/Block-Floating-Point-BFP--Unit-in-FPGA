#include <iostream>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

// Profiler
#include "timer.hpp"

// BFP common definitions (UPDATED with compact format + delta)
#include "common_bfp.h"

// Helper to compute metrics (MAE and MAPE)
void compute_metrics(const float* ref, const float* got, unsigned int len,
                     double& mae, double& mape) {
    double abs_sum = 0.0, ape_sum = 0.0;
    unsigned int mape_cnt = 0;
    for (unsigned int i = 0; i < len; ++i) {
        const double r = ref[i], g = got[i];
        const double ae = std::fabs(g - r);
        abs_sum += ae;
        if (std::fabs(r) > 1e-12) { 
            ape_sum += ae / std::fabs(r); 
            ++mape_cnt; 
        }
    }
    mae = abs_sum / double(len);
    mape = (mape_cnt ? (ape_sum / double(mape_cnt)) * 100.0 : 0.0);
}

int main(int argc, char** argv) {
    INIT_PROFILER(bfp_profiler)
    int device_index = 0;

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <operation> <n_blocks>" << std::endl;
        std::cerr << "  operation: 0=ENCODE, 1=DECODE, 2=ADD, 3=SUB, 4=MUL, 5=DIV, 6=RCP" << std::endl;
        std::cerr << "  n_blocks: number of blocks (e.g., 2)" << std::endl;
        return EXIT_FAILURE;
    }

    // Get input parameters
    static std::string binaryFile = "../HW/package.hw/kernels.xclbin";
    unsigned int operation = std::stoi(argv[1]);
    unsigned int n_blocks = std::stoi(argv[2]);

    if (operation > 6) {
        std::cerr << "Error: Invalid operation code. Must be 0-6" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "BFP Accelerator Test (COMPACT FORMAT)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Operation: " << OP_NAMES[operation] << " (" << operation << ")" << std::endl;
    std::cout << "Number of blocks: " << n_blocks << std::endl;
    std::cout << "Block size (N): " << N << std::endl;
    std::cout << "BFP Config: WE=" << WE << ", WM=" << WM << std::endl;
    std::cout << "BFP_BLOCK_SIZE: " << BFP_BLOCK_SIZE << " uints per block" << std::endl;
    std::cout << std::endl;

    // Compute sizes
    unsigned int size_fp32 = n_blocks * N;  // Total FP32 elements
    unsigned int size_bfp = n_blocks * BFP_BLOCK_SIZE;  // Total BFP uints

    GET_PROFILE_INSTANCE(setup_time, bfp_profiler);
    setup_time->reset();

    std::cout << "Opening device " << device_index << "..." << std::endl;
    auto device = xrt::device(device_index);
    
    std::cout << "Loading xclbin: " << binaryFile << "..." << std::endl;
    auto uuid = device.load_xclbin(binaryFile);
    
    std::cout << "Creating kernel handle..." << std::endl;
    auto bfp_kernel = xrt::kernel(device, uuid, "bfp_kernel");
    
    setup_time->tick();

    std::cout << "Allocating buffers in global memory..." << std::endl;
    
    // KERNEL SIGNATURE (7 arguments):
    // Arguments are indexed in declaration order (following softmax pattern)
    //   0: operation (scalar - s_axilite auto-grouped)
    //   1: n_blocks (scalar - s_axilite auto-grouped)
    //   2: in_fp32     -> m_axi bundle gmem0
    //   3: in_bfp_a    -> m_axi bundle gmem1 (COMPACT: size = n_blocks * BFP_BLOCK_SIZE)
    //   4: in_bfp_b    -> m_axi bundle gmem1 (COMPACT: size = n_blocks * BFP_BLOCK_SIZE)
    //   5: out_fp32    -> m_axi bundle gmem0
    //   6: out_bfp     -> m_axi bundle gmem2 (COMPACT: size = n_blocks * BFP_BLOCK_SIZE)
    
    auto bo_in_fp32  = xrt::bo(device, size_fp32 * sizeof(float), 
                               bfp_kernel.group_id(2));
    
    auto bo_in_bfp_a = xrt::bo(device, size_bfp * sizeof(uint32_t), 
                               bfp_kernel.group_id(3));
    
    auto bo_in_bfp_b = xrt::bo(device, size_bfp * sizeof(uint32_t), 
                               bfp_kernel.group_id(4));
    
    auto bo_out_fp32 = xrt::bo(device, size_fp32 * sizeof(float), 
                               bfp_kernel.group_id(5));
    
    auto bo_out_bfp  = xrt::bo(device, size_bfp * sizeof(uint32_t), 
                               bfp_kernel.group_id(6));

    // Map buffers
    auto bo_in_fp32_map  = bo_in_fp32.map<float*>();
    auto bo_in_bfp_a_map = bo_in_bfp_a.map<uint32_t*>();
    auto bo_in_bfp_b_map = bo_in_bfp_b.map<uint32_t*>();
    auto bo_out_fp32_map = bo_out_fp32.map<float*>();
    auto bo_out_bfp_map  = bo_out_bfp.map<uint32_t*>();

    // Test data - Two different block patterns
    std::cout << "Preparing test data..." << std::endl;
    
    float A0[N] = {
        12.35f,  6.50f, 10.20f,  6.60f,  8.80f,  2.56f, 11.11f,  8.00f,
         5.45f,  9.99f,  0.15f, 18.00f,  3.80f, 90.10f, 14.00f, 10.00f
    };
    float A1[N] = {
         1.0f, 2.0f, 3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
         9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f
    };
    float B0[N] = {
        2.0f, 1.0f, 2.0f, 3.0f, 2.0f, 2.0f, 2.0f, 2.0f,
        3.0f, 3.0f, 5.0f, 3.0f, 6.0f, 3.0f, 8.0f, 2.0f
    };
    float B1[N] = {
        15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f,
         7.0f,  6.0f,  5.0f,  4.0f,  3.0f,  2.0f, 1.0f, 0.5f
    };

    // Initialize all buffers to zero
    std::fill(bo_in_fp32_map, bo_in_fp32_map + size_fp32, 0.0f);
    std::fill(bo_in_bfp_a_map, bo_in_bfp_a_map + size_bfp, 0u);
    std::fill(bo_in_bfp_b_map, bo_in_bfp_b_map + size_bfp, 0u);
    std::fill(bo_out_fp32_map, bo_out_fp32_map + size_fp32, 0.0f);
    std::fill(bo_out_bfp_map, bo_out_bfp_map + size_bfp, 0u);

    // Prepare test data based on operation
    std::vector<float> A_fp(size_fp32), B_fp(size_fp32);
    std::vector<float> golden_ref(size_fp32);
    
    // Fill data for all blocks
    for (unsigned int blk = 0; blk < n_blocks; ++blk) {
        unsigned int offset = blk * N;
        if (blk % 2 == 0) {
            std::memcpy(&A_fp[offset], A0, sizeof(float) * N);
            std::memcpy(&B_fp[offset], B0, sizeof(float) * N);
        } else {
            std::memcpy(&A_fp[offset], A1, sizeof(float) * N);
            std::memcpy(&B_fp[offset], B1, sizeof(float) * N);
        }
    }

    // Compute golden reference
    for (unsigned int i = 0; i < size_fp32; ++i) {
        switch (operation) {
            case OP_ENCODE:
            case OP_DECODE:
                golden_ref[i] = A_fp[i];
                break;
            case OP_ADD:
                golden_ref[i] = A_fp[i] + B_fp[i];
                break;
            case OP_SUB:
                golden_ref[i] = A_fp[i] - B_fp[i];
                break;
            case OP_MUL:
                golden_ref[i] = A_fp[i] * B_fp[i];
                break;
            case OP_DIV:
                if (std::fabs(B_fp[i]) > 1e-30f) {
                    golden_ref[i] = A_fp[i] / B_fp[i];
                } else {
                    // Division by zero -> Inf
                    union {float f; uint32_t u;} inf_val;
                    inf_val.u = (A_fp[i] < 0) ? 0xFF800000 : 0x7F800000;
                    golden_ref[i] = inf_val.f;
                }
                break;
            case OP_RCP:
                if (std::fabs(B_fp[i]) > 1e-30f) {
                    golden_ref[i] = 1.0f / B_fp[i];
                } else {
                    // Reciprocal of zero -> Inf
                    union {float f; uint32_t u;} inf_val;
                    inf_val.u = (B_fp[i] < 0) ? 0xFF800000 : 0x7F800000;
                    golden_ref[i] = inf_val.f;
                }
                break;
        }
    }

    // Fill input buffers based on operation
    if (operation == OP_ENCODE) {
        // ENCODE: input is FP32
        std::memcpy(bo_in_fp32_map, A_fp.data(), sizeof(float) * size_fp32);
        
    } else if (operation == OP_DECODE) {
        // DECODE: input is BFP - encode A on CPU and pack
        for (unsigned int blk = 0; blk < n_blocks; ++blk) {
            unsigned int fp_offset = blk * N;
            unsigned int bfp_offset = blk * BFP_BLOCK_SIZE;
            
            SimpleBFP bfp_a = encode_fp32_to_bfp_cpu(&A_fp[fp_offset], N);
            pack_bfp_block_cpu(bfp_a, bo_in_bfp_a_map, bfp_offset);
        }
        
    } else if (operation == OP_RCP) {
        // RCP: input is BFP B only
        for (unsigned int blk = 0; blk < n_blocks; ++blk) {
            unsigned int fp_offset = blk * N;
            unsigned int bfp_offset = blk * BFP_BLOCK_SIZE;
            
            SimpleBFP bfp_b = encode_fp32_to_bfp_cpu(&B_fp[fp_offset], N);
            pack_bfp_block_cpu(bfp_b, bo_in_bfp_b_map, bfp_offset);
        }
        
    } else {
        // Binary ops: encode both A and B on CPU and pack
        for (unsigned int blk = 0; blk < n_blocks; ++blk) {
            unsigned int fp_offset = blk * N;
            unsigned int bfp_offset = blk * BFP_BLOCK_SIZE;
            
            SimpleBFP bfp_a = encode_fp32_to_bfp_cpu(&A_fp[fp_offset], N);
            SimpleBFP bfp_b = encode_fp32_to_bfp_cpu(&B_fp[fp_offset], N);
            
            pack_bfp_block_cpu(bfp_a, bo_in_bfp_a_map, bfp_offset);
            pack_bfp_block_cpu(bfp_b, bo_in_bfp_b_map, bfp_offset);
        }
    }

    std::cout << "Syncing input buffers to device..." << std::endl;
    
    START_PROFILE(kernel_execution, bfp_profiler, 10)
    
    bo_in_fp32.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_in_bfp_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_in_bfp_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::cout << "Executing kernel: " << OP_NAMES[operation] << "..." << std::endl;
    
    // KERNEL CALL (7 arguments in declaration order)
    auto run = bfp_kernel(
        operation,      // 0: operation code
        n_blocks,       // 1: number of blocks
        bo_in_fp32,     // 2: FP32 input (for ENCODE)
        bo_in_bfp_a,    // 3: BFP input A (compact)
        bo_in_bfp_b,    // 4: BFP input B (compact)
        bo_out_fp32,    // 5: FP32 output (for DECODE)
        bo_out_bfp      // 6: BFP output (compact)
    );
    
    run.wait();
    std::cout << "Kernel completed!" << std::endl;

    std::cout << "Reading output buffers from device..." << std::endl;
    bo_out_fp32.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_out_bfp.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    
    END_PROFILE(kernel_execution);

    // Display results
    std::cout << "\n========================================" << std::endl;
    std::cout << "Results" << std::endl;
    std::cout << "========================================" << std::endl;
    
    if (operation == OP_DECODE) {
        // DECODE: output is FP32
        std::cout << "\nFirst block (8 elements) - FP32 output:" << std::endl;
        for (int i = 0; i < 8; ++i) {
            std::cout << "  [" << i << "] Result: " << bo_out_fp32_map[i] 
                      << " (expected: " << golden_ref[i] << ")" << std::endl;
        }
        
    } else {
        // ENCODE or arithmetic ops: output is BFP (compact)
        // Unpack first block to display
        SimpleBFP result_blk0;
        unpack_bfp_block_cpu(bo_out_bfp_map, 0, result_blk0);
        
        std::cout << "\nFirst block - BFP output (unpacked):" << std::endl;
        std::cout << "  exp_shared: " << result_blk0.exp_shared << std::endl;
        
        for (int i = 0; i < 8; ++i) {
            std::cout << "  [" << i << "] sign: " << result_blk0.sign[i]
                      << ", mant: " << result_blk0.mant[i]
                      << ", delta: " << result_blk0.delta[i];
            
            // Decode to FP32 for comparison
            float decoded = decode_bfp_element_to_fp32(result_blk0, i);
            std::cout << " -> FP32: " << decoded;
            
            if (operation == OP_ENCODE) {
                std::cout << " (input: " << A_fp[i] << ")";
            }
            std::cout << std::endl;
        }
    }

    // Validate results
    std::cout << "\n========================================" << std::endl;
    std::cout << "Validation" << std::endl;
    std::cout << "========================================" << std::endl;
    
    if (operation == OP_DECODE) {
        // Direct comparison with FP32 output
        double mae = 0.0, mape = 0.0;
        compute_metrics(golden_ref.data(), bo_out_fp32_map, size_fp32, mae, mape);
        
        std::cout << "MAE:  " << mae << std::endl;
        std::cout << "MAPE: " << mape << "%" << std::endl;
        
        bool passed = (mae < 0.1 && mape < 5.0);
        std::cout << "\n" << (passed ? "✓ TEST PASSED" : "✗ TEST FAILED") << std::endl;
        
    } else if (operation == OP_ENCODE) {
        // For ENCODE, verify that output has non-zero data
        bool has_data = false;
        for (unsigned int i = 0; i < size_bfp; ++i) {
            if (bo_out_bfp_map[i] != 0) {
                has_data = true;
                break;
            }
        }
        std::cout << (has_data ? "✓ TEST PASSED (data encoded)" : "✗ TEST FAILED (no data)") << std::endl;
        
    } else {
        // For arithmetic operations, decode the result and compare
        std::vector<float> decoded_result(size_fp32);
        
        for (unsigned int blk = 0; blk < n_blocks; ++blk) {
            unsigned int fp_offset = blk * N;
            unsigned int bfp_offset = blk * BFP_BLOCK_SIZE;
            
            SimpleBFP result_blk;
            unpack_bfp_block_cpu(bo_out_bfp_map, bfp_offset, result_blk);
            
            for (int i = 0; i < N; ++i) {
                decoded_result[fp_offset + i] = decode_bfp_element_to_fp32(result_blk, i);
            }
        }
        
        double mae = 0.0, mape = 0.0;
        compute_metrics(golden_ref.data(), decoded_result.data(), size_fp32, mae, mape);
        
        std::cout << "MAE:  " << mae << std::endl;
        std::cout << "MAPE: " << mape << "%" << std::endl;
        
        bool passed = (mae < 1.0 && mape < 10.0);
        std::cout << "\n" << (passed ? "✓ TEST PASSED" : "✗ TEST FAILED") << std::endl;
    }

    std::cout << "\n" << bfp_profiler << std::endl;
    std::cout << "\n========================================" << std::endl;

    return 0;
}
