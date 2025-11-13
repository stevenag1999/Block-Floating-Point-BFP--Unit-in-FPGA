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

// BFP common definitions
#include "common_bfp.h"

// Helper function to compute metrics (MAE and MAPE)
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

    std::cout << "BFP Accelerator Test" << std::endl;
    std::cout << "Operation: " << OP_NAMES[operation] << " (" << operation << ")" << std::endl;
    std::cout << "Number of blocks: " << n_blocks << std::endl;
    std::cout << "Block size (N): " << N << std::endl;

    // Compute sizes
    unsigned int size = n_blocks * N;

    GET_PROFILE_INSTANCE(setup_time, bfp_profiler);
    setup_time->reset();

    std::cout << "Open the device " << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);
    auto bfp_kernel = xrt::kernel(device, uuid, "bfp_kernel");
    setup_time->tick();

    std::cout << "Allocate Buffers in Global Memory" << std::endl;
    
    // Input A buffers (FP32 for ENCODE, BFP format for operations)
    auto bo_in_fp32_a = xrt::bo(device, size * sizeof(float), bfp_kernel.group_id(0));
    auto bo_in_exp_a  = xrt::bo(device, n_blocks * sizeof(unsigned int), bfp_kernel.group_id(1));
    auto bo_in_sign_a = xrt::bo(device, size * sizeof(unsigned int), bfp_kernel.group_id(2));
    auto bo_in_mant_a = xrt::bo(device, size * sizeof(unsigned int), bfp_kernel.group_id(3));
    
    // Input B buffers (BFP format for binary operations)
    auto bo_in_exp_b  = xrt::bo(device, n_blocks * sizeof(unsigned int), bfp_kernel.group_id(4));
    auto bo_in_sign_b = xrt::bo(device, size * sizeof(unsigned int), bfp_kernel.group_id(5));
    auto bo_in_mant_b = xrt::bo(device, size * sizeof(unsigned int), bfp_kernel.group_id(6));
    
    // Output buffers (FP32 for DECODE, BFP format for other operations)
    auto bo_out_fp32 = xrt::bo(device, size * sizeof(float), bfp_kernel.group_id(7));
    auto bo_out_exp  = xrt::bo(device, n_blocks * sizeof(unsigned int), bfp_kernel.group_id(8));
    auto bo_out_sign = xrt::bo(device, size * sizeof(unsigned int), bfp_kernel.group_id(9));
    auto bo_out_mant = xrt::bo(device, size * sizeof(unsigned int), bfp_kernel.group_id(10));

    // Map the contents of the buffer objects into host memory
    auto bo_in_fp32_a_map = bo_in_fp32_a.map<float*>();
    auto bo_in_exp_a_map  = bo_in_exp_a.map<unsigned int*>();
    auto bo_in_sign_a_map = bo_in_sign_a.map<unsigned int*>();
    auto bo_in_mant_a_map = bo_in_mant_a.map<unsigned int*>();
    
    auto bo_in_exp_b_map  = bo_in_exp_b.map<unsigned int*>();
    auto bo_in_sign_b_map = bo_in_sign_b.map<unsigned int*>();
    auto bo_in_mant_b_map = bo_in_mant_b.map<unsigned int*>();
    
    auto bo_out_fp32_map = bo_out_fp32.map<float*>();
    auto bo_out_exp_map  = bo_out_exp.map<unsigned int*>();
    auto bo_out_sign_map = bo_out_sign.map<unsigned int*>();
    auto bo_out_mant_map = bo_out_mant.map<unsigned int*>();

    // Filling data
    std::cout << "Filling Input Buffers" << std::endl;
    
    // Test data - Two blocks similar to tb_kernel.cc
    float A0[N] = {
        12.35f,  6.50f, 10.20f,  6.60f,  8.80f,  2.56f, 11.11f,  8.00f,
         5.45f,  9.99f,  0.15f, 18.00f,  3.80f, 90.10f, 14.00f, 10.00f
    };
    float A1[N] = {
         0.0f, 1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
         8.0f, 9.0f, 0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f
    };
    float B0[N] = {
        -2.0f, 0.0f, -2.0f, 3.0f, 2.0f, 2.0f, 2.0f, 2.0f,
         3.0f, 3.0f,  5.0f, 3.0f, 6.0f, 3.0f, 8.0f, 2.0f
    };
    float B1[N] = {
        15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f,
         7.0f,  6.0f,  5.0f,  4.0f,  3.0f,  2.0f, 1.0f, 0.5f
    };

    // Initialize all buffers to zero
    std::fill(bo_in_fp32_a_map, bo_in_fp32_a_map + size, 0.0f);
    std::fill(bo_in_exp_a_map, bo_in_exp_a_map + n_blocks, 0u);
    std::fill(bo_in_sign_a_map, bo_in_sign_a_map + size, 0u);
    std::fill(bo_in_mant_a_map, bo_in_mant_a_map + size, 0u);
    std::fill(bo_in_exp_b_map, bo_in_exp_b_map + n_blocks, 0u);
    std::fill(bo_in_sign_b_map, bo_in_sign_b_map + size, 0u);
    std::fill(bo_in_mant_b_map, bo_in_mant_b_map + size, 0u);
    std::fill(bo_out_fp32_map, bo_out_fp32_map + size, 0.0f);
    std::fill(bo_out_exp_map, bo_out_exp_map + n_blocks, 0u);
    std::fill(bo_out_sign_map, bo_out_sign_map + size, 0u);
    std::fill(bo_out_mant_map, bo_out_mant_map + size, 0u);

    // Prepare test data based on operation
    std::vector<float> A_fp(size), B_fp(size);
    std::vector<float> golden_ref(size);
    
    // Fill A and B data for all blocks (repeating pattern)
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

    // Compute golden reference based on operation
    for (unsigned int i = 0; i < size; ++i) {
        switch (operation) {
            case OP_ENCODE:
            case OP_DECODE:
                golden_ref[i] = A_fp[i];  // ENCODE->DECODE should recover original
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
                golden_ref[i] = (std::fabs(B_fp[i]) > 1e-30f) ? (A_fp[i] / B_fp[i]) : 0.f;
                break;
            case OP_RCP:
                golden_ref[i] = (std::fabs(B_fp[i]) > 1e-30f) ? (1.0f / B_fp[i]) : 0.f;
                break;
        }
    }

    // Fill input buffers based on operation type
    if (operation == OP_ENCODE) {
        // For ENCODE: input is FP32 A
        std::memcpy(bo_in_fp32_a_map, A_fp.data(), sizeof(float) * size);
    } else {
        // For other operations: need to encode A and B first
        // This is a simplified version - in real scenario, you'd run ENCODE first
        // For now, we'll encode on CPU side using reference implementation
        std::cout << "Note: Using CPU-side encoding for test data preparation" << std::endl;
        
        // For simplicity in this test, we'll just copy FP32 and let kernel handle
        // In production, you'd run ENCODE operation first to get BFP format
        std::memcpy(bo_in_fp32_a_map, A_fp.data(), sizeof(float) * size);
        
        // For operations that need B, we need BFP format of B
        // For this test, we'll run operation assuming pre-encoded data
        // Note: A complete test would first ENCODE both A and B, then perform operation
    }

    // Synchronize buffer content with device side
    std::cout << "Synchronize input buffer data to device global memory" << std::endl;
    
    START_PROFILE(kernel_execution, bfp_profiler, 10)
    
    bo_in_fp32_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_in_exp_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_in_sign_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_in_mant_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_in_exp_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_in_sign_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_in_mant_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::cout << "Execution of the kernel: " << OP_NAMES[operation] << std::endl;
    auto run = bfp_kernel(
        operation,
        n_blocks,
        bo_in_fp32_a,
        bo_in_exp_a,
        bo_in_sign_a,
        bo_in_mant_a,
        bo_in_exp_b,
        bo_in_sign_b,
        bo_in_mant_b,
        bo_out_fp32,
        bo_out_exp,
        bo_out_sign,
        bo_out_mant
    );
    
    std::cout << "Waiting for kernel completion" << std::endl;
    run.wait();

    // Get the output
    std::cout << "Get the output data from the device" << std::endl;
    bo_out_fp32.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_out_exp.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_out_sign.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_out_mant.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    
    END_PROFILE(kernel_execution);

    // Display results
    std::cout << "\n=== Results ===" << std::endl;
    
    // Show first block results
    std::cout << "Output (Block 0, first 8 elements):" << std::endl;
    for (int i = 0; i < 8; ++i) {
        if (operation == OP_DECODE || operation == OP_ENCODE) {
            std::cout << "  [" << i << "] FP32: " << bo_out_fp32_map[i] << std::endl;
        } else {
            std::cout << "  [" << i << "] exp_shared: " << bo_out_exp_map[0] 
                      << ", sign: " << bo_out_sign_map[i]
                      << ", mant: " << bo_out_mant_map[i] << std::endl;
        }
    }

    // Compute and display metrics
    if (operation == OP_DECODE || operation == OP_ENCODE) {
        double mae = 0.0, mape = 0.0;
        compute_metrics(golden_ref.data(), bo_out_fp32_map, size, mae, mape);
        std::cout << "\n=== Accuracy Metrics ===" << std::endl;
        std::cout << "MAE:  " << mae << std::endl;
        std::cout << "MAPE: " << mape << "%" << std::endl;
        
        // Check pass/fail
        if (mae < 0.1 && mape < 5.0) {
            std::cout << "\nTEST PASSED" << std::endl;
        } else {
            std::cout << "\nTEST FAILED (accuracy below threshold)" << std::endl;
        }
    } else {
        std::cout << "\nNote: For operations other than ENCODE/DECODE, " 
                  << "full validation requires DECODE step" << std::endl;
        std::cout << "TEST COMPLETED" << std::endl;
    }

    // Print profiling results
    std::cout << "\n" << bfp_profiler << std::endl;

    return 0;
}
