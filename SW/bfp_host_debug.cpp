#include <iostream>
#include <cstdint>

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

int main(int argc, char** argv) {
    std::cout << "=== BFP Host Minimal Debug Version ===" << std::endl;
    std::cout << std::endl;

    try {
        // Step 1: Device
        std::cout << "[Step 1] Opening device 0..." << std::endl;
        auto device = xrt::device(0);
        std::cout << "  SUCCESS: Device opened" << std::endl;
        std::cout << std::endl;

        // Step 2: Load xclbin
        std::cout << "[Step 2] Loading xclbin..." << std::endl;
        std::string binaryFile = "../HW/package.hw/kernels.xclbin";
        std::cout << "  Path: " << binaryFile << std::endl;
        auto uuid = device.load_xclbin(binaryFile);
        std::cout << "  SUCCESS: xclbin loaded" << std::endl;
        std::cout << std::endl;

        // Step 3: Get kernel handle
        std::cout << "[Step 3] Getting kernel handle..." << std::endl;
        auto bfp_kernel = xrt::kernel(device, uuid, "bfp_kernel");
        std::cout << "  SUCCESS: Kernel handle obtained" << std::endl;
        std::cout << std::endl;

        // Step 4: Allocate buffers ONE BY ONE
        unsigned int n_blocks = 2;
        unsigned int N = 16;
        unsigned int size = n_blocks * N;

        std::cout << "[Step 4] Allocating buffers..." << std::endl;
        std::cout << "  Configuration: n_blocks=" << n_blocks << ", N=" << N << ", size=" << size << std::endl;
        std::cout << std::endl;

        // Buffer 0: in_fp32_a (gmem0)
        std::cout << "  [4.1] Allocating bo_in_fp32_a (gmem0, " << size * sizeof(float) << " bytes)..." << std::flush;
        auto bo_in_fp32_a = xrt::bo(device, size * sizeof(float), bfp_kernel.group_id(0));
        std::cout << " OK" << std::endl;

        // Buffer 1: in_exp_a (gmem1)
        std::cout << "  [4.2] Allocating bo_in_exp_a (gmem1, " << n_blocks * sizeof(unsigned int) << " bytes)..." << std::flush;
        auto bo_in_exp_a = xrt::bo(device, n_blocks * sizeof(unsigned int), bfp_kernel.group_id(1));
        std::cout << " OK" << std::endl;

        // Buffer 2: in_sign_a (gmem2)
        std::cout << "  [4.3] Allocating bo_in_sign_a (gmem2, " << size * sizeof(unsigned int) << " bytes)..." << std::flush;
        auto bo_in_sign_a = xrt::bo(device, size * sizeof(unsigned int), bfp_kernel.group_id(2));
        std::cout << " OK" << std::endl;

        // Buffer 3: in_mant_a (gmem3)
        std::cout << "  [4.4] Allocating bo_in_mant_a (gmem3, " << size * sizeof(unsigned int) << " bytes)..." << std::flush;
        auto bo_in_mant_a = xrt::bo(device, size * sizeof(unsigned int), bfp_kernel.group_id(3));
        std::cout << " OK" << std::endl;

        // Buffer 4: in_exp_b (gmem1 - shared)
        std::cout << "  [4.5] Allocating bo_in_exp_b (gmem1, " << n_blocks * sizeof(unsigned int) << " bytes)..." << std::flush;
        auto bo_in_exp_b = xrt::bo(device, n_blocks * sizeof(unsigned int), bfp_kernel.group_id(1));
        std::cout << " OK" << std::endl;

        // Buffer 5: in_sign_b (gmem2 - shared)
        std::cout << "  [4.6] Allocating bo_in_sign_b (gmem2, " << size * sizeof(unsigned int) << " bytes)..." << std::flush;
        auto bo_in_sign_b = xrt::bo(device, size * sizeof(unsigned int), bfp_kernel.group_id(2));
        std::cout << " OK" << std::endl;

        // Buffer 6: in_mant_b (gmem3 - shared)
        std::cout << "  [4.7] Allocating bo_in_mant_b (gmem3, " << size * sizeof(unsigned int) << " bytes)..." << std::flush;
        auto bo_in_mant_b = xrt::bo(device, size * sizeof(unsigned int), bfp_kernel.group_id(3));
        std::cout << " OK" << std::endl;

        // Buffer 7: out_fp32 (gmem0 - shared)
        std::cout << "  [4.8] Allocating bo_out_fp32 (gmem0, " << size * sizeof(float) << " bytes)..." << std::flush;
        auto bo_out_fp32 = xrt::bo(device, size * sizeof(float), bfp_kernel.group_id(0));
        std::cout << " OK" << std::endl;

        // Buffer 8: out_exp (gmem1 - shared)
        std::cout << "  [4.9] Allocating bo_out_exp (gmem1, " << n_blocks * sizeof(unsigned int) << " bytes)..." << std::flush;
        auto bo_out_exp = xrt::bo(device, n_blocks * sizeof(unsigned int), bfp_kernel.group_id(1));
        std::cout << " OK" << std::endl;

        // Buffer 9: out_sign (gmem2 - shared)
        std::cout << "  [4.10] Allocating bo_out_sign (gmem2, " << size * sizeof(unsigned int) << " bytes)..." << std::flush;
        auto bo_out_sign = xrt::bo(device, size * sizeof(unsigned int), bfp_kernel.group_id(2));
        std::cout << " OK" << std::endl;

        // Buffer 10: out_mant (gmem3 - shared)
        std::cout << "  [4.11] Allocating bo_out_mant (gmem3, " << size * sizeof(unsigned int) << " bytes)..." << std::flush;
        auto bo_out_mant = xrt::bo(device, size * sizeof(unsigned int), bfp_kernel.group_id(3));
        std::cout << " OK" << std::endl;

        std::cout << "  SUCCESS: All buffers allocated" << std::endl;
        std::cout << std::endl;

        // Step 5: Map buffers
        std::cout << "[Step 5] Mapping buffers to host memory..." << std::endl;
        auto bo_in_fp32_a_map = bo_in_fp32_a.map<float*>();
        std::cout << "  Mapped bo_in_fp32_a" << std::endl;
        
        auto bo_out_fp32_map = bo_out_fp32.map<float*>();
        std::cout << "  Mapped bo_out_fp32" << std::endl;
        
        std::cout << "  SUCCESS: Buffers mapped" << std::endl;
        std::cout << std::endl;

        // Step 6: Fill with test data
        std::cout << "[Step 6] Filling test data..." << std::endl;
        for (unsigned int i = 0; i < size; ++i) {
            bo_in_fp32_a_map[i] = float(i) * 0.5f;
        }
        std::cout << "  SUCCESS: Test data filled" << std::endl;
        std::cout << std::endl;

        // Step 7: Sync to device
        std::cout << "[Step 7] Syncing data to device..." << std::endl;
        bo_in_fp32_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        std::cout << "  SUCCESS: Data synced to device" << std::endl;
        std::cout << std::endl;

        // Step 8: Run kernel
        std::cout << "[Step 8] Running kernel (ENCODE operation)..." << std::endl;
        auto run = bfp_kernel(
            0,              // operation = ENCODE
            n_blocks,       // n_blocks
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
        std::cout << "  Kernel launched, waiting for completion..." << std::endl;
        run.wait();
        std::cout << "  SUCCESS: Kernel completed" << std::endl;
        std::cout << std::endl;

        // Step 9: Read results
        std::cout << "[Step 9] Reading results..." << std::endl;
        bo_out_fp32.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::cout << "  First 8 results:" << std::endl;
        for (int i = 0; i < 8; ++i) {
            std::cout << "    [" << i << "] " << bo_out_fp32_map[i] << std::endl;
        }
        std::cout << std::endl;

        std::cout << "=== ALL TESTS PASSED ===" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << std::endl;
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
