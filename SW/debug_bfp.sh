#!/bin/bash
# Advanced debugging script for BFP host application
# Helps identify the exact cause of "Killed" error

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "========================================"
echo "BFP Host Application - Debug Analysis"
echo "========================================"
echo ""

# Check 1: System Memory
echo -e "${BLUE}[1] System Memory Status${NC}"
free -h
echo ""
AVAILABLE_MEM=$(free -m | awk 'NR==2{print $7}')
echo "Available memory: ${AVAILABLE_MEM} MB"
if [ $AVAILABLE_MEM -lt 1000 ]; then
    echo -e "${RED}WARNING: Low memory! Need at least 1GB available${NC}"
fi
echo ""

# Check 2: Device Memory
echo -e "${BLUE}[2] FPGA Device Memory${NC}"
if command -v xbutil &> /dev/null; then
    xbutil examine -d 0 2>&1 | grep -A 5 "Memory"
else
    echo -e "${YELLOW}xbutil not found, skipping device memory check${NC}"
fi
echo ""

# Check 3: Device Permissions
echo -e "${BLUE}[3] Device File Permissions${NC}"
ls -la /dev/dri/renderD* 2>/dev/null || echo -e "${RED}No render devices found${NC}"
echo ""

# Check 4: XRT Environment
echo -e "${BLUE}[4] XRT Environment${NC}"
echo "XILINX_XRT: $XILINX_XRT"
if [ -z "$XILINX_XRT" ]; then
    echo -e "${RED}ERROR: XILINX_XRT not set!${NC}"
    echo "Run: source /opt/xilinx/xrt/setup.sh"
else
    echo -e "${GREEN}XRT environment configured${NC}"
fi
echo ""

# Check 5: XRT Libraries
echo -e "${BLUE}[5] XRT Libraries${NC}"
ldd ./bfp_host 2>/dev/null | grep -E "xrt|not found" || echo -e "${YELLOW}bfp_host not compiled yet${NC}"
echo ""

# Check 6: Recent kernel messages (last 20 lines)
echo -e "${BLUE}[6] Recent Kernel Messages${NC}"
echo "Checking for OOM killer or XRT errors..."
sudo dmesg | tail -20 | grep -E "killed|oom|xrt|xocl|zocl|Out of memory" || echo "No relevant errors found"
echo ""

# Check 7: ulimit settings
echo -e "${BLUE}[7] Process Limits (ulimit)${NC}"
echo "Max memory size: $(ulimit -m)"
echo "Max virtual memory: $(ulimit -v)"
echo "Max stack size: $(ulimit -s)"
echo "Max file size: $(ulimit -f)"
echo ""

# Check 8: XCLBIN file
echo -e "${BLUE}[8] XCLBIN File Check${NC}"
XCLBIN_PATH="../HW/package.hw/kernels.xclbin"
if [ -f "$XCLBIN_PATH" ]; then
    XCLBIN_SIZE=$(ls -lh "$XCLBIN_PATH" | awk '{print $5}')
    echo -e "${GREEN}Found: $XCLBIN_PATH ($XCLBIN_SIZE)${NC}"
    
    # Check if xclbin is valid
    if command -v xclbinutil &> /dev/null; then
        echo "Validating xclbin..."
        xclbinutil --info --input "$XCLBIN_PATH" 2>&1 | head -10
    fi
else
    echo -e "${RED}ERROR: XCLBIN not found at $XCLBIN_PATH${NC}"
fi
echo ""

# Check 9: Try minimal buffer allocation test
echo -e "${BLUE}[9] Minimal XRT Test${NC}"
cat > /tmp/test_xrt_minimal.cpp << 'EOFCPP'
#include <iostream>
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_bo.h"

int main() {
    try {
        std::cout << "Opening device..." << std::endl;
        auto device = xrt::device(0);
        std::cout << "SUCCESS: Device opened" << std::endl;
        
        std::cout << "Loading xclbin..." << std::endl;
        auto uuid = device.load_xclbin("../HW/package.hw/kernels.xclbin");
        std::cout << "SUCCESS: xclbin loaded" << std::endl;
        
        std::cout << "Getting kernel handle..." << std::endl;
        auto kernel = xrt::kernel(device, uuid, "bfp_kernel");
        std::cout << "SUCCESS: Kernel handle obtained" << std::endl;
        
        std::cout << "Allocating small buffer (1KB)..." << std::endl;
        auto bo_test = xrt::bo(device, 1024, kernel.group_id(0));
        std::cout << "SUCCESS: Buffer allocated" << std::endl;
        
        std::cout << "\nAll basic XRT operations work!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
EOFCPP

echo "Compiling minimal test..."
g++ -o /tmp/test_xrt_minimal /tmp/test_xrt_minimal.cpp \
    -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib \
    -lxrt_core -lxrt_coreutil -std=c++17 2>&1

if [ -f /tmp/test_xrt_minimal ]; then
    echo "Running minimal test..."
    /tmp/test_xrt_minimal
    RESULT=$?
    if [ $RESULT -eq 0 ]; then
        echo -e "${GREEN}Minimal XRT test PASSED${NC}"
    else
        echo -e "${RED}Minimal XRT test FAILED${NC}"
    fi
    rm -f /tmp/test_xrt_minimal /tmp/test_xrt_minimal.cpp
else
    echo -e "${RED}Failed to compile minimal test${NC}"
fi
echo ""

# Summary and recommendations
echo "========================================"
echo "Summary and Recommendations"
echo "========================================"
echo ""

# Check dmesg for the actual kill reason
echo "Checking dmesg for the kill reason (requires sudo)..."
LAST_KILL=$(sudo dmesg | grep -i "killed process" | tail -1)
if [ -n "$LAST_KILL" ]; then
    echo -e "${RED}Found kill event:${NC}"
    echo "$LAST_KILL"
    
    if echo "$LAST_KILL" | grep -q "Out of memory"; then
        echo ""
        echo -e "${YELLOW}DIAGNOSIS: Out of Memory (OOM) Killer${NC}"
        echo "Solutions:"
        echo "  1. Close other applications to free memory"
        echo "  2. Reduce n_blocks (try ./bfp_host 0 1)"
        echo "  3. Check for memory leaks in kernel"
        echo "  4. Increase swap space"
    fi
fi

echo ""
echo "If the problem persists:"
echo "  1. Check: sudo dmesg | tail -50"
echo "  2. Monitor: watch -n 1 free -h"
echo "  3. Run with strace: strace -o trace.log ./bfp_host 0 2"
echo "  4. Check XRT logs in /var/log/"
echo ""
