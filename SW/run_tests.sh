#!/bin/bash
# Test script for BFP Accelerator
# Executes all operations with standard test parameters

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
EXECUTABLE="./bfp_host"
N_BLOCKS=2

echo "========================================"
echo "BFP Accelerator Test Suite"
echo "========================================"
echo ""

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo -e "${RED}Error: $EXECUTABLE not found${NC}"
    echo "Please run 'make' first to build the host application"
    exit 1
fi

# Check if xclbin exists
XCLBIN_PATH="../HW/package.hw/kernels.xclbin"
if [ ! -f "$XCLBIN_PATH" ]; then
    echo -e "${YELLOW}Warning: xclbin not found at $XCLBIN_PATH${NC}"
    echo "Make sure the HW build is complete"
fi

# Test operations
echo -e "${YELLOW}Starting tests with $N_BLOCKS blocks...${NC}"
echo ""

# Test 1: ENCODE
echo "----------------------------------------"
echo "Test 1: ENCODE Operation"
echo "----------------------------------------"
$EXECUTABLE 0 $N_BLOCKS
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ ENCODE test completed${NC}"
else
    echo -e "${RED}✗ ENCODE test failed${NC}"
fi
echo ""

# Test 2: DECODE
echo "----------------------------------------"
echo "Test 2: DECODE Operation"
echo "----------------------------------------"
$EXECUTABLE 1 $N_BLOCKS
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ DECODE test completed${NC}"
else
    echo -e "${RED}✗ DECODE test failed${NC}"
fi
echo ""

# Test 3: ADD
echo "----------------------------------------"
echo "Test 3: ADD Operation"
echo "----------------------------------------"
$EXECUTABLE 2 $N_BLOCKS
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ ADD test completed${NC}"
else
    echo -e "${RED}✗ ADD test failed${NC}"
fi
echo ""

# Test 4: SUB
echo "----------------------------------------"
echo "Test 4: SUB Operation"
echo "----------------------------------------"
$EXECUTABLE 3 $N_BLOCKS
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ SUB test completed${NC}"
else
    echo -e "${RED}✗ SUB test failed${NC}"
fi
echo ""

# Test 5: MUL
echo "----------------------------------------"
echo "Test 5: MUL Operation"
echo "----------------------------------------"
$EXECUTABLE 4 $N_BLOCKS
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ MUL test completed${NC}"
else
    echo -e "${RED}✗ MUL test failed${NC}"
fi
echo ""

# Test 6: DIV
echo "----------------------------------------"
echo "Test 6: DIV Operation"
echo "----------------------------------------"
$EXECUTABLE 5 $N_BLOCKS
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ DIV test completed${NC}"
else
    echo -e "${RED}✗ DIV test failed${NC}"
fi
echo ""

# Test 7: RCP
echo "----------------------------------------"
echo "Test 7: RCP (Reciprocal) Operation"
echo "----------------------------------------"
$EXECUTABLE 6 $N_BLOCKS
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ RCP test completed${NC}"
else
    echo -e "${RED}✗ RCP test failed${NC}"
fi
echo ""

echo "========================================"
echo "All tests completed"
echo "========================================"
