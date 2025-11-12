[GUIDE.md](https://github.com/user-attachments/files/23502457/GUIDE.md)
# 🚀 BFP on Alveo U55C - Complete Guide
## Block Floating Point Operations on FPGA

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Hardware Configuration](#hardware-configuration)
3. [Prerequisites](#prerequisites)
4. [Quick Start](#quick-start)
5. [Project Structure](#project-structure)
6. [Detailed Build Process](#detailed-build-process)
7. [Running Tests](#running-tests)
8. [PYNQ Integration](#pynq-integration)
9. [Troubleshooting](#troubleshooting)

---

## Overview

This project implements Block Floating Point (BFP) arithmetic operations on Xilinx Alveo U55C FPGA using:
- **Hardware**: Vitis HLS kernels (ENCODE, DECODE, ADD, MUL, DIV, RCP, SUB)
- **Software**: XRT-based host application
- **Configuration**: WE=5, WM=7, Block Size=16

### Test Flow
```
FP32 Input → ENCODE → BFP Operations (ADD/MUL/DIV) → DECODE → FP32 Output
```

---

## Hardware Configuration

### Your Alveo U55C Specifications

```bash
Platform:    xilinx_u55c_gen3x16_xdma_3_202210_1
Part Number: xcu55c-fsvh2892-2L-e
PCIe:        Gen3 x16
Memory:      16GB HBM2
Shell:       XDMA 3.202210.1
```

### Verification Commands

```bash
# Check device presence
xbutil examine

# Validate device
xbutil validate -d 0

# View detailed info
xbutil examine -d 0 -r platform
```

---

## Prerequisites

### Required Software

1. **Vitis 2024.2** (or compatible version)
2. **XRT** (Xilinx Runtime)
3. **G++** with C++17 support
4. **Make**

### Environment Setup

**CRITICAL**: You must source these files before building:

```bash
# 1. Vitis (for building kernels)
source /tools/Xilinx/Vitis/2024.2/settings64.sh

# 2. XRT (for host application and runtime)
source /opt/xilinx/xrt/setup.sh

# 3. Verify
echo $XILINX_VITIS
echo $XILINX_XRT
```

**Add to your `.bashrc` for permanent setup:**

```bash
# Add these lines to ~/.bashrc
source /tools/Xilinx/Vitis/2024.2/settings64.sh
source /opt/xilinx/xrt/setup.sh
```

---

## Quick Start

### Method 1: Full Build (2-4 hours)

```bash
# 1. Source environment
source /tools/Xilinx/Vitis/2024.2/settings64.sh
source /opt/xilinx/xrt/setup.sh

# 2. Build everything
cd bfp_alveo
make

# 3. Program FPGA
make flash

# 4. Run test
make test
```

### Method 2: Fast Emulation (10 minutes)

```bash
# Build for hardware emulation (for development/testing)
make hw-emu
make sw

# Run emulation
cd SW
XCL_EMULATION_MODE=hw_emu ./build/bfp_host ../HW/build/bfp_kernel.xclbin
```

---

## Project Structure

```
bfp_fpga/
├── Makefile              # Main build orchestration
├── GUIDE.md             # This file
│
├── HW/                  # Hardware (Vitis HLS)
│   ├── Makefile         # Kernel build
│   ├── src/
│   │   ├── bfp_kernel.cpp      # Main Vitis kernel
│   │   ├── bfp_hls.h           # BFP core (encode/decode)
│   │   └── bfp_ops_hls.h       # BFP operations
│   ├── tcl/             # (Optional) TCL scripts
│   └── build/           # Build outputs
│       ├── bfp_kernel.xo       # Compiled kernel object
│       └── bfp_kernel.xclbin   # FPGA bitstream
│
├── SW/                  # Software (Host Application)
│   ├── Makefile         # Host build
│   ├── src/
│   │   └── bfp_host.cpp        # XRT host application
│   ├── include/         # (Empty, for future headers)
│   └── build/
│       └── bfp_host            # Executable
│
└── scripts/             # Utility scripts (future)
```

---

## Detailed Build Process

### Step 1: Build Hardware Kernel

```bash
cd HW
make

# What happens:
# 1. Compile C++ to .xo (kernel object)
#    - Takes ~5-10 minutes
#    - Output: build/bfp_kernel.xo
#
# 2. Link .xo to .xclbin (bitstream)
#    - Takes ~2-4 hours
#    - Output: build/bfp_kernel.xclbin
```

**Build Stages:**
```
C++ Source (.cpp)
    ↓ [v++ -c]
Kernel Object (.xo)
    ↓ [v++ -l]
FPGA Bitstream (.xclbin)
```

**Monitor Progress:**
```bash
# In another terminal, watch logs
tail -f HW/build/logs/v++*.log
```

### Step 2: Build Software (Host)

```bash
cd SW
make

# What happens:
# - Compiles bfp_host.cpp with XRT libraries
# - Links against libxrt_coreutil
# - Output: build/bfp_host
# - Duration: ~10 seconds
```

### Step 3: Program FPGA

```bash
# From project root
make flash

# Or manually:
xbutil program -d 0 -u HW/build/bfp_kernel.xclbin
```

**Verify Programming:**
```bash
xbutil examine -d 0 -r platform
# Should show your xclbin loaded
```

---

## Running Tests

### Basic Test (Command Line)

```bash
# From project root
make test

# Or manually
cd SW
./build/bfp_host ../HW/build/bfp_kernel.xclbin
```

### Expected Output

```
============================================================
BFP ALVEO U55C TEST - Full Pipeline
Configuration: WE=5, WM=7, N=16
============================================================

[1/7] Initializing Alveo device...
      Device name: xilinx_u55c_gen3x16_xdma_3_202210_1
      XCLBIN loaded successfully
      Kernel 'bfp_kernel' found

[2/7] Preparing test data...
      Test vectors loaded (16 elements)

[3/7] Allocating device buffers...
      Buffers allocated

[4/7] TEST 1: ENCODE block A
      Encoded exp_shared = 134
      Time: 1234 us

[4/7] TEST 1b: ENCODE block B
      Encoded exp_shared = 128

[5/7] TEST 2: ADDITION (A + B)
      Max error: 2.3456%
      Sample: 12.35 + 2.0 = 14.34 (ref: 14.35)

[6/7] TEST 3: MULTIPLICATION (A * B)
      Max error: 3.1234%
      Sample: 12.35 * 2.0 = 24.68 (ref: 24.70)

[7/7] TEST 4: DIVISION (A / B)
      [Not shown for brevity - same pattern as MUL]

============================================================
ALL TESTS COMPLETED SUCCESSFULLY!
BFP operations executed on Alveo U55C FPGA
============================================================
```

### Performance Monitoring

```bash
# Monitor FPGA utilization in real-time
xbutil top -d 0

# Check kernel execution time
xbutil examine -d 0 -r dynamic-regions
```

---

## PYNQ Integration

### Setup (On Alveo Node)

```bash
# 1. Setup environment
cd /mnt/scratch/$USER
export http_proxy=http://proxy.ethz.ch:3128
export https_proxy=http://proxy.ethz.ch:3128

# 2. Activate conda
source /mnt/scratch/anaconda3/bin/activate

# 3. Install PYNQ
pip install pynq

# 4. Launch JupyterLab
jupyter-lab --ip='0.0.0.0'
```

### Using BFP from Jupyter Notebook

Create a new notebook with:

```python
import pynq
import numpy as np

# Load XCLBIN
overlay = pynq.Overlay('/path/to/bfp_kernel.xclbin')

# Allocate buffers
input_buffer = pynq.allocate(shape=(16,), dtype=np.float32)
output_buffer = pynq.allocate(shape=(16,), dtype=np.float32)

# Fill input
input_buffer[:] = [12.35, 6.50, 10.20, 6.60, 8.80, 2.56, 11.11, 8.00,
                   5.45, 9.99, 0.15, 18.00, 3.80, 90.10, 14.00, 10.00]

# Run kernel
overlay.bfp_kernel.call(0,  # OP_ENCODE
                        1,  # n_blocks
                        input_buffer,
                        # ... other args
                        )

# Read results
print("Output:", output_buffer)
```

---

## Troubleshooting

### Problem: Environment not set

```
ERROR: XILINX_VITIS not set
```

**Solution:**
```bash
source /tools/Xilinx/Vitis/2024.2/settings64.sh
source /opt/xilinx/xrt/setup.sh
```

---

### Problem: Device not found

```
ERROR: No devices found
```

**Solutions:**

1. Check device:
```bash
lspci | grep Xilinx
# Should show Alveo card
```

2. Reset device:
```bash
xbutil reset -d 0
```

3. Reload drivers:
```bash
sudo rmmod xocl xclmgmt
sudo modprobe xocl
sudo modprobe xclmgmt
```

---

### Problem: Build fails with timing violation

```
ERROR: Timing not met
```

**Solutions:**

1. **Reduce clock frequency** in `HW/Makefile`:
```makefile
# Change from:
--kernel_frequency 250

# To:
--kernel_frequency 200
```

2. **Add pipeline directives** in `bfp_kernel.cpp`:
```cpp
#pragma HLS PIPELINE II=1 rewind
```

---

### Problem: XCLBIN won't load

```
ERROR: Failed to load xclbin
```

**Solutions:**

1. Verify shell version matches:
```bash
xbutil examine -d 0 -r platform
# Check shell version
```

2. Rebuild with correct platform:
```bash
cd HW
make clean
make PLATFORM=xilinx_u55c_gen3x16_xdma_3_202210_1
```

---

### Problem: Host crashes

```
Segmentation fault
```

**Solutions:**

1. Check buffer sizes match:
```cpp
// In bfp_host.cpp, verify:
const unsigned int total_size = N * n_blocks;  // Must match kernel
```

2. Enable XRT debug:
```bash
export XRT_INI=/path/to/xrt.ini
```

Create `xrt.ini`:
```ini
[Debug]
profile=true
timeline_trace=true
```

---

### Problem: Performance is slow

**Optimizations:**

1. **Enable burst transfers** in kernel:
```cpp
#pragma HLS INTERFACE m_axi ... max_read_burst_length=64
```

2. **Use larger block sizes**:
```cpp
#define N 64  // Instead of 16
```

3. **Profile with XRT**:
```bash
xbutil top -d 0
```

---

## Advanced Usage

### Custom Block Sizes

Edit configuration in both kernel and host:

**HW/src/bfp_kernel.cpp:**
```cpp
#define N 32  // Change from 16
```

**SW/src/bfp_host.cpp:**
```cpp
#define N 32  // Must match kernel
```

Then rebuild:
```bash
make clean
make
```

### Custom Precision (WE/WM)

**HW/src/bfp_kernel.cpp:**
```cpp
#define WE 6   // Increase exponent bits
#define WM 10  // Increase mantissa bits
```

**Note:** Higher precision = more resources

---

## Performance Benchmarks

### Expected Performance (Alveo U55C @ 250MHz)

| Operation | Latency (cycles) | Throughput (GOPS) |
|-----------|------------------|-------------------|
| ENCODE    | ~50              | ~80               |
| DECODE    | ~30              | ~130              |
| ADD       | ~40              | ~100              |
| MUL       | ~35              | ~110              |
| DIV       | ~60              | ~65               |

**Test your performance:**
```bash
cd SW
./build/bfp_host ../HW/build/bfp_kernel.xclbin | grep "Time:"
```

---

## References

### Documentation
- Vitis HLS Guide: `/opt/Xilinx/Vitis/2024.2/doc/`
- XRT Documentation: `/opt/xilinx/xrt/share/doc/`
- Alveo U55C User Guide: Xilinx website

### Useful Commands

```bash
# Device info
xbutil examine -d 0

# Validate device
xbutil validate -d 0

# Monitor utilization
xbutil top -d 0

# Reset device
xbutil reset -d 0

# View logs
dmesg | grep xocl
```

---

## Support

### ECASLab Resources
- GitHub: https://github.com/ECASLab/
- Email: [Your lab contact]

### Xilinx Resources
- Forum: https://forums.xilinx.com
- Support: https://support.xilinx.com

---

## Changelog

- **v1.0** - Initial release with ENCODE/DECODE/ADD/MUL/DIV
- Configuration: WE=5, WM=7, N=16
- Target: Alveo U55C (xcu55c-fsvh2892-2L-e)

---

## License

[Your license here]

---

**Last Updated:** November 2025
**Maintainer:** ECASLab
**Platform:** Alveo U55C @ ETH Zurich HACC
