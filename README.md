# BFP on FPPGA

Block Floating Point (BFP) arithmetic operations accelerated using FPGA.

## Quick Start

```bash
# 1. Setup environment
source /tools/Xilinx/Vitis/2024.2/settings64.sh
source /opt/xilinx/xrt/setup.sh

# 2. Build and run
make                  # Build everything (2-4 hours)
make flash            # Program FPGA
make test             # Run tests
```

## Configuration

- **Platform**: Alveo U55C (xcu55c-fsvh2892-2L-e)
- **Precision**: WE=5 bits, WM=7 bits
- **Block Size**: 16 elements
- **Operations**: ENCODE, DECODE, ADD, SUB, MUL, DIV, RCP

## Project Structure

```
bfp_alveo/
├── HW/                 # Vitis HLS kernels
│   ├── src/            # BFP implementation
│   └── Makefile        # Build kernel
├── SW/                 # XRT host application
│   ├── src/            # Host code
│   └── Makefile        # Build host
├── Makefile            # Main build
└── GUIDE.md            # Complete documentation
```

## Test Flow

```
Input (FP32) → ENCODE → BFP OPS (ADD/MUL/DIV) → DECODE → Output (FP32)
                ↓                                    ↓
         [16 x (5+7+1) bits]              [Verified vs FP32]
```

## Build Targets

```bash
make           # Build hardware + software
make hw        # Build kernel (slow)
make hw-emu    # Build emulation (fast)
make sw        # Build host app
make flash     # Program FPGA
make test      # Run test suite
make clean     # Clean all
```

## Requirements

- Vitis 2024.2
- XRT (Xilinx Runtime)
- Alveo U55C with XDMA shell
- C++17 compiler

## Documentation

See **[GUIDE.md](GUIDE.md)** for:
- Detailed build instructions
- PYNQ integration
- Troubleshooting
- Performance tuning

## Support

- **Lab**: ECASLab - Instituto Tecnológico de Costa Rica
- **Hardware**: Alveo U55C @ ETH Zurich HACC
- **GitHub**: https://github.com/ECASLab/

## License

[Your license]

---

**ECASLab** | *Efficient Computing Across the Stack*
