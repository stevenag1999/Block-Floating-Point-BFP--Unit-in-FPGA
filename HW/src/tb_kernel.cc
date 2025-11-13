#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <iomanip>
#include <bitset>
#include <cstring>
#include <type_traits>

#include "bfp_hls.h"
#include "bfp_ops_hls.h"

// Top kernel HLS
extern "C" void bfp_kernel(
    const unsigned int operation,
    const unsigned int n_blocks,
    const float* in_fp32_a,
    const unsigned int* in_exp_a,
    const unsigned int* in_sign_a,
    const unsigned int* in_mant_a,
    const unsigned int* in_exp_b,
    const unsigned int* in_sign_b,
    const unsigned int* in_mant_b,
    float* out_fp32,
    unsigned int* out_exp,
    unsigned int* out_sign,
    unsigned int* out_mant
);

//------------------------ Configuración ------------------------
#define WE 5
#define WM 7
#define N  16

using Cfg = BFP_bias<WE, WM>;

enum : unsigned {
    OP_ENCODE = 0,
    OP_DECODE = 1,
    OP_ADD    = 2,
    OP_SUB    = 3,
    OP_MUL    = 4,
    OP_DIV    = 5,
    OP_RCP    = 6
};

//------------------------ Helpers de error ------------------------
inline float calc_rel_error(float computed, float reference) {
    if (reference == 0.0f) return std::fabs(computed);
    return std::fabs((computed - reference) / reference) * 100.0f;
}

void print_error_stats(const char* op_name, double mean_abs, double max_err) {
    std::cout << "\n" << op_name << " Error Statistics:\n";
    std::cout << "  Mean Absolute Error: " << mean_abs << "\n";
    std::cout << "  Max Absolute Error:  " << max_err << "\n";
    std::cout << std::string(60, '=') << "\n\n";
}

//------------------------ MAIN ------------------------
int main() {
    const unsigned n_blocks = 1;
    const unsigned sz = N * n_blocks;

    std::cout << std::string(60, '=') << "\n";
    std::cout << "BFP KERNEL TESTBENCH (via bfp_kernel)\n";
    std::cout << "Configuration: WE=" << Cfg::we
              << ", WM=" << Cfg::wm
              << ", Block Size=" << N << ", n_blocks=" << n_blocks << "\n";
    std::cout << "Bias: " << Cfg::bias_bfp << "\n";
    std::cout << std::string(60, '=') << "\n\n";

    //======================== Datos de prueba ========================
    std::array<float, N> inputs = {
        12.35f, 6.50f, 10.20f, 6.60f, 8.80f, 2.56f, 11.11f, 8.00f,
         5.45f, 9.99f, 0.15f, 18.00f, 3.80f, 90.10f, 14.00f, 10.00f
    };

    std::array<float, N> inputs_b = {
        -2.00f, 0.00f, -2.00f, 3.00f, 2.00f, 2.00f, 2.00f, 2.00f,
         3.00f, 3.00f, 5.00f, 3.00f, 6.00f, 3.00f, 8.00f, 2.00f
    };

    // Buffers host para el kernel
    std::vector<float>    in_fp32_a(sz, 0.f);
    std::vector<unsigned> in_exp_a(n_blocks, 0), in_sign_a(sz, 0), in_mant_a(sz, 0);
    std::vector<unsigned> in_exp_b(n_blocks, 0), in_sign_b(sz, 0), in_mant_b(sz, 0);

    std::vector<float>    out_fp32(sz, 0.f);
    std::vector<unsigned> out_exp(n_blocks, 0), out_sign(sz, 0), out_mant(sz, 0);

    // Dummies para puertos no usados
    std::vector<float>    dummy_fp32(sz, 0.f);
    std::vector<unsigned> dummy_exp(n_blocks, 0), dummy_sign(sz, 0), dummy_mant(sz, 0);

    //======================== ENCODE A ========================
    std::cout << "Encoding Block A via kernel...\n";

    std::memcpy(in_fp32_a.data(), inputs.data(), sizeof(float)*N);

    std::fill(out_exp.begin(),  out_exp.end(),  0u);
    std::fill(out_sign.begin(), out_sign.end(), 0u);
    std::fill(out_mant.begin(), out_mant.end(), 0u);

    bfp_kernel(OP_ENCODE, n_blocks,
               in_fp32_a.data(),
               dummy_exp.data(), dummy_sign.data(), dummy_mant.data(),
               dummy_exp.data(), dummy_sign.data(), dummy_mant.data(),
               dummy_fp32.data(), out_exp.data(), out_sign.data(), out_mant.data());

    in_exp_a[0] = out_exp[0];
    for (unsigned i = 0; i < N; ++i) {
        in_sign_a[i] = out_sign[i];
        in_mant_a[i] = out_mant[i];
    }

    std::cout << "Block A: exp_shared=" << in_exp_a[0]
              << " (real=" << (int(in_exp_a[0]) - Cfg::bias_bfp) << ")\n\n";

    //======================== ENCODE B ========================
    std::cout << "Encoding Block B via kernel...\n";

    std::memcpy(in_fp32_a.data(), inputs_b.data(), sizeof(float)*N);

    std::fill(out_exp.begin(),  out_exp.end(),  0u);
    std::fill(out_sign.begin(), out_sign.end(), 0u);
    std::fill(out_mant.begin(), out_mant.end(), 0u);

    bfp_kernel(OP_ENCODE, n_blocks,
               in_fp32_a.data(),
               dummy_exp.data(), dummy_sign.data(), dummy_mant.data(),
               dummy_exp.data(), dummy_sign.data(), dummy_mant.data(),
               dummy_fp32.data(), out_exp.data(), out_sign.data(), out_mant.data());

    in_exp_b[0] = out_exp[0];
    for (unsigned i = 0; i < N; ++i) {
        in_sign_b[i] = out_sign[i];
        in_mant_b[i] = out_mant[i];
    }

    std::cout << "Block B: exp_shared=" << in_exp_b[0]
              << " (real=" << (int(in_exp_b[0]) - Cfg::bias_bfp) << ")\n\n";

    //======================== Referencias FP32 ========================
    std::array<float, N> ref_add{}, ref_sub{}, ref_mul{}, ref_div{};

    for (std::size_t i = 0; i < N; i++) {
        ref_add[i] = inputs[i] + inputs_b[i];
        ref_sub[i] = inputs[i] - inputs_b[i];
        ref_mul[i] = inputs[i] * inputs_b[i];
        ref_div[i] = (inputs_b[i] == 0.0f)
                     ? std::copysign(INFINITY, inputs[i])
                     : inputs[i] / inputs_b[i];
    }

    auto run_op_and_report = [&](unsigned op, const char* name,
                                 const std::array<float, N>& ref,
                                 bool uses_both_operands = true) {
        std::cout << std::string(60, '=') << "\n";
        std::cout << "TEST: " << name << "\n";
        std::cout << std::string(60, '=') << "\n";

        // limpiar salidas
        std::fill(out_exp.begin(),  out_exp.end(),  0u);
        std::fill(out_sign.begin(), out_sign.end(), 0u);
        std::fill(out_mant.begin(), out_mant.end(), 0u);
        std::fill(out_fp32.begin(), out_fp32.end(), 0.f);

        // Ejecutar operación en formato BFP codificado
        if (uses_both_operands) {
            bfp_kernel(op, n_blocks,
                       dummy_fp32.data(),
                       in_exp_a.data(), in_sign_a.data(), in_mant_a.data(),
                       in_exp_b.data(), in_sign_b.data(), in_mant_b.data(),
                       dummy_fp32.data(), out_exp.data(), out_sign.data(), out_mant.data());
        } else {
            // Por si quisieras probar RCP(B) como prueba aparte
            bfp_kernel(op, n_blocks,
                       dummy_fp32.data(),
                       dummy_exp.data(), dummy_sign.data(), dummy_mant.data(),
                       in_exp_b.data(), in_sign_b.data(), in_mant_b.data(),
                       dummy_fp32.data(), out_exp.data(), out_sign.data(), out_mant.data());
        }

        int E_real = int(out_exp[0]) - Cfg::bias_bfp;
        std::cout << "Result exp_shared: " << out_exp[0]
                  << " (dec), " << std::bitset<Cfg::we>(out_exp[0])
                  << " (bin), real=" << E_real << "\n\n";

        // Decodificar a FP32
        bfp_kernel(OP_DECODE, n_blocks,
                   dummy_fp32.data(),
                   out_exp.data(), out_sign.data(), out_mant.data(),
                   dummy_exp.data(), dummy_sign.data(), dummy_mant.data(),
                   out_fp32.data(), dummy_exp.data(), dummy_sign.data(), dummy_mant.data());

        // Tabla por elemento, igual estilo que bfp_hls_tb
        std::cout << std::setw(3) << "i"
                  << std::setw(12) << "A"
                  << std::setw(12) << "B"
                  << std::setw(16) << "BFP Result"
                  << std::setw(16) << "FP32 Ref"
                  << std::setw(12) << "Err (%)\n";
        std::cout << std::string(60, '-') << "\n";

        double mean_abs = 0.0, max_err = 0.0;

        for (std::size_t i = 0; i < N; ++i) {
            float bfp_res = out_fp32[i];
            float ref_val = ref[i];
            float err = calc_rel_error(bfp_res, ref_val);

            mean_abs += std::fabs(bfp_res - ref_val);
            if (err > max_err) max_err = err;

            std::cout << std::setw(3) << i
                      << std::setw(12) << std::fixed << std::setprecision(4) << inputs[i]
                      << std::setw(12) << inputs_b[i]
                      << std::setw(16) << bfp_res
                      << std::setw(16) << ref_val
                      << std::setw(12) << std::setprecision(4) << err << "\n";
        }

        mean_abs /= double(N);
        print_error_stats(name, mean_abs, max_err);
    };

    //======================== TESTS: ADD, SUB, MUL, DIV ========================
    run_op_and_report(OP_ADD, "ADDITION (A + B)", ref_add);
    run_op_and_report(OP_SUB, "SUBTRACTION (A - B)", ref_sub);
    run_op_and_report(OP_MUL, "MULTIPLICATION (A * B)", ref_mul);
    run_op_and_report(OP_DIV, "DIVISION (A / B)", ref_div);

    //======================== TEST: ENCODE/DECODE ROUND-TRIP ===================
    std::cout << std::string(60, '=') << "\n";
    std::cout << "TEST: ENCODE/DECODE ROUND-TRIP (Block A)\n";
    std::cout << std::string(60, '=') << "\n\n";

    std::fill(out_fp32.begin(), out_fp32.end(), 0.f);

    bfp_kernel(OP_DECODE, n_blocks,
               dummy_fp32.data(),
               in_exp_a.data(), in_sign_a.data(), in_mant_a.data(),
               dummy_exp.data(), dummy_sign.data(), dummy_mant.data(),
               out_fp32.data(), dummy_exp.data(), dummy_sign.data(), dummy_mant.data());

    std::cout << std::setw(3) << "i"
              << std::setw(16) << "Original"
              << std::setw(16) << "Decoded"
              << std::setw(12) << "Err (%)\n";
    std::cout << std::string(47, '-') << "\n";

    double mean_abs = 0.0, max_err = 0.0;

    for (std::size_t i = 0; i < N; ++i) {
        float orig = inputs[i];
        float dec  = out_fp32[i];
        float err  = calc_rel_error(dec, orig);

        mean_abs += std::fabs(dec - orig);
        if (err > max_err) max_err = err;

        std::cout << std::setw(3) << i
                  << std::setw(16) << std::fixed << std::setprecision(6) << orig
                  << std::setw(16) << dec
                  << std::setw(12) << std::setprecision(4) << err << "\n";
    }

    mean_abs /= double(N);
    print_error_stats("ENCODE/DECODE", mean_abs, max_err);

    //======================== RESUMEN FINAL ========================
    std::cout << std::string(60, '=') << "\n";
    std::cout << "ALL KERNEL TESTS COMPLETED SUCCESSFULLY!\n";
    std::cout << std::string(60, '=') << "\n";

    return 0;
}
