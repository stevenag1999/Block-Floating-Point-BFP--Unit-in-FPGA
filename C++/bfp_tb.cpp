#include <iostream>
#include <array>
#include <bitset>
#include <cmath>
#include <iomanip>
#include <limits>   // ADD: para inf/metricas

#include "bfp.h"
#include "bfp_ops.h"

/*
    Cfg-A: WE =3, WM =4, W∆=4 (perfil ultra-compacto).
    Cfg-B: WE =4, WM =5, W∆ =5 (equilibrado).
    Cfg-C: WE =5, WM =7, W∆ =6 (mayor precisión).
    */
using Cfg = BFP_bias<8,23>;       
constexpr std::size_t N = 16;

static void print_block(const char* title,
                        const BFP_Global<Cfg,N>& blk,
                        const std::array<float,N>& src)
{
    int E = int(blk.exp_shared) - Cfg::bias_bfp;
    std::cout << title << "\n";
    std::cout << "exp_shared: dec=" << blk.exp_shared
              << " bin=" << std::bitset<Cfg::we>(blk.exp_shared)
              << " | Exp(real)=" << E << "\n";
    for (std::size_t i=0;i<N;++i){
        std::cout << "i="<< std::setw(2) << i
                  << "  FP32=" << std::setw(10) << src[i]
                  << " | sign=" << blk.sign[i]
                  << " mant(dec)=" << std::setw(6) << blk.mant[i]
                  << " mant(bin)=" << std::bitset<Cfg::wm+1>(blk.mant[i])
                  << "  D=" << std::setw(3) << blk.delta[i]
                  << "  rec=" << std::setw(10) << blk.rebuild_FP32(i)
                  << "\n";
    }
    std::cout << "--------------------------------\n";
}

// ADD: chequeo rápido de que Δ+codificación reconstruyen bien el bloque
static void check_recon(const char* label,
                        const BFP_Global<Cfg,N>& blk,
                        const std::array<float,N>& src)
{
    double mae = 0.0, maxe = 0.0;
    for (std::size_t i=0;i<N;++i){
        float rec = blk.rebuild_FP32(i);
        float e = std::fabs(rec - src[i]);
        mae += e; if (e > maxe) maxe = e;
    }
    mae /= double(N);
    std::cout << label << "  | Recon MAE=" << mae << "  MAX_ERR=" << maxe << "\n\n";
}

// ADD: util para correr un par (A,B) con el mismo estilo de tu TB
static void run_pair(const char* titulo,
                     const std::array<float,N>& A,
                     const std::array<float,N>& B)
{
    // Referencias FP32
    std::array<float,N> ref_add{}, ref_sub{}, ref_mul{}, ref_div{};
    for (std::size_t i=0;i<N;++i){
        ref_add[i] = A[i] + B[i];
        ref_sub[i] = A[i] - B[i];
        ref_mul[i] = A[i] * B[i];
        ref_div[i] = (B[i]==0.0f) ? std::copysign(INFINITY, A[i]) : A[i]/B[i];
    }

    // Codificación BFP
    auto blkA = encode_block<Cfg>(A);
    auto blkB = encode_block<Cfg>(B);

    // Imprime bloques (Δ incluido, como te gusta) + verificación de reconstrucción
    std::cout << "\n==== " << titulo << " ====\n";
    print_block("=== BLOQUE A (entrada codificada, con D_A[i]) ===", blkA, A);
    check_recon("CHECK D/A", blkA, A);  // ADD

    print_block("=== BLOQUE B (entrada codificada, con D_B[i]) ===", blkB, B);
    check_recon("CHECK D/B", blkB, B);  // ADD

    // Operaciones
    auto blk_add = add_blocks<Cfg>(blkA, blkB);
    auto blk_sub = sub_blocks<Cfg>(blkA, blkB);
    auto blk_mul = mul_blocks<Cfg>(blkA, blkB);
    auto blk_rcp = rcp_blocks<Cfg>(blkB);
    auto blk_div = div_blocks<Cfg>(blkA, blkB);

    // Reporte por operación (misma forma que tu TB original)
    auto report_op = [&](const char* name,
                         const BFP_Global<Cfg,N>& Z,
                         const std::array<float,N>& ref,
                         const std::array<float,N>& Aref,
                         const std::array<float,N>& Bref,
                         const char* mant_tag)
    {
        int Eop = int(Z.exp_shared) - Cfg::bias_bfp;
        std::cout << "=========== " << name << " ==========\n";
        std::cout << "exp_shared("<< name << "): dec=" << Z.exp_shared
                  << " bin=" << std::bitset<Cfg::we>(Z.exp_shared)
                  << " | Exp(real)=" << Eop << "\n";
        double mae = 0.0, maxe = 0.0;
        for (std::size_t i=0;i<N;++i){
            float rec = Z.rebuild_FP32(i);
            float err = std::fabs(rec - ref[i]);
            mae += err; if (err>maxe) maxe = err;

            std::cout << "i="<< std::setw(2) << i
                      << "  A="<< std::setw(10) << Aref[i]
                      << "  B="<< std::setw(10) << Bref[i]
                      << " | " << mant_tag << "=" << std::bitset<Cfg::wm+1>(Z.mant[i])
                      << "  rec="<< std::setw(12) << rec
                      << "  ref="<< std::setw(12) << ref[i]
                      << "  | abs_err="<< (std::isfinite(ref[i]) && ref[i]!=0.f ? (err/std::fabs(ref[i])*100.f) : err)
                      << (std::isfinite(ref[i]) && ref[i]!=0.f ? "%" : "")
                      << "\n";
        }
        mae /= double(N);
        std::cout << name << ":  MAE=" << mae << "   MAX_ERR=" << maxe << "\n\n";
    };

    report_op("ADD (A+B, por bloque)", blk_add, ref_add, A, B, "mant(ADD)");
    report_op("SUB (A-B, por bloque)", blk_sub, ref_sub, A, B, "mant(SUB)");
    report_op("MUL (A*B, por bloque)", blk_mul, ref_mul, A, B, "mant(MUL)");

    // RCP: bloque de B
    {
        int Erec = int(blk_rcp.exp_shared) - Cfg::bias_bfp;
        std::cout << "===== RCP (1/B) =====\n";
        std::cout << "exp_shared(RCP): dec=" << blk_rcp.exp_shared
                  << " bin=" << std::bitset<Cfg::we>(blk_rcp.exp_shared)
                  << " | Exp(real)=" << Erec << "\n";
        for (std::size_t i=0;i<N;++i){
            std::cout << "i="<< std::setw(2) << i
                      << "  B="<< std::setw(10) << B[i]
                      << " | mant(RCP)=" << std::bitset<Cfg::wm+1>(blk_rcp.mant[i])
                      << "  rec="<< std::setw(12) << blk_rcp.rebuild_FP32(i)
                      << "\n";
        }
        std::cout << "\n";
    }

    report_op("DIV via RCP (A*(1/B))", blk_div, ref_div, A, B, "mant(DIV)");
}

int main() {
    // ===== Par base (tu caso) =====
    std::array<float,N> A = {
        2.359f, 6.577f, 8.203f, 2.654f, 8.806f, 7.516f, 4.110f, 8.100f,
        5.454f, 9.992f, 2.153f, 8.330f, 3.800f, 3.196f, 7.823f, 1.912f
    };
    std::array<float,N> B = {
        -2.369f, 1.954f, -2.147f, 3.583f, 2.855f, 2.444f, 2.376f, 2.085f,
         3.030f, 3.606f, 5.555f, 3.100f, 6.330f, 3.405f, 8.901f, 2.789f
    };
    run_pair("Caso base (enunciado)", A, B);

    // ===== Pares extra para validar Δ =====
    // 1) Cerca de cero
    std::array<float,N> A1 = { 0.09f, 0.12f, 0.15f, 0.18f, 0.21f, 0.24f, 0.27f, 0.30f,
                               0.45f, 0.60f, 0.72f, 0.81f, 0.90f, 0.95f, 0.99f, 0.33f };
    std::array<float,N> B1 = { 0.11f, 0.13f, 0.16f, 0.19f, 0.22f, 0.25f, 0.28f, 0.31f,
                               0.47f, 0.58f, 0.70f, 0.83f, 0.88f, 0.93f, 0.97f, 0.35f };
    run_pair("Cerca de cero (A,B ~ 0)", A1, B1);

    // 2) Normales casi iguales (~10)
    std::array<float,N> A2 = { 9.98f,10.01f,10.02f, 9.97f,10.00f,10.05f, 9.95f,10.03f,
                               9.99f,10.04f,10.01f,10.02f, 9.96f,10.00f,10.03f, 9.97f };
    std::array<float,N> B2 = { 10.02f, 9.99f,10.00f,10.01f, 9.98f, 9.97f,10.03f,10.02f,
                               10.01f,10.00f, 9.96f,10.04f,10.05f, 9.95f,10.02f,10.00f };
    run_pair("Normales casi iguales (~10)", A2, B2);

    // 3) Muy disparejos (magnitudes y signos mezclados)
    std::array<float,N> A3 = { 1.0f,   3.0f,   7.0f,  12.0f,  25.0f,  40.0f,  60.0f,  85.0f,
                               120.0f, 175.0f, 140.0f, 320.0f, 410.0f, 450.0f, 40.0f,   5.0f };
    std::array<float,N> B3 = { 2.0f,   5.0f,   9.0f,  15.0f,  30.0f,  45.0f,  70.0f,  95.0f,
                               130.0f, 190.0f, 60.0f, 80.0f, 30.0f, 72.0f, 100.0f,  10.0f };
    run_pair("Muy disparejos (magnitudes mixtas)", A3, B3);

    return 0;
}
