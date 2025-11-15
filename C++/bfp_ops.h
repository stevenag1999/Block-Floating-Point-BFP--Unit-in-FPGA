#ifndef BFP_OPS_H
#define BFP_OPS_H

#include <cstdint>
#include <cstdlib>
#include "bfp.h"

// Utilidad local para clamp de exponente real a WE bits (sesgado)
template<class Cfg>
static inline uint32_t clamp_E_to_bfp(int Er){
    int Es = Er + Cfg::bias_bfp;
    if (Es < 0) Es = 0;
    if (Es > (1 << Cfg::we) - 1) Es = (1 << Cfg::we) - 1;
    return uint32_t(Es);
}

//* SUMA
// ALINEAR + SUMAR CON SIGNO + NORMALIZAR
template<class Cfg, std::size_t Block_size>
BFP_Global<Cfg, Block_size> add_blocks(const BFP_Global<Cfg, Block_size>& A,
                                        const BFP_Global<Cfg, Block_size>& B){

    BFP_Global<Cfg, Block_size> Z{};

    const uint32_t MANT_MAX = (1u << (Cfg::wm + 1)) - 1u;

    const int Ea = int(A.exp_shared) - Cfg::bias_bfp;
    const int Eb = int(B.exp_shared) - Cfg::bias_bfp;

    // FIX: E_base entre exponentes COMPARTIDOS (no por elemento)
    const int E_base = (Ea > Eb) ? Ea : Eb;

    // FIX: Alinear SOLO entre Ea→E_base y Eb→E_base (Δ NO se usa aquí)
    const int shiftA = (E_base > Ea) ? (E_base - Ea) : 0;
    const int shiftB = (E_base > Eb) ? (E_base - Eb) : 0;

    bool overflow_any = false;
    std::array<uint32_t, Block_size> Mag{};
    std::array<uint32_t, Block_size> Sgn{};

    for (std::size_t i = 0; i < Block_size; ++i) {
        const uint32_t Ma = (shiftA >= 32) ? 0u : helper_rne(A.mant[i], shiftA);
        const uint32_t Mb = (shiftB >= 32) ? 0u : helper_rne(B.mant[i], shiftB);

        const int32_t Sa = A.sign[i] ? -int32_t(Ma) : int32_t(Ma);
        const int32_t Sb = B.sign[i] ? -int32_t(Mb) : int32_t(Mb);
        const int32_t S  = Sa + Sb;

        const uint32_t sign = (S < 0) ? 1u : 0u;
        uint32_t mag = (S < 0) ? uint32_t(-S) : uint32_t(S);

        if (mag == 0u) { Sgn[i] = 0u; Mag[i] = 0u; }
        else {
            Sgn[i] = sign; Mag[i] = mag;
            if (mag > MANT_MAX) overflow_any = true;
        }
    }

    // 3) Normalización global (con saturación de exponente si hiciera falta)
    int E = E_base;
    if (overflow_any) {
        // sube exponente y divide mantisas por 2 con RNE
        ++E;
        for (std::size_t i = 0; i < Block_size; ++i) {
            uint32_t m = helper_rne(Mag[i], 1);
            if (m > MANT_MAX) m = MANT_MAX;
            Mag[i] = m;
            if (Mag[i] == 0u) Sgn[i] = 0u; // -0 → +0
        }
    } else {
        // “Llenado” si la magnitud global quedó por debajo de 2^WM
        uint32_t max_mag = 0u;
        for (auto v : Mag) if (v > max_mag) max_mag = v;

        if (max_mag == 0u) {
            Z.exp_shared = 0; Z.sign.fill(0u); Z.mant.fill(0u); Z.delta.fill(0);
            return Z;
        }

        int msb = -1;
        for (int b = Cfg::wm + 1; b >= 0; --b) if (max_mag & (1u << b)) { msb = b; break; }
        if (msb < Cfg::wm) {
            int shl = Cfg::wm - msb;
            E -= shl;
            for (std::size_t i = 0; i < Block_size; ++i) {
                uint64_t up = (uint64_t)Mag[i] << shl;
                if (up > MANT_MAX) up = MANT_MAX;
                Mag[i] = (uint32_t)up;
                if (Mag[i] == 0u) Sgn[i] = 0u;
            }
        }
    }

    // 4) Salida (Δ_out = 0)
    Z.exp_shared = clamp_E_to_bfp<Cfg>(E);
    for (std::size_t i = 0; i < Block_size; ++i) {
        Z.mant[i]  = Mag[i];
        Z.sign[i]  = (Mag[i] == 0u) ? 0u : Sgn[i];
        Z.delta[i] = 0;
    }

    bool all_zero = true;
    for (auto m : Z.mant) if (m != 0u) { all_zero = false; break; }
    if (all_zero) Z.exp_shared = 0;

    return Z;
}

//* RESTA 
// REINVERTIR EL SIGNO DE LOS BLOQUES 
template<class Cfg, std::size_t Block_size>
BFP_Global<Cfg, Block_size>
sub_blocks(const BFP_Global<Cfg, Block_size>& A,
           const BFP_Global<Cfg, Block_size>& B)
{

    BFP_Global<Cfg, Block_size> Bneg = B;

    for (std::size_t i = 0; i < Block_size; ++i) {
        if (Bneg.mant[i] == 0u) {Bneg.sign[i] = 0u;}          // FORZAR CERO 
        else { Bneg.sign[i] = Bneg.sign[i] ^ 1u;} // INVIERTE SIGNO
    }
    return add_blocks<Cfg, Block_size>(A, Bneg);
}


//* MULTIPLICACION  
// PRODUCTO + SHIFT + NORMALIZACION
template<class Cfg, std::size_t Block_size>
BFP_Global<Cfg,Block_size> mul_blocks(const BFP_Global<Cfg,Block_size> &A,
                                       const BFP_Global<Cfg,Block_size> &B){

    BFP_Global<Cfg, Block_size> Z{};

    const uint32_t MANT_MAX = (1u << (Cfg::wm + 1)) - 1u;
    // Exponentes reales compartidos de entrada
    const int Ea = int(A.exp_shared) - Cfg::bias_bfp;
    const int Eb = int(B.exp_shared) - Cfg::bias_bfp; 

    // FIX: Exponente del producto por bloque (constante)
    int E = Ea + Eb;

    bool overflow_any = false;
    std::array<uint32_t, Block_size> Mag{};
    std::array<uint32_t, Block_size> Sgn{};

    for (std::size_t i = 0; i < Block_size; ++i) {
        uint32_t sign = A.sign[i] ^ B.sign[i];

        // Producto de mantisas (hasta ~2*(WM+1) bits), reducir a 2^WM con RNE
        const uint64_t P    = uint64_t(A.mant[i]) * uint64_t(B.mant[i]);
        uint64_t q    = P >> Cfg::wm;
        uint64_t rem  = P & ((uint64_t(1) << Cfg::wm) - 1);
        const uint64_t half = uint64_t(1) << (Cfg::wm - 1);
        if (rem > half || (rem == half && (q & 1u))) ++q;

        uint32_t m = (uint32_t)q; // FIX: sin alineos por i (Δ ya pagado en encode)

        if (m == 0u) sign = 0u;
        Sgn[i] = sign;
        Mag[i] = m;

        if (m > MANT_MAX) overflow_any = true;
    }

    // 3) Normalización global
    if (overflow_any) {
        ++E;
        for (std::size_t i = 0; i < Block_size; ++i) {
            uint32_t m = helper_rne(Mag[i], 1);
            if (m > MANT_MAX) m = MANT_MAX;
            Mag[i] = m;
            if (Mag[i] == 0u) Sgn[i] = 0u;
        }
    } else {
        uint32_t max_mag = 0u;
        for (auto v : Mag) if (v > max_mag) max_mag = v;

        if (max_mag == 0u) {
            Z.exp_shared = 0; Z.sign.fill(0u); Z.mant.fill(0u); Z.delta.fill(0);
            return Z;
        }

        int msb = -1;
        for (int b = Cfg::wm + 1; b >= 0; --b) if (max_mag & (1u << b)) { msb = b; break; }
        if (msb < Cfg::wm) {
            int shl = Cfg::wm - msb;
            E -= shl;
            for (std::size_t i = 0; i < Block_size; ++i) {
                uint64_t up = (uint64_t)Mag[i] << shl;
                if (up > MANT_MAX) up = MANT_MAX;
                Mag[i] = (uint32_t)up;
                if (Mag[i] == 0u) Sgn[i] = 0u;
            }
        }
    }

    // 4) Construir salida (Δ_out = 0)
    Z.exp_shared = clamp_E_to_bfp<Cfg>(E);
    for (std::size_t i = 0; i < Block_size; ++i) {
        Z.mant[i]  = Mag[i];
        Z.sign[i]  = (Mag[i] == 0u) ? 0u : Sgn[i];
        Z.delta[i] = 0;
    }

    // Si todo cero → exp=0
    bool all_zero = true;
    for (auto m : Z.mant) if (m != 0u) { all_zero = false; break; }
    if (all_zero) Z.exp_shared = 0;

    return Z;

}


//* RECIPROCO (1/B) 
template<class Cfg, std::size_t Block_size>
BFP_Global<Cfg, Block_size>
rcp_blocks(const BFP_Global<Cfg, Block_size>& B)
{
    BFP_Global<Cfg, Block_size> R{};

    const uint32_t MANT_MAX = (1u << (Cfg::wm + 1)) - 1u;

    // Exponente real compartido del bloque B
    const int Eb = int(B.exp_shared) - Cfg::bias_bfp;

    // FIX: Exponente del recíproco por bloque (constante)
    int E = -Eb; // 2^(-eb)

    bool overflow_any = false;
    std::array<uint32_t, Block_size> Mag{};
    std::array<uint32_t, Block_size> Sgn{};

    for (std::size_t i = 0; i < Block_size; ++i) {
        if (B.mant[i] == 0u) {
            // 1/0 -> saturación representable
            Mag[i] = MANT_MAX;
            Sgn[i] = B.sign[i];
            continue;
        }

        uint32_t sign = B.sign[i];

        // qq ≈ (1 / mant_Bi) * 2^WM = (2^(2*WM)) / mant_Bi, con RNE
        const uint64_t Num = 1ull << (2 * Cfg::wm); // 1/mant -> 2^(mant)
        const uint64_t Den = (uint64_t)B.mant[i];
        uint64_t qq  = Num / Den;
        uint64_t rem = Num % Den;
        if ( (rem << 1) > Den || ((rem << 1) == Den && (qq & 1ull)) ) ++qq;

        uint32_t m = (uint32_t)qq; // FIX: sin realineos por i

        if (m == 0u) sign = 0u;
        Sgn[i] = sign;
        Mag[i] = m;

        if (m > MANT_MAX) overflow_any = true;
    }

    // 3) Normalización global
    if (overflow_any) {
        ++E;
        for (std::size_t i = 0; i < Block_size; ++i) {
            uint32_t m = helper_rne(Mag[i], 1);
            if (m > MANT_MAX) m = MANT_MAX;
            Mag[i] = m;
            if (Mag[i] == 0u) Sgn[i] = 0u;
        }
    } else {
        uint32_t max_mag = 0u;
        for (auto v : Mag) if (v > max_mag) max_mag = v;

        if (max_mag == 0u) {
            R.exp_shared = 0; R.sign.fill(0u); R.mant.fill(0u); R.delta.fill(0);
            return R;
        }

        int msb = -1;
        for (int b = Cfg::wm + 1; b >= 0; --b) if (max_mag & (1u << b)) { msb = b; break; }
        if (msb < Cfg::wm) {
            int shl = Cfg::wm - msb;
            E -= shl;
            for (std::size_t i = 0; i < Block_size; ++i) {
                uint64_t up = (uint64_t)Mag[i] << shl;
                if (up > MANT_MAX) up = MANT_MAX;
                Mag[i] = (uint32_t)up;
                if (Mag[i] == 0u) Sgn[i] = 0u;
            }
        }
    }

    // 4) Salida (Δ_out = 0)
    R.exp_shared = clamp_E_to_bfp<Cfg>(E);
    for (std::size_t i = 0; i < Block_size; ++i) {
        R.mant[i]  = Mag[i];
        R.sign[i]  = (Mag[i] == 0u) ? 0u : Sgn[i];
        R.delta[i] = 0;
    }

    // Si todo cero → exp=0
    bool all_zero = true;
    for (auto m : R.mant) if (m != 0u) { all_zero = false; break; }
    if (all_zero) R.exp_shared = 0;

    return R;
}



//* DIVISION VIA RECIPROCO:  A ÷ B  ≈  A × (1/B)  usando mul_blocks
template<class Cfg, std::size_t Block_size>
BFP_Global<Cfg, Block_size>
div_blocks(const BFP_Global<Cfg, Block_size>& A,
           const BFP_Global<Cfg, Block_size>& B)
{
    auto R = rcp_blocks<Cfg, Block_size>(B);
    return mul_blocks<Cfg, Block_size>(A, R); // A/R -> 1/B -> A(1/B)
}


#endif // BFP_OPS_H



       




