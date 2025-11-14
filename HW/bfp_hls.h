#ifndef BFP_H
#define BFP_H

#include <ap_int.h>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <array>
#include <cmath>

//*============================================================================
//* CONFIGURACION DE BIAS PARA FORMATO BFP 
//*============================================================================
template<int WE, int WM>
struct BFP_bias {
    static constexpr int we = WE;                      
    static constexpr int wm = WM;                      
    static constexpr int bias_bfp = (1 << (WE - 1)) - 1;
};

//*============================================================================
//* ROUND TO NEAREST EVEN (SHIFT RIGHT) - Optimizado para HLS
//*============================================================================
static inline uint32_t helper_rne(uint32_t x, int shift) {
#pragma HLS INLINE
    
    // SHIFT LEFT (desplazamiento negativo)
    if (shift <= 0) {
        int s = -shift;
        if (s >= 32) return 0u;
        return (s == 0) ? x : (x << s);
    }

    // SHIFT RIGHT
    if (shift >= 32) return 0u;

    uint32_t q    = x >> shift;
    uint32_t rem  = x & ((1u << shift) - 1u);
    uint32_t half = 1u << (shift - 1);

    // RNE: NEAREST / TIES-TO-EVEN
    if (rem > half || (rem == half && (q & 1u))) {
        ++q;
    }
    return q;
}

//*============================================================================
//* REPRESENTACION DE BLOQUE BFP CON EXPONENTE GLOBAL
//*============================================================================
template<class Cfg, std::size_t Block_size>
struct BFP_Global {
    uint32_t exp_shared;                        // Exponente compartido E 
    std::array<uint32_t, Block_size> sign;      // Signos por elemento
    std::array<uint32_t, Block_size> mant;      // Mantissas (sin 1 implícito)
    std::array<uint32_t, Block_size> delta;    // NUEVO DELTA

    // RECONSTRUIR VALORES A FP32 PARA VALIDACION 
    float rebuid_FP32(std::size_t i) const {
#pragma HLS INLINE
        if (i >= Block_size) return 0.0f;

        const uint32_t mant_max = (1u << (Cfg::wm + 1)) - 1;
    
        // Detectar NaN
        if (mant[i] == (mant_max - 1) && delta[i] == 0) {
            // Construir NaN en FP32
            union {float f; uint32_t u;} nan_val;
            nan_val.u = 0x7FC00000;  // NaN canónico
            return nan_val.f;
        }
        
        // Detectar Infinito
        if (mant[i] == mant_max && delta[i] == 0) {
            // Construir Inf en FP32
            union {float f; uint32_t u;} inf_val;
            inf_val.u = sign[i] ? 0xFF800000 : 0x7F800000;  // -Inf : +Inf
            return inf_val.f;
        }
        // Detectar cero
        if (exp_shared == 0 && mant[i] == 0) return 0.0f;

        // Etapa de reconstruccion
        int   exp_shared_unbiased = int(exp_shared) - Cfg::bias_bfp;
        int   exp_real = exp_shared_unbiased - int(delta[i]); //  RECONSTRUCCION CON DELTA

        uint32_t mant_unshifted = mant[i] << delta[i];

        float mant_val     = float(mant_unshifted) / float(1u << Cfg::wm);
        float value        = std::ldexp(mant_val, exp_real); 
        //float value = std::ldexp(mant_val, exp_shared_unbiased);    
        return sign[i] ? -value : value;
    }
};

//*============================================================================
//* CODIFICACION DE BLOQUE: FP32 ARRAY -> BFP_Global
//* Calcula Emax y usa RNE para cuantización y DELTA
//*============================================================================
template<class Cfg, std::size_t Block_size>
BFP_Global<Cfg, Block_size> encode_block(const std::array<float, Block_size>& xs) {
#pragma HLS INLINE off
    
    BFP_Global<Cfg, Block_size> out{};

    //*========================================================================
    //* FASE 1: HALLAR EL EXPONENTE MAXIMO (Emax)
    //*========================================================================
    int Emax = std::numeric_limits<int>::min();
    
FIND_EMAX:
    for (std::size_t i = 0; i < Block_size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
        
        float num_fp32 = xs[i];
        
        // Interpretación de float a 32 bits
        union {float f; uint32_t u;} u = {num_fp32};
        
        int exp_fp32 = int((u.u >> 23) & 0xFF);
        if (exp_fp32 == 0) continue;  // Skip ceros denormalizados
        
        // Exponente real sin bias
        int exp_unbiased = exp_fp32 - 127; 
        if (exp_unbiased > Emax) {
            Emax = exp_unbiased;
        }
    }

    //*========================================================================
    //* VALIDAR SI EL BLOQUE SON TODOS CEROS
    //*========================================================================
    if (Emax == std::numeric_limits<int>::min()) {
        out.exp_shared = 0;
        out.sign.fill(0);
        out.mant.fill(0);
        out.delta.fill(0);
        return out;
    }   
    
    //*========================================================================
    //* FASE 2: CALCULAR EXPONENTE COMPARTIDO PARA BFP CON BIAS
    //*========================================================================
    int exp_shared_bfp = Emax + Cfg::bias_bfp;
    
    // Clamping para asegurar que cabe en WE bits
    if (exp_shared_bfp < 0) exp_shared_bfp = 0;
    if (exp_shared_bfp > (1 << Cfg::we) - 1) exp_shared_bfp = (1 << Cfg::we) - 1;
    
    out.exp_shared = uint32_t(exp_shared_bfp);

    //*========================================================================
    //* FASE 3: CUANTIZAR CADA ELEMENTO CON EXPONENTE MAX (SHIFT & RNE)
    //* CALCULO DE DELTAS
    //*========================================================================
    const uint32_t mant_max = (1u << (Cfg::wm + 1)) - 1;

QUANTIZE_ELEMENTS:
    for (std::size_t i = 0; i < Block_size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
        
        float num_fp32 = xs[i];
        
        if (num_fp32 == 0.0f) {
            out.sign[i] = 0;
            out.mant[i] = 0;
            out.delta[i] = 0; //DELTA
            continue;
        }
        
        //* EXTRAER LOS CAMPOS DE FP32
        union {float f; uint32_t u;} u = {num_fp32};
        
        uint32_t s = (u.u >> 31) & 0x1;
        int exp_fp32 = int((u.u >> 23) & 0xFF);
        uint32_t mant_fp32 = u.u & 0x7FFFFF;  // Mantisa de FP32

        // Deteccion de Nan/inf 
        if (exp_fp32 == 0xFF) {
            out.sign[i] = s;
            // Distinguir entre Inf y NaN
            if (mant_fp32 == 0) {
                // Es INFINITO: exp_max, mant_max, delta=0
                out.mant[i] = mant_max;
                out.delta[i] = 0;
            } else {
                // Es NaN: exp_max, mant_max-1 (para diferenciar), delta=0
                out.mant[i] = mant_max - 1;
                out.delta[i] = 0;
            }
        
            // Actualizar Emax para que el exponente compartido sea máximo
            if (Emax < ((1 << Cfg::we) - 1 - Cfg::bias_bfp)) {
                Emax = (1 << Cfg::we) - 1 - Cfg::bias_bfp;
            }
            continue;
        }
        
        if (exp_fp32 == 0) {
            out.sign[i] = 0;
            out.mant[i] = 0;
            out.delta[i] = 0; //DELTA PARA DESNORMALIZAR
            continue;
        }
        
        //* CONSTRUIR MANTISSA DE 24 BITS CON 1 IMPLICITO
        uint32_t mant24 = (u.u & 0x7FFFFF) | (1u << 23);
        int exp_unbiased = exp_fp32 - 127;

        //* CALCULO PARA GUARDAR DELTA 
        int delta_val = Emax - exp_unbiased;
        out.delta[i] = uint32_t(delta_val);     //GUARDAR DELTA
        
        //* REDUCCION DE MANTISSA A WM BITS & ALINEAR CON EXPONENTE COMPARTIDO
        int shift_total = (23 - Cfg::wm) + (Emax - exp_unbiased);
        
        uint32_t mant_reduced;
        
        if (shift_total >= 31) {
            mant_reduced = 0u;  // Underflow
        } else if (shift_total >= 0) {
            mant_reduced = helper_rne(mant24, shift_total);
        } else {
            mant_reduced = mant24 << (-shift_total);
        }
        
        // Saturar si excede el máximo permitido
        if (mant_reduced > mant_max) {
            mant_reduced = mant_max;
        }
        
        //* GUARDAR SIGNO Y MANTISSA
        out.sign[i] = s;
        out.mant[i] = mant_reduced;
    }
    
    return out;
}

//*============================================================================
//* DECODIFICACION DE BLOQUE: BFP_Global -> FP32 ARRAY
//* Reconstruye valores FP32 desde representación BFP
//*============================================================================
template<class Cfg, std::size_t Block_size>
std::array<float, Block_size> decode_block(const BFP_Global<Cfg, Block_size>& blk) {
#pragma HLS INLINE off
    
    std::array<float, Block_size> result;
    
DECODE_ELEMENTS:
    for (std::size_t i = 0; i < Block_size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
        
        result[i] = blk.rebuid_FP32(i);
    }
    
    return result;
}

#endif // BFP_H
