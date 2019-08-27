/*
    Copyright (c) 2017 Richard Veras, Doru Popovici, Ford Seidel,
    Tze Meng Low, Franz Franchetti

    This file is part of The Wrapped Schedulable (WS) Macro Instructions.

    The Wrapped Schedulable (WS) Macro Instructions is free software:
    you can redistribute it and/or modify it under the terms of the
    GNU General Public License as published by the Free Software
    Foundation, either version 3 of the License, or
    (at your option) any later version.

    The Wrapped Schedulable (WS) Macro Instructions is distributed
    in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    without even the implied warranty of MERCHANTABILITY or FITNESS
    FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with The Wrapped Schedulable (WS) Macro Instructions.  If not,
    see <http://www.gnu.org/licenses/>.


    If you are using The Wrapped Schedulable (WS) Macro Instructions in
    your research project, please cite:

    @inproceedings{veras_hands_off_hands_on,
      author = {Veras, Richard and Popovici, Doru Thom and Low,
                Tze Meng and Franchetti, Franz},
      title = {Compilers, Hands-off My Hands-on Optimizations},
      booktitle = {Proceedings of the 3rd Workshop on Programming
                   Models for SIMD/Vector Processing},
      series = {WPMVP '16},
      year = {2016},
      isbn = {978-1-4503-4060-1},
      location = {Barcelona, Spain},
      pages = {4:1--4:8},
      articleno = {4},
      numpages = {8},
      url = {http://doi.acm.org/10.1145/2870650.2870654},
      doi = {10.1145/2870650.2870654},
      acmid = {2870654},
      publisher = {ACM},
      address = {New York, NY, USA}, }
*/

#ifndef _WS_MAPPING_AVX_H
#define _WS_MAPPING_AVX_H

#include "immintrin.h"

/*
  DATATYPES:
  {vlen}x{basetype}

  i.e. 4xF64 is a vector of 4 doubles

  where basetupes are:
  + Generic ... GEN
  + Signed Integer .. I8,I16,I32,I64 ..
  + Unsigned Integer .. U8,U16,U32,U64 ..
  + Floating Point   .. F16, F32, F64 ...

  Inside instructions we tag the data type with its location
  {loc}{vlen}x{basetype}
  + R --- register
  + M --- memory
  + CL1, CL2, CL3, CL4 --- L1,L2,L3,L4 cache
  + SP1, SP2.. --- Scratch Pad


  INSTRUCTIONS:
  if all operands are the same type:
  WS_{OP}_{LOC}{DT}(src_a,src_b,dst_c)

  if the operands are different:

  WS_{OP}_{dtA}_TO_{dtB}
  WS_{OP}_{dtA}_{dtB}_TO_{dtC}
  WS_{OP}_{dtA}_{dtB}_{dtC}_TO_{dtD}


  PERMUTATIONS:

*/


/*
  Macros
*/
#define DTYPE(vlen,base)         vlen##x##base
#define DTYPE_LOC(loc,vlen,base) loc##vlen##x##base

#define WS_OP(op,dt) WS_##op##_##dt

// WS_UNARY
// WS_BINARY
// WS_TERNARY

/*
  Data Types
*/
// basetypes
#define WS_I32_T int // NOTE: on some archs
#define WS_F32_T float
#define WS_F64_T double

// Vector
#define WS_8xI32_T __m256i
#define WS_8xF32_T __m256
#define WS_4xF64_T __m256d

#define WS_4xF32_T __m128
#define WS_2xF64_T __m128d


/*
  Generic Operators:
  this should be moved

*/

#define WS_OP_MIN(_areg_,_breg_,_creg_)                                 \
    _creg_ = ({ __typeof__ (_areg_) _a = (_areg_);                      \
            __typeof__ (_breg_) _b = (_breg_);                          \
            _a < _b ? _a : _b; })

#define WS_OP_MAX(_areg_,_breg_,_creg_)                                 \
    _creg_ = ({ __typeof__ (_areg_) _a = (_areg_);                      \
            __typeof__ (_breg_) _b = (_breg_);                          \
            _a < _b ? _a : _b; })


/*
   Binary Operations
*/
// TODO: These macros can get in a really, really nasty
// corner case. I need to throw _areg_, _breg_ into temp
// variables, Or assume SSA


#define WS_GEN_BINARY_R1xGEN(_op_,_areg_,_breg_,_creg_) \
    {\
        _creg_ = (_areg_) _op_ (_breg_);        \
    }

#define WS_GEN_BINARY_MOD_R1xGEN(_op_,_areg_,_breg_,_creg_,_n_)    \
    {                                                       \
        _creg_ = ((_areg_) _op_ (_breg_)) % (_n_);          \
    }


#define WS_GEN_UNARY_R1xGEN(_op_,_areg_,_creg_) \
    {\
        _creg_ =  _op_ (_areg_);                \
    }




//  AND: c = a & b
#define WS_AND_R1xI32(areg,breg,creg) WS_GEN_BINARY_R1xGEN(&,areg,breg,creg)
#define WS_AND_R1xI64(areg,breg,creg) WS_GEN_BINARY_R1xGEN(&,areg,breg,creg)
#define WS_AND_R1xF32(areg,breg,creg) WS_GEN_BINARY_R1xGEN(&,areg,breg,creg)
#define WS_AND_R1xF64(areg,breg,creg) WS_GEN_BINARY_R1xGEN(&,areg,breg,creg)

#define WS_AND_R8xI32(areg,breg,creg)           \
    {                                                                   \
        __asm__ volatile(                                               \
            "vpand %[vsrca],%[vsrcb],%[vsrcc]\n"                       \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }

#define WS_AND_R4xI64(areg,breg,creg)           \
    {                                                                   \
        __asm__ volatile(                                               \
            "vpand %[vsrca],%[vsrcb],%[vsrcc]\n"                       \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }

#define WS_AND_R4xF64(areg,breg,creg)                                   \
    {                                                                   \
        __asm__ volatile(                                               \
            "vandpd %[vsrca],%[vsrcb],%[vsrcc]\n"                       \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }

#define WS_AND_R8xF32(areg,breg,creg)           \
    {                                                                   \
        __asm__ volatile(                                               \
            "vandps %[vsrca],%[vsrcb],%[vsrcc]\n"                       \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }

#define WS_AND_R2xF64(areg,breg,creg) WS_AND_R4xF64(areg,breg,creg)
#define WS_AND_R4xF32(areg,breg,creg) WS_AND_R8xF32(areg,breg,creg)

// OR: c = a | b
#define WS_OR_R1xI32(areg,breg,creg) WS_GEN_BINARY_R1xGEN(|,areg,breg,creg)
#define WS_OR_R1xI64(areg,breg,creg) WS_GEN_BINARY_R1xGEN(|,areg,breg,creg)
#define WS_OR_R1xF32(areg,breg,creg) WS_GEN_BINARY_R1xGEN(|,areg,breg,creg)
#define WS_OR_R1xF64(areg,breg,creg) WS_GEN_BINARY_R1xGEN(|,areg,breg,creg)

#define WS_OR_R8xI32(areg,breg,creg)                                    \
    {                                                                   \
        __asm__ volatile(                                               \
            "vpor %[vsrca],%[vsrcb],%[vsrcc]\n"                         \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }

#define WS_OR_R4xI64(areg,breg,creg)                                    \
    {                                                                   \
        __asm__ volatile(                                               \
            "vpor %[vsrca],%[vsrcb],%[vsrcc]\n"                         \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }

#define WS_OR_R4xF64(areg,breg,creg)                                    \
    {                                                                   \
        __asm__ volatile(                                               \
            "vorpd %[vsrca],%[vsrcb],%[vsrcc]\n"                        \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }

#define WS_OR_R8xF32(areg,breg,creg)                                    \
    {                                                                   \
        __asm__ volatile(                                               \
            "vorps %[vsrca],%[vsrcb],%[vsrcc]\n"                        \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }


// XOR: c = a ^ b
#define WS_XOR_R1xI32(areg,breg,creg) WS_GEN_BINARY_R1xGEN(^,areg,breg,creg)
#define WS_XOR_R1xI64(areg,breg,creg) WS_GEN_BINARY_R1xGEN(^,areg,breg,creg)

#define WS_XOR_R8xI32(areg,breg,creg)                                   \
    {                                                                   \
        __asm__ volatile(                                               \
            "vpxor %[vsrca],%[vsrcb],%[vsrcc]\n"                        \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }

#define WS_XOR_R4xI64(areg,breg,creg)                                   \
    {                                                                   \
        __asm__ volatile(                                               \
            "vpxor %[vsrca],%[vsrcb],%[vsrcc]\n"                        \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }

#define WS_XOR_R4xF64(areg,breg,creg)                                   \
    {                                                                   \
        __asm__ volatile(                                               \
            "vxorpd %[vsrca],%[vsrcb],%[vsrcc]\n"                       \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }

#define WS_XOR_R8xF32(areg,breg,creg)                                   \
    {                                                                   \
        __asm__ volatile(                                               \
            "vxorps %[vsrca],%[vsrcb],%[vsrcc]\n"                       \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }




// MOD: c = a mod b
#define WS_MOD_R1xI32(areg,breg,creg) WS_GEN_BINARY_R1xGEN(%,areg,breg,creg)
#define WS_MOD_R1xI64(areg,breg,creg) WS_GEN_BINARY_R1xGEN(%,areg,breg,creg)

// MAGIC MOD!
// TODO: This is hardcoded for 91
// http://www.hackersdelight.org/magic.htm
// https://stackoverflow.com/questions/27560819/how-to-calculate-mod-remainder-using-sse
#define WS_MODCONST_R8xI32(areg,creg,_modval_)                          \
    {                                                                   \
        const int factor = 3020636341;                                  \
        const int shift  = 6;                                           \
        __typeof__(areg) vfactor = _mm256_castsi256_ps(_mm256_set1_epi32(factor)); \
        __typeof__(areg) vshift;                                        \
        __typeof__(areg) mulres;                                        \
        WS_MUL_R8xI32(areg,vfactor,mulres);                             \
        creg = _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_castps_si256(mulres),6)); \
    }



// Shift Right Arithmetic
/*
#define WS_SRA_R8xI32(areg,breg,creg)                                   \
    {                                                                   \
        __asm__ volatile(                                               \
            "vpsrad %[vsrca],%[vsrcb],%[vsrcc]\n"                       \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }
*/

// ADDMOD
#define WS_ADDMOD_R1xI32(areg,breg,creg,_n_) WS_GEN_BINARY_MOD_R1xGEN(+,areg,breg,creg,_n_)
#define WS_ADDMOD_R1xI64(areg,breg,creg,_n_) WS_GEN_BINARY_MOD_R1xGEN(+,areg,breg,creg,_n_)

// MULMOD
#define WS_MULMOD_R1xI32(areg,breg,creg,_n_) WS_GEN_BINARY_MOD_R1xGEN(*,areg,breg,creg,_n_)
#define WS_MULMOD_R1xI64(areg,breg,creg,_n_) WS_GEN_BINARY_MOD_R1xGEN(*,areg,breg,creg,_n_)


// ADD: c = a + b
#define WS_ADD_R1xF32(areg,breg,creg) WS_GEN_BINARY_R1xGEN(+,areg,breg,creg)
#define WS_ADD_R1xF64(areg,breg,creg) WS_GEN_BINARY_R1xGEN(+,areg,breg,creg)
#define WS_ADD_R1xI32(areg,breg,creg) WS_GEN_BINARY_R1xGEN(+,areg,breg,creg)
#define WS_ADD_R1xI64(areg,breg,creg) WS_GEN_BINARY_R1xGEN(+,areg,breg,creg)

#define WS_ADD_R8xI32(areg,breg,creg)                                   \
    {                                                                   \
        __asm__ volatile(                                               \
            "vpaddd %[vsrca],%[vsrcb],%[vsrcc]\n"                       \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }

#define WS_ADD_R4xI64(areg,breg,creg)                                   \
    {                                                                   \
        __asm__ volatile(                                               \
            "vpaddq %[vsrca],%[vsrcb],%[vsrcc]\n"                       \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }

#define WS_ADD_R8xF32(areg,breg,creg)                                   \
    {                                                                   \
        __asm__ volatile(                                               \
            "vaddps %[vsrca],%[vsrcb],%[vsrcc]\n"                       \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }

#define WS_ADD_R4xF64(areg,breg,creg)                                   \
    {                                                                   \
        __asm__ volatile(                                               \
            "vaddpd %[vsrca],%[vsrcb],%[vsrcc]\n"                       \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }

#define WS_ADD_R4xF32(areg,breg,creg) WS_ADD_R8xF32(areg,breg,creg)
#define WS_ADD_R2xF64(areg,breg,creg) WS_ADD_R4xF64(areg,breg,creg)

// MUL: c = a * b
// Note: integer multiply only does the lower bits on AVX
#define WS_MUL_R1xF32(areg,breg,creg) WS_GEN_BINARY_R1xGEN(*,areg,breg,creg)
#define WS_MUL_R1xF64(areg,breg,creg) WS_GEN_BINARY_R1xGEN(*,areg,breg,creg)
#define WS_MUL_R1xI32(areg,breg,creg) WS_GEN_BINARY_R1xGEN(*,areg,breg,creg)
#define WS_MUL_R1xI64(areg,breg,creg) WS_GEN_BINARY_R1xGEN(*,areg,breg,creg)

#define WS_MUL_R4xF64(areg,breg,creg)           \
    {                                                                   \
        __asm__ volatile(                                               \
            "vmulpd %[vsrca],%[vsrcb],%[vsrcc]\n"                       \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }

#define WS_MUL_R8xF32(areg,breg,creg)                                   \
    {                                                                   \
        __asm__ volatile(                                               \
            "vmulps %[vsrca],%[vsrcb],%[vsrcc]\n"                       \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }


// The AVX interger multiply can only do one lane
// a --> al,ah
// b --> bl,bh
// al*bl
// cl = al*bl
// ch = ah*bh
// creg <-- cl,ch
#define WS_HALF_MUL_R8xI32(areg,breg,creg)                              \
    {                                                                   \
        __asm__ volatile(                                               \
            "vpmuldq %[vsrca],%[vsrcb],%[vsrcc]\n"                       \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }

// TODO: this is incorrect. It only does the lower half of each lane
//       not the lower lane
#define WS_MUL_R8xI32(areg,breg,creg)                                   \
    {                                                                   \
        __typeof__(areg) al = areg;                                     \
        __typeof__(areg) ah;                                            \
        __typeof__(areg) bl = breg;                                     \
        __typeof__(areg) bh;                                            \
        __typeof__(areg) cl;                                            \
        __typeof__(areg) chr;                                           \
        __typeof__(areg) ch;                                            \
        WS_PERMUTE_ID2xP2xID2_R8xF32(WS_PERM_ID2xP2xID2(1,0),areg,ah);          \
        WS_PERMUTE_ID2xP2xID2_R8xF32(WS_PERM_ID2xP2xID2(1,0),breg,bh);          \
        WS_HALF_MUL_R8xI32(al,bl,cl);                                   \
        WS_HALF_MUL_R8xI32(ah,bh,chr);                                  \
        WS_PERMUTE_ID2xP2xID2_R8xF32(WS_PERM_ID2xP2xID2(1,0),chr,ch);           \
        creg = _mm256_blend_ps(cl,ch,0b00001111);                       \
    }


// DIV: c = a/b
#define WS_DIV_R4xF64(areg,breg,creg)                                   \
    {                                                                   \
        __asm__ volatile(                                               \
            "vdivpd %[vsrca],%[vsrcb],%[vsrcc]\n"                       \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }

#define WS_DIV_R8xF32(areg,breg,creg)                                   \
    {                                                                   \
        __asm__ volatile(                                               \
            "vdivps %[vsrca],%[vsrcb],%[vsrcc]\n"                       \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }

// SUB:  c = a - b
#define WS_SUB_R1xF32(areg,breg,creg) WS_GEN_BINARY_R1xGEN(-,areg,breg,creg)
#define WS_SUB_R1xF64(areg,breg,creg) WS_GEN_BINARY_R1xGEN(-,areg,breg,creg)

#define WS_SUB_R8xI32(areg,breg,creg)                                   \
    {                                                                   \
        __asm__ volatile(                                               \
            "vpsubd %[vsrca],%[vsrcb],%[vsrcc]\n"                       \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }

#define WS_SUB_R4xF64(areg,breg,creg)                                   \
    {                                                                   \
        __asm__ volatile(                                               \
            "vsubpd %[vsrca],%[vsrcb],%[vsrcc]\n"                       \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }

#define WS_SUB_R8xF32(areg,breg,creg)                                   \
    {                                                                   \
        __asm__ volatile(                                               \
            "vsubps %[vsrca],%[vsrcb],%[vsrcc]\n"                       \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }


// MIN: c = min(a,b)
#define WS_MIN_R1xF32(areg,breg,creg) WS_OP_MIN(areg,breg,creg)
#define WS_MIN_R1xF64(areg,breg,creg) WS_OP_MIN(areg,breg,creg)
#define WS_MIN_R1xI32(areg,breg,creg) WS_OP_MIN(areg,breg,creg)
#define WS_MIN_R1xI64(areg,breg,creg) WS_OP_MIN(areg,breg,creg)


#define WS_MIN_R8xI32(areg,breg,creg)                                   \
    {                                                                   \
        __asm__ volatile(                                               \
            "vpminsd %[vsrca],%[vsrcb],%[vsrcc]\n"                      \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }

#define WS_MIN_R4xF64(areg,breg,creg)                                   \
    {                                                                   \
        __asm__ volatile(                                               \
            "vminpd %[vsrca],%[vsrcb],%[vsrcc]\n"                       \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }

#define WS_MIN_R8xF32(areg,breg,creg)                                   \
    {                                                                   \
        __asm__ volatile(                                               \
            "vminps %[vsrca],%[vsrcb],%[vsrcc]\n"                       \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }

// MAX: c = max(a,b)
#define WS_MAX_R1xF32(areg,breg,creg) WS_OP_MAX(areg,breg,creg)
#define WS_MAX_R1xF64(areg,breg,creg) WS_OP_MAX(areg,breg,creg)
#define WS_MAX_R1xI32(areg,breg,creg) WS_OP_MAX(areg,breg,creg)
#define WS_MAX_R1xI64(areg,breg,creg) WS_OP_MAX(areg,breg,creg)

#define WS_MAX_R8xI32(areg,breg,creg)                                   \
    {                                                                   \
        __asm__ volatile(                                               \
            "vpmaxsd %[vsrca],%[vsrcb],%[vsrcc]\n"                      \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }

#define WS_MAX_R4xF64(areg,breg,creg)                                   \
    {                                                                   \
        __asm__ volatile(                                               \
            "vmaxpd %[vsrca],%[vsrcb],%[vsrcc]\n"                       \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }

#define WS_MAX_R8xF32(areg,breg,creg)                                   \
    {                                                                   \
        __asm__ volatile(                                               \
            "vmaxps %[vsrca],%[vsrcb],%[vsrcc]\n"                       \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }


// POPCOUNT
#define WS_POPCOUNT_R1xI32(areg,creg)           \
    {                                           \
        creg = __builtin_popcount( areg );      \
    }
#define WS_POPCOUNT_R1xI64(areg,creg)           \
    {                                           \
        creg = __builtin_popcountl( areg );      \
    }



/* TODO: Need to figure out flags for vcmpps */
#define WS_EQ_R8xF32(areg,breg,creg)
#define WS_GT_R8xF32(areg,breg,creg)
#define WS_LT_R8xF32(areg,breg,creg)
#define WS_GTEQ_R8xF32(areg,breg,creg)
#define WS_LTEQ_R8xF32(areg,breg,creg)












/*
   Fused Binary Operations
*/
#define WS_FMA_R4xF64(areg,breg,creg_pre,creg_acc)                      \
    {                                                                   \
        creg_acc = creg_pre;                                            \
        __asm__ volatile(                                               \
            "vfmadd231pd %[vsrca],%[vsrcb],%[vsrcc]\n"                  \
            : [vsrcc] "+x"(creg_acc)                                    \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }

#define WS_FMA_R8xF32(areg,breg,creg_pre,creg_acc)                      \
    {                                                                   \
        creg_acc = creg_pre;                                            \
        __asm__ volatile(                                               \
            "vfmadd231ps %[vsrca],%[vsrcb],%[vsrcc]\n"                  \
            : [vsrcc] "+x"(creg_acc)                                    \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg)                                         \
            );                                                          \
    }

#define WS_FMA_R2xF64(areg,breg,creg_pre,creg_acc) WS_FMA_R4xF64(areg,breg,creg_pre,creg_acc)
#define WS_FMA_R4xF32(areg,breg,creg_pre,creg_acc) WS_FMA_R8xF32(areg,breg,creg_pre,creg_acc)

/*
  Broadcast Operations
*/
#define WS_BCAST_M1xF64_TO_R4xF64(src,dst)      \
    __asm__ volatile(                           \
        "vbroadcastsd %[addr], %[rvdest]\n"     \
        : [rvdest] "=x"(dst)                    \
        : [addr] "m"(src)                       \
        );

#define WS_BCAST_M1xF32_TO_R8xF32(src,dst)      \
    __asm__ volatile(                           \
        "vbroadcastss %[addr], %[rvdest]\n"     \
        : [rvdest] "=x"(dst)                    \
        : [addr] "m"(src)                       \
        );

#define WS_BCAST_M1xI32_TO_R8xI32(src,dst) \
    dst = _mm256_set1_epi32(src)


#define WS_BCAST_M1xF64_TO_R2xF64(src,dst) \
    dst = _mm_set1_pd(src)

#define WS_BCAST_M1xF32_TO_R4xF32(src,dst) \
    dst = _mm_set1_ps(src)

/*
  Reduction Operations
*/

/*
  Permutation Operations
*/

// TODO: Generate correct values
#define WS_PERM_P2xID2(_o0,_o1) 1
#define WS_PERM_ID2xP2(_o0,_o1) 5

// Figure out:
#define WS_PERM_P2xID4(_o0,_o1) 5
#define WS_PERM_ID2xP2xID2(_o0,_o1) 78
#define WS_PERM_ID4xP2(_o0,_o1) 177 //( (0x3&(_o1+2)<<6) | (0x3&(_o0+2) << 4) | (0x3&(_o1)<<2) | (0x3&(_o0)) ) // 177

#define WS_PERMUTE_ID2xP2_R4xF64(perm,areg,creg)                        \
    {                                                                   \
        __asm__ volatile(                                               \
            "vpermilpd %[ishf], %[vsrca],%[vsrcc]\n"                    \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [ishf]  "i"(perm)                                         \
            );                                                          \
    }

#define WS_PERMUTE_P2xID2_R4xF64(perm,areg,creg)                        \
    {                                                                   \
        __asm__ volatile(                                               \
            "vperm2f128 %[ishf], %[vsrca],%[vsrcb],%[vsrcc]\n"          \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(areg),                                        \
              [ishf]  "i"(perm)                                         \
            );                                                          \
    }

#define WS_PERMUTE_ID2xP4_R8xF32(perm,areg,creg)                        \
    {                                                                   \
        __asm__ volatile(                                               \
            "vpermilps %[ishf], %[vsrca],%[vsrcc]\n"                    \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [ishf]  "i"(perm)                                         \
            );                                                          \
    }

#define WS_PERMUTE_P2xID4_R8xF32(perm,areg,creg)                        \
    {                                                                   \
        __asm__ volatile(                                               \
            "vperm2f128 %[ishf], %[vsrca],%[vsrcb],%[vsrcc]\n"          \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(areg),                                        \
              [ishf]  "i"(perm)                                         \
            );                                                          \
    }


#define WS_PERMUTE_ID4xP2_R8xF32(perm,areg,creg)                        \
    {                                                                   \
        __asm__ volatile(                                               \
            "vpermilps %[ishf], %[vsrca],%[vsrcc]\n"                    \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [ishf]  "i"(perm)                                         \
            );                                                          \
    }


#define WS_PERMUTE_ID2xP2xID2_R8xF32(perm,areg,creg)                        \
    {                                                                   \
        __asm__ volatile(                                               \
            "vpermilps %[ishf], %[vsrca],%[vsrcc]\n"                    \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [ishf]  "i"(perm)                                         \
            );                                                          \
    }


#define WS_PERMUTE_ID2xP2_R4xF32(perm,areg,creg)                        \
    {                                                                   \
        const int perm_fix = 177;                                       \
        __asm__ volatile(                                               \
            "vshufps %[ishf], %[vsrca],%[vsrca],%[vsrcc]\n"                    \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [ishf]  "i"(perm_fix)                                         \
            );                                                          \
    }

#define WS_PERMUTE_P2xID2_R4xF32(perm,areg,creg)                      \
    {                                                                   \
        const int perm_fix = 78;\
        __asm__ volatile(                                               \
            "vshufps %[ishf], %[vsrca],%[vsrca],%[vsrcc]\n"                       \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [ishf]  "i"(perm_fix)                                         \
            );                                                          \
    }


#define WS_PERMUTE_P2_R2xF64(perm,areg,creg)                            \
    {                                                                   \
        const int perm_fix = 78;\
        __asm__ volatile(                                               \
            "vshufps %[ishf], %[vsrca],%[vsrca],%[vsrcc]\n"             \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [ishf]  "i"(perm_fix)                                     \
            );                                                          \
    }


#define WS_PERMUTE_2F128(perm,areg,breg,creg)                          \
    {                                                                   \
        __asm__ volatile(                                               \
            "vperm2f128 %[ishf], %[vsrca],%[vsrcb],%[vsrcc]\n"          \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
              [vsrcb] "x"(breg),                                        \
              [ishf]  "i"(perm)                                         \
            );                                                          \
    }

#define WS_PERMUTE4X64_PD(perm,areg,creg)                              \
    {                                                                   \
        __asm__ volatile(                                                \
            "vpermpd %[ishf], %[vsrca],%[vsrcc]\n"                      \
            : [vsrcc] "=x"(creg)                                        \
            : [vsrca] "x"(areg),                                        \
            [ishf]  "i"(perm)                                           \
            );                                                          \
    }
    
/*
  Movement Operations
*/

// read from memory
// NOTE: Playing single precision for double trick
#define WS_MOVE_M4xF64_TO_R4xF64(src,dst)    \
    __asm__ volatile(                        \
        "vmovapd %[addr], %[rvdest]\n"       \
        : [rvdest] "=x"(dst)                 \
        : [addr] "m"(*src)                   \
        );


#define WS_MOVE_M8xF32_TO_R8xF32(src,dst)    \
    __asm__ volatile(                        \
        "vmovaps %[addr], %[rvdest]\n"       \
        : [rvdest] "=x"(dst)                 \
        : [addr] "m"(*src)                   \
        );

#define WS_MOVEU_M4xF64_TO_R4xF64(src,dst)    \
    __asm__ volatile(                        \
        "vmovupd %[addr], %[rvdest]\n"       \
        : [rvdest] "=x"(dst)                 \
        : [addr] "m"(*src)                   \
        );


#define WS_MOVEU_M8xF32_TO_R8xF32(src,dst)    \
    __asm__ volatile(                        \
        "vmovups %[addr], %[rvdest]\n"       \
        : [rvdest] "=x"(dst)                 \
        : [addr] "m"(*src)                   \
        );


#define WS_MOVE_M8xI32_TO_R8xI32(src,dst) \
    dst = _mm256_load_si256(src)


#define WS_MOVE_M2xF64_TO_R2xF64(src,dst) WS_MOVE_M4xF64_TO_R4xF64(src,dst)
#define WS_MOVE_M4xF32_TO_R4xF32(src,dst) WS_MOVE_M8xF32_TO_R8xF32(src,dst)
#define WS_MOVEU_M2xF64_TO_R2xF64(src,dst) WS_MOVEU_M4xF64_TO_R4xF64(src,dst)
#define WS_MOVEU_M4xF32_TO_R4xF32(src,dst) WS_MOVEU_M8xF32_TO_R8xF32(src,dst)


// Write to memory
#define WS_MOVE_R4xF64_TO_M4xF64(src,dst)      \
    __asm__ volatile(                          \
        "vmovapd %[rvsrc], %[addr]\n"         \
        : [addr] "=m"(*dst)                     \
        : [rvsrc] "x"(src)                     \
        );

#define WS_MOVE_R8xF32_TO_M8xF32(src,dst)    \
    __asm__ volatile(                          \
        "vmovaps %[rvsrc], %[addr]\n"         \
        : [addr] "=m"(*dst)                     \
        : [rvsrc] "x"(src)                     \
        );

#define WS_MOVEU_R4xF64_TO_M4xF64(src,dst)    \
    __asm__ volatile(                          \
        "vmovupd %[rvsrc], %[addr]\n"         \
        : [addr] "=m"(*dst)                     \
        : [rvsrc] "x"(src)                     \
        );


#define WS_MOVEU_R8xF32_TO_M8xF32(src,dst)    \
    __asm__ volatile(                          \
        "vmovups %[rvsrc], %[addr]\n"         \
        : [addr] "=m"(*dst)                     \
        : [rvsrc] "x"(src)                     \
        );


#define WS_MOVEU_R2xF64_TO_M2xF64(src,dst) WS_MOVEU_R4xF64_TO_M4xF64(src,dst)
#define WS_MOVEU_R4xF32_TO_M4xF32(src,dst) WS_MOVEU_R8xF32_TO_M8xF32(src,dst)


/* Movement/Set Operations */
#define WS_BCAST_IMM1xF64_TO_R4xF64(imm,dst) \
    dst = _mm256_set1_pd(imm);

#define WS_BCAST_IMM1xF32_TO_R8xF32(imm,dst) \
    dst = _mm256_set1_ps(imm);

#define WS_BCAST_IMM1xI32_TO_R8xI32(imm,dst) \
    dst = _mm256_set1_epi32(imm);

#define WS_BCAST_IMM1xF64_TO_R2xF64(imm,dst) \
    dst = _mm_set1_pd(imm);

#define WS_BCAST_IMM1xF32_TO_R4xF32(imm,dst) \
    dst = _mm_set1_ps(imm);


#define WS_ASSIGN_GEN(areg,creg) (creg) = (areg)

//#define WS_TCAST_GEN(atype,ctype,areg,creg) ((ctype)creg) = ((atype)areg)

// Absolute value:
#define WS_ABS_R1xF32(areg,creg) (creg) = fabsf(areg)
#define WS_ABS_R1xF64(areg,creg) (creg) = fabs(areg)


#define WS_ABS_R4xF64(areg,creg)                                        \
    {                                                                   \
        const __m256d mask = _mm256_set_epi64x( ~(0b1),~(0b1),~(0b1),~(0b1) ); \
        creg = _mm256_and_pd(mask,areg);                                \
    }

#define WS_ABS_R8xF32(areg,creg)                                        \
    {                                                                   \
        const __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32( ~(0b1) )); \
        creg = _mm256_and_ps(mask,areg);                                \
    }



// Square root
#define WS_SQRT_R1xF32(areg,creg) (creg) = sqrtf(areg)
#define WS_SQRT_R1xF64(areg,creg) (creg) = sqrt(areg)



// Scaling
//#define WS_SCALE_R8xF32(areg,creg,scale_ptr) WS_ASSIGN_GEN(areg,creg)

/*
#define WS_SCALE_R8xF32(areg,creg,scale_ptr)                    \
    {                                                           \
        __typeof__(areg) _scaling_;                             \
        __typeof__(creg) _res_;                                 \
        WS_BCAST_M1xF32_TO_R8xF32((scale_ptr[0]),_scaling_);    \
        WS_MUL_R8xF32(_scaling_,areg,_res_);                    \
        WS_ASSIGN_GEN(_res_,creg);                              \
    }
*/

#define WS_SCALE_R8xF32(areg,creg,scale_ptr)                    \
    {                                                           \
        __typeof__(areg) _scaling_;                               \
        __typeof__(creg) _res_;                                   \
        _scaling_ = _mm256_set1_ps((scale_ptr[0]));               \
        _res_     = _mm256_mul_ps(_scaling_,areg);                    \
        WS_ASSIGN_GEN(_res_,creg);                                    \
    }


#define WS_SCALE_R4xF64(areg,creg,scale_ptr)                    \
    {                                                           \
        __typeof__(areg) _scaling_;                             \
        __typeof__(creg) _res_;                                 \
        WS_BCAST_M1xF64_TO_R4xF64((scale_ptr[0]),_scaling_);    \
        WS_MUL_R4xF64(_scaling_,areg,_res_);                    \
        WS_ASSIGN_GEN(_res_,creg);                              \
    }

// Other
#define WS_PREFETCH(addr) _mm_prefetch((char *)(addr), _MM_HINT_T0)





/*
  Complex Functions
*/



/*
   Unscramble

   for i in v
     C[i]=pos_i[i]
*/

// note: not <sse4.1 compatible
#define WS_UNSCRAMBLE_R2xF64(pos0,pos1,creg )     \
    {                                             \
        creg = _mm_blend_pd(pos0,pos1,0b10);      \
    }

// note: not <sse4.1 compatible
#define WS_UNSCRAMBLE_R4xF32(pos0,pos1,pos2,pos3,creg )     \
    {                                                           \
        WS_4xF32_T pos01 = _mm_blend_ps(pos0,pos1,10);   \
        WS_4xF32_T pos23 = _mm_blend_ps(pos2,pos3,10);   \
        creg = _mm_blend_ps(pos01,pos23,12);               \
    }

#define WS_UNSCRAMBLE_R4xF64(pos0,pos1,pos2,pos3,creg )     \
    {                                                           \
        WS_4xF64_T pos01 = _mm256_blend_pd(pos0,pos1,10);   \
        WS_4xF64_T pos23 = _mm256_blend_pd(pos2,pos3,10);   \
        creg = _mm256_blend_pd(pos01,pos23,12);               \
    }

// TODO: get correct values for blend
#define WS_UNSCRAMBLE_R8xF32(pos0,pos1,pos2,pos3,pos4,pos5,pos6,pos7,creg ) \
    {                                                                   \
        WS_8xF32_T pos01   = _mm256_blend_ps(pos0,pos1,0b00000010);     \
        WS_8xF32_T pos23   = _mm256_blend_ps(pos2,pos3,0b00001000);     \
        WS_8xF32_T pos0123 = _mm256_blend_ps(pos01,pos23,0b00001100);   \
        WS_8xF32_T pos45   = _mm256_blend_ps(pos4,pos5,0b00100000);     \
        WS_8xF32_T pos67   = _mm256_blend_ps(pos6,pos7,0b10000000);     \
        WS_8xF32_T pos4567 = _mm256_blend_ps(pos45,pos67,0b11000000);   \
        creg = _mm256_blend_ps(pos0123,pos4567,0b11110000);             \
    }

#define WS_UNSCRAMBLE_R8xI32 WS_UNSCRAMBLE_R8xF32

#endif  /* _WS_MAPPING_AVX_H */
