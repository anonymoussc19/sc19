#ifndef _SCATTER_KERNELS_H
#define _SCATTER_KERNELS_H
#include "tensor_kernels.h"
/*     #define DGEMM_OUTPUT_GS_BETA_NZ \ */
	/* vextractf128(imm(1), ymm0, xmm1) \ */
	/* vmovlpd(xmm0, mem(rcx)) \ */
	/* vmovhpd(xmm0, mem(rcx, rsi, 1)) \ */
	/* vmovlpd(xmm1, mem(rcx, rsi, 2)) \ */
	/* vmovhpd(xmm1, mem(rcx, r13, 1)) /\*\ */


#define DGEMM_INPUT_BETA_NZ_SCATTER_03          \
 mov(mem(rdx,0*8), r8)                          \
 add(rax, r8)                                   \
 vmovlpd(mem(rcx, r8, 8), xmm0, xmm0)           \
 mov(mem(rdx,1*8), r9)                          \
 add(rax, r9)                                   \
 vmovhpd(mem(rcx, r9, 8), xmm0, xmm0)           \
 mov(mem(rdx,2*8), r10)                         \
 add(rax, r10)                                  \
vmovlpd(mem(rcx, r10, 8), xmm1, xmm1)           \
mov(mem(rdx,3*8), r11)                          \
add(rax, r11)                                   \
vmovhpd(mem(rcx, r11, 8), xmm1, xmm1)           \
vperm2f128(imm(0x20), ymm1, ymm0, ymm0)


#define DGEMM_OUTPUT_BETA_NZ_SCATTER            \
 vextractf128(imm(1), ymm0, xmm1)               \
 vmovlpd(xmm0, mem(rcx, r8,8))                  \
 vmovhpd(xmm0, mem(rcx, r9, 8))                 \
 vmovlpd(xmm1, mem(rcx, r10, 8))                \
 vmovhpd(xmm1, mem(rcx, r11, 8))





#define DGEMM_INPUT_BETA_NZ_SCATTER_47 \
 mov(mem(rdx,4*8), r8)                          \
 add(rax, r8)                                   \
 vmovlpd(mem(rcx, r8, 8), xmm0, xmm0)           \
 mov(mem(rdx,5*8), r9)                          \
 add(rax, r9)                                   \
 vmovhpd(mem(rcx, r9, 8), xmm0, xmm0)           \
 mov(mem(rdx,6*8), r10)                         \
 add(rax, r10)                                  \
 vmovlpd(mem(rcx, r10, 8), xmm1, xmm1)          \
 mov(mem(rdx,7*8), r11)                         \
 add(rax, r11)                                  \
 vmovhpd(mem(rcx, r11, 8), xmm1, xmm1)          \
 vperm2f128(imm(0x20), ymm1, ymm0, ymm0)


#define DGEMM_SCT_PREFETCH_1ROW                 \
 mov(mem(rdx), r8)                              \
 add(rax, r8)                                   \
 prefetch(0, mem(rcx, r8, 8))                   \
 mov(mem(rdx,1*8), r9)                          \
 add(rax, r9)                                   \
 prefetch(0, mem(rcx, r9, 8))                   \
 mov(mem(rdx,2*8), r10)                         \
 add(rax, r10)                                  \
 prefetch(0, mem(rcx, r10, 8))                  \
 mov(mem(rdx,3*8), r11)                         \
 add(rax, r11)                                  \
 prefetch(0, mem(rcx, r11, 8))                  \
     mov(mem(rdx,4*8), r12)                     \
 add(rax, r12)                                  \
 prefetch(0, mem(rcx, r12, 8))                  \
     mov(mem(rdx,5*8), r13)                     \
 add(rax, r13)                                  \
 prefetch(0, mem(rcx, r13, 8))                  \
     mov(mem(rdx,6*8), r14)                     \
 add(rax, r14)                                  \
 prefetch(0, mem(rcx, r14, 8))                  \
 mov(mem(rdx,7*8), r15)                         \
 add(rax, r15)                                  \
 prefetch(0, mem(rcx, r15, 8))


void sct_dgemm_haswell_asm_6x8
(
unsigned int      k0,
    double*    restrict alpha,
    double*    restrict a,
    double*    restrict b,
    double*    restrict beta,
    double*    restrict c,
    unsigned long long*  rsct_c,   // 6 int
    unsigned long long *  csct_c    // 8 int
    )
{
    unsigned long long k_iter = k0/4;
unsigned     long long  k_left = k0%4;
    
    begin_asm()
        vzeroall()


        mov(var(b), rbx) // load b address -> rbx

        add(imm(32*4), rbx)
        // initialize loop by pre-loading
        vmovapd(mem(rbx, -4*32), ymm0)
        vmovapd(mem(rbx, -3*32), ymm1)
        // rbx[0-3] -> ymm0
        //rbx[4-7]  -> ymm1

/* #define MEM_4_(reg,off,scale,disp) [reg + off*scale + disp] */
/* #define MEM_3_(reg,off,scale) [reg + off*scale] */
/* #define MEM_2_(reg,disp) [reg + disp] */
/* #define MEM_1_(reg) [reg] */

        mov(var(c), rcx) // address c -> rcx
        mov(var(rsct_c), rdi)// address row_sct -> rdi
        mov(var(csct_c), rdx)// address col sct -> rdx



        //row 0
        mov(mem(rdi), rax)// [address row_sct+0*8] -> ax
        DGEMM_SCT_PREFETCH_1ROW


        //row 1
        mov(mem(rdi,1*8), rax)// [address row_sct+0*8] -> rsp
        DGEMM_SCT_PREFETCH_1ROW

        //row 2
        mov(mem(rdi,2*8), rax)// [address row_sct+0*8] -> rsp
        DGEMM_SCT_PREFETCH_1ROW
        

        //row 3
        mov(mem(rdi,3*8), rax)// [address row_sct+0*8] -> rsp
        DGEMM_SCT_PREFETCH_1ROW

        mov(mem(rdi,4*8), rax)// [address row_sct+0*8] -> rsp
        DGEMM_SCT_PREFETCH_1ROW

        mov(mem(rdi,5*8), rax)// [address row_sct+0*8] -> rsp
        DGEMM_SCT_PREFETCH_1ROW

        mov(var(a), rax) // load a address -> rax
        
        mov(var(k_iter), rsi) // i = k_iter;
        test(rsi, rsi) // check i via logical AND.
        je(.DCONSIDKLEFT) // if i == 0, jump to code that
	 // contains the k_left loop.
	

        
        label(.DLOOPKITER) // MAIN LOOP

        // iteration 0
        prefetch(0, mem(rax, 64*8))


        vbroadcastsd(mem(rax, 0*8), ymm2)
        vbroadcastsd(mem(rax, 1*8), ymm3)
        vfmadd231pd(ymm0, ymm2, ymm4)
        vfmadd231pd(ymm1, ymm2, ymm5)
        vfmadd231pd(ymm0, ymm3, ymm6)
        vfmadd231pd(ymm1, ymm3, ymm7)
	
        vbroadcastsd(mem(rax, 2*8), ymm2)
        vbroadcastsd(mem(rax, 3*8), ymm3)
        vfmadd231pd(ymm0, ymm2, ymm8)
        vfmadd231pd(ymm1, ymm2, ymm9)
        vfmadd231pd(ymm0, ymm3, ymm10)
        vfmadd231pd(ymm1, ymm3, ymm11)
	
        vbroadcastsd(mem(rax, 4*8), ymm2)
        vbroadcastsd(mem(rax, 5*8), ymm3)
        vfmadd231pd(ymm0, ymm2, ymm12)
        vfmadd231pd(ymm1, ymm2, ymm13)
        vfmadd231pd(ymm0, ymm3, ymm14)
        vfmadd231pd(ymm1, ymm3, ymm15)
	
        vmovapd(mem(rbx, -2*32), ymm0)
        vmovapd(mem(rbx, -1*32), ymm1)
	
        // iteration 1
        vbroadcastsd(mem(rax, 6*8), ymm2)
        vbroadcastsd(mem(rax, 7*8), ymm3)
        vfmadd231pd(ymm0, ymm2, ymm4)
        vfmadd231pd(ymm1, ymm2, ymm5)
        vfmadd231pd(ymm0, ymm3, ymm6)
        vfmadd231pd(ymm1, ymm3, ymm7)
	
        vbroadcastsd(mem(rax, 8*8), ymm2)
        vbroadcastsd(mem(rax, 9*8), ymm3)
        vfmadd231pd(ymm0, ymm2, ymm8)
        vfmadd231pd(ymm1, ymm2, ymm9)
        vfmadd231pd(ymm0, ymm3, ymm10)
        vfmadd231pd(ymm1, ymm3, ymm11)
	
        vbroadcastsd(mem(rax, 10*8), ymm2)
        vbroadcastsd(mem(rax, 11*8), ymm3)
        vfmadd231pd(ymm0, ymm2, ymm12)
        vfmadd231pd(ymm1, ymm2, ymm13)
        vfmadd231pd(ymm0, ymm3, ymm14)
        vfmadd231pd(ymm1, ymm3, ymm15)
	
        vmovapd(mem(rbx, 0*32), ymm0)
        vmovapd(mem(rbx, 1*32), ymm1)
	
        // iteration 2
        prefetch(0, mem(rax, 76*8))
	
        vbroadcastsd(mem(rax, 12*8), ymm2)
        vbroadcastsd(mem(rax, 13*8), ymm3)
        vfmadd231pd(ymm0, ymm2, ymm4)
        vfmadd231pd(ymm1, ymm2, ymm5)
        vfmadd231pd(ymm0, ymm3, ymm6)
        vfmadd231pd(ymm1, ymm3, ymm7)
	
        vbroadcastsd(mem(rax, 14*8), ymm2)
        vbroadcastsd(mem(rax, 15*8), ymm3)
        vfmadd231pd(ymm0, ymm2, ymm8)
        vfmadd231pd(ymm1, ymm2, ymm9)
        vfmadd231pd(ymm0, ymm3, ymm10)
        vfmadd231pd(ymm1, ymm3, ymm11)
	
        vbroadcastsd(mem(rax, 16*8), ymm2)
        vbroadcastsd(mem(rax, 17*8), ymm3)
        vfmadd231pd(ymm0, ymm2, ymm12)
        vfmadd231pd(ymm1, ymm2, ymm13)
        vfmadd231pd(ymm0, ymm3, ymm14)
        vfmadd231pd(ymm1, ymm3, ymm15)
	
        vmovapd(mem(rbx, 2*32), ymm0)
        vmovapd(mem(rbx, 3*32), ymm1)
	
        // iteration 3
        vbroadcastsd(mem(rax, 18*8), ymm2)
        vbroadcastsd(mem(rax, 19*8), ymm3)
        vfmadd231pd(ymm0, ymm2, ymm4)
        vfmadd231pd(ymm1, ymm2, ymm5)
        vfmadd231pd(ymm0, ymm3, ymm6)
        vfmadd231pd(ymm1, ymm3, ymm7)
	
        vbroadcastsd(mem(rax, 20*8), ymm2)
        vbroadcastsd(mem(rax, 21*8), ymm3)
        vfmadd231pd(ymm0, ymm2, ymm8)
        vfmadd231pd(ymm1, ymm2, ymm9)
        vfmadd231pd(ymm0, ymm3, ymm10)
        vfmadd231pd(ymm1, ymm3, ymm11)
	

        vbroadcastsd(mem(rax, 22*8), ymm2)
        vbroadcastsd(mem(rax, 23*8), ymm3)
        vfmadd231pd(ymm0, ymm2, ymm12)
        vfmadd231pd(ymm1, ymm2, ymm13)
        vfmadd231pd(ymm0, ymm3, ymm14)
        vfmadd231pd(ymm1, ymm3, ymm15)
	
        add(imm(4*6*8), rax) // a += 4*6 (unroll x mr)
        add(imm(4*8*8), rbx) // b += 4*8 (unroll x nr)
	
        vmovapd(mem(rbx, -4*32), ymm0)
        vmovapd(mem(rbx, -3*32), ymm1)
	
	
        dec(rsi) // i -= 1;
        jne(.DLOOPKITER) // iterate again if i != 0.


        label(.DCONSIDKLEFT)

        mov(var(k_left), rsi) // i = k_left;
        test(rsi, rsi) // check i via logical AND.
        je(.DPOSTACCUM) // if i == 0, we're done; jump to end.
        // else, we prepare to enter k_left loop.
	
	
        label(.DLOOPKLEFT) // EDGE LOOP

        prefetch(0, mem(rax, 64*8))
	
        vbroadcastsd(mem(rax, 0*8), ymm2)
        vbroadcastsd(mem(rax, 1*8), ymm3)
        vfmadd231pd(ymm0, ymm2, ymm4)
        vfmadd231pd(ymm1, ymm2, ymm5)
        vfmadd231pd(ymm0, ymm3, ymm6)
        vfmadd231pd(ymm1, ymm3, ymm7)
	
        vbroadcastsd(mem(rax, 2*8), ymm2)
        vbroadcastsd(mem(rax, 3*8), ymm3)
        vfmadd231pd(ymm0, ymm2, ymm8)
        vfmadd231pd(ymm1, ymm2, ymm9)
        vfmadd231pd(ymm0, ymm3, ymm10)
        vfmadd231pd(ymm1, ymm3, ymm11)
	
        vbroadcastsd(mem(rax, 4*8), ymm2)
        vbroadcastsd(mem(rax, 5*8), ymm3)
        vfmadd231pd(ymm0, ymm2, ymm12)
        vfmadd231pd(ymm1, ymm2, ymm13)
        vfmadd231pd(ymm0, ymm3, ymm14)
        vfmadd231pd(ymm1, ymm3, ymm15)
	
        add(imm(1*6*8), rax) // a += 1*6 (unroll x mr)
        add(imm(1*8*8), rbx) // b += 1*8 (unroll x nr)
	
        vmovapd(mem(rbx, -4*32), ymm0)
        vmovapd(mem(rbx, -3*32), ymm1)


        dec(rsi) // i -= 1;
        jne(.DLOOPKLEFT) // iterate again if i != 0.
	


	
        label(.DPOSTACCUM)


	
	
        mov(var(alpha), rax) // load address of alpha
        mov(var(beta), rbx) // load address of beta
        vbroadcastsd(mem(rax), ymm0) // load alpha and duplicate
        vbroadcastsd(mem(rbx), ymm3) // load beta and duplicate
	
        vmulpd(ymm0, ymm4, ymm4) // scale by alpha
        vmulpd(ymm0, ymm5, ymm5)
        vmulpd(ymm0, ymm6, ymm6)
        vmulpd(ymm0, ymm7, ymm7)
        vmulpd(ymm0, ymm8, ymm8)
        vmulpd(ymm0, ymm9, ymm9)
        vmulpd(ymm0, ymm10, ymm10)
        vmulpd(ymm0, ymm11, ymm11)
        vmulpd(ymm0, ymm12, ymm12)
        vmulpd(ymm0, ymm13, ymm13)
        vmulpd(ymm0, ymm14, ymm14)
        vmulpd(ymm0, ymm15, ymm15)
	

        vxorpd(ymm0, ymm0, ymm0) // set ymm0 to zero.
        vucomisd(xmm0, xmm3) // set ZF if beta == 0.
        je(.DBETAZERO) // if ZF = 1, jump to beta == 0 case

/* #define DGEMM_INPUT_GS_BETA_NZ \ */
/* 	vmovlpd(mem(rcx), xmm0, xmm0) \ */
/* 	vmovhpd(mem(rcx, rsi, 1), xmm0, xmm0) \ */
/* 	vmovlpd(mem(rcx, rsi, 2), xmm1, xmm1) \ */
/* 	vmovhpd(mem(rcx, r13, 1), xmm1, xmm1) \ */
/* 	vperm2f128(imm(0x20), ymm1, ymm0, ymm0) /\*\ */


    /*     #define DGEMM_OUTPUT_GS_BETA_NZ \ */
	/* vextractf128(imm(1), ymm0, xmm1) \ */
	/* vmovlpd(xmm0, mem(rcx)) \ */
	/* vmovhpd(xmm0, mem(rcx, rsi, 1)) \ */
	/* vmovlpd(xmm1, mem(rcx, rsi, 2)) \ */
	/* vmovhpd(xmm1, mem(rcx, r13, 1)) /\*\ */




        mov(mem(rdi,0*8), rax)// [address row_sct+0*8] -> rsp        

        DGEMM_INPUT_BETA_NZ_SCATTER_03
        vfmadd213pd(ymm4, ymm3, ymm0)
        DGEMM_OUTPUT_BETA_NZ_SCATTER

        
        
        DGEMM_INPUT_BETA_NZ_SCATTER_47
        vfmadd213pd(ymm5, ymm3, ymm0)
        DGEMM_OUTPUT_BETA_NZ_SCATTER

        
        mov(mem(rdi,1*8), rax)// [address row_sct+0*8] -> rax        
        
        DGEMM_INPUT_BETA_NZ_SCATTER_03
        vfmadd213pd(ymm6, ymm3, ymm0)
        DGEMM_OUTPUT_BETA_NZ_SCATTER

        DGEMM_INPUT_BETA_NZ_SCATTER_47
        vfmadd213pd(ymm7, ymm3, ymm0)
        DGEMM_OUTPUT_BETA_NZ_SCATTER

        mov(mem(rdi,2*8), rax)// [address row_sct+0*8] -> rax        
        
        DGEMM_INPUT_BETA_NZ_SCATTER_03
        vfmadd213pd(ymm8, ymm3, ymm0)
        DGEMM_OUTPUT_BETA_NZ_SCATTER

        DGEMM_INPUT_BETA_NZ_SCATTER_47
        vfmadd213pd(ymm9, ymm3, ymm0)
        DGEMM_OUTPUT_BETA_NZ_SCATTER

        mov(mem(rdi,3*8), rax)// [address row_sct+0*8] -> rax        
        
        DGEMM_INPUT_BETA_NZ_SCATTER_03
        vfmadd213pd(ymm10, ymm3, ymm0)
        DGEMM_OUTPUT_BETA_NZ_SCATTER

        DGEMM_INPUT_BETA_NZ_SCATTER_47
        vfmadd213pd(ymm11, ymm3, ymm0)
        DGEMM_OUTPUT_BETA_NZ_SCATTER

        mov(mem(rdi,4*8), rax)// [address row_sct+0*8] -> rax        
        
        DGEMM_INPUT_BETA_NZ_SCATTER_03
        vfmadd213pd(ymm12, ymm3, ymm0)
        DGEMM_OUTPUT_BETA_NZ_SCATTER

        DGEMM_INPUT_BETA_NZ_SCATTER_47
        vfmadd213pd(ymm13, ymm3, ymm0)
        DGEMM_OUTPUT_BETA_NZ_SCATTER
        
        mov(mem(rdi,5*8), rax)// [address row_sct+0*8] -> rax        
        
        DGEMM_INPUT_BETA_NZ_SCATTER_03
        vfmadd213pd(ymm14, ymm3, ymm0)
        DGEMM_OUTPUT_BETA_NZ_SCATTER

        DGEMM_INPUT_BETA_NZ_SCATTER_47
        vfmadd213pd(ymm15, ymm3, ymm0)
        DGEMM_OUTPUT_BETA_NZ_SCATTER

        label(.DBETAZERO)
        jmp(.DDONE) // jump to end.       
        
        
	label(.DDONE)
    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter), // 0
      [k_left] "m" (k_left), // 1
      [a]      "m" (a),      // 2
      [b]      "m" (b),      // 3
      [alpha]  "m" (alpha),  // 4
      [beta]   "m" (beta),   // 5
      [c]      "m" (c),      // 6
      [rsct_c]   "m" (rsct_c),   // 7
      [csct_c]   "m" (csct_c)/*,   // 8
      [b_next] "m" (b_next), // 9
 p     [a_next] "m" (a_next)*/  // 10
	: // register clobber list
    "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	)        
}

#define DGEMM_SCT_PREFETCH_1VECROW              \
    

void sct_dgemm_haswell_asm_6x8_8vec
(
unsigned int      k0,
    double*    restrict alpha,
    double*    restrict a,
    double*    restrict b,
    double*    restrict beta,
    double*    restrict c,
    unsigned long long*  rsct_c   // 6 int
    )
{
    unsigned long long k_iter = k0/4;
    unsigned     long long  k_left = k0%4;

    begin_asm()
        vzeroall()


        mov(var(b), rbx) // load b address -> rbx

        add(imm(32*4), rbx)
        // initialize loop by pre-loading
        vmovapd(mem(rbx, -4*32), ymm0)
        vmovapd(mem(rbx, -3*32), ymm1)

        mov(var(c), rcx) // address c -> rcx
        mov(var(rsct_c), rdi)// address row_sct -> rdi

        //row 0
        mov(mem(rdi), rax)// [address row_sct+0*8] -> ax
        prefetch(0, mem(rcx, rax, 8))
        prefetch(0, mem(rcx, rax, 8,7*8))
        //row 1 
        mov(mem(rdi,1*8), rax)// [address row_sct+0*8] -> rsp
        prefetch(0, mem(rcx, rax, 8))
        prefetch(0, mem(rcx, rax, 8,7*8))

        mov(mem(rdi,2*8), rax)// [address row_sct+0*8] -> rsp
        prefetch(0, mem(rcx, rax, 8))
        prefetch(0, mem(rcx, rax, 8,7*8))

        mov(mem(rdi,3*8), rax)// [address row_sct+0*8] -> rsp
        prefetch(0, mem(rcx, rax, 8))
        prefetch(0, mem(rcx, rax, 8,7*8))

        mov(mem(rdi,4*8), rax)// [address row_sct+0*8] -> rsp
        prefetch(0, mem(rcx, rax, 8))
        prefetch(0, mem(rcx, rax, 8,7*8))

        mov(mem(rdi,5*8), rax)// [address row_sct+0*8] -> rsp
        prefetch(0, mem(rcx, rax, 8))
        prefetch(0, mem(rcx, rax, 8,7*8))


                mov(var(a), rax) // load a address -> rax
        
        mov(var(k_iter), rsi) // i = k_iter;
        test(rsi, rsi) // check i via logical AND.
        je(.DCONSIDKLEFT) // if i == 0, jump to code that
	 // contains the k_left loop.
	

        
        label(.DLOOPKITER) // MAIN LOOP

        // iteration 0
        prefetch(0, mem(rax, 64*8))


        vbroadcastsd(mem(rax, 0*8), ymm2)
        vbroadcastsd(mem(rax, 1*8), ymm3)
        vfmadd231pd(ymm0, ymm2, ymm4)
        vfmadd231pd(ymm1, ymm2, ymm5)
        vfmadd231pd(ymm0, ymm3, ymm6)
        vfmadd231pd(ymm1, ymm3, ymm7)
	
        vbroadcastsd(mem(rax, 2*8), ymm2)
        vbroadcastsd(mem(rax, 3*8), ymm3)
        vfmadd231pd(ymm0, ymm2, ymm8)
        vfmadd231pd(ymm1, ymm2, ymm9)
        vfmadd231pd(ymm0, ymm3, ymm10)
        vfmadd231pd(ymm1, ymm3, ymm11)
	
        vbroadcastsd(mem(rax, 4*8), ymm2)
        vbroadcastsd(mem(rax, 5*8), ymm3)
        vfmadd231pd(ymm0, ymm2, ymm12)
        vfmadd231pd(ymm1, ymm2, ymm13)
        vfmadd231pd(ymm0, ymm3, ymm14)
        vfmadd231pd(ymm1, ymm3, ymm15)
	
        vmovapd(mem(rbx, -2*32), ymm0)
        vmovapd(mem(rbx, -1*32), ymm1)
	
        // iteration 1
        vbroadcastsd(mem(rax, 6*8), ymm2)
        vbroadcastsd(mem(rax, 7*8), ymm3)
        vfmadd231pd(ymm0, ymm2, ymm4)
        vfmadd231pd(ymm1, ymm2, ymm5)
        vfmadd231pd(ymm0, ymm3, ymm6)
        vfmadd231pd(ymm1, ymm3, ymm7)
	
        vbroadcastsd(mem(rax, 8*8), ymm2)
        vbroadcastsd(mem(rax, 9*8), ymm3)
        vfmadd231pd(ymm0, ymm2, ymm8)
        vfmadd231pd(ymm1, ymm2, ymm9)
        vfmadd231pd(ymm0, ymm3, ymm10)
        vfmadd231pd(ymm1, ymm3, ymm11)
	
        vbroadcastsd(mem(rax, 10*8), ymm2)
        vbroadcastsd(mem(rax, 11*8), ymm3)
        vfmadd231pd(ymm0, ymm2, ymm12)
        vfmadd231pd(ymm1, ymm2, ymm13)
        vfmadd231pd(ymm0, ymm3, ymm14)
        vfmadd231pd(ymm1, ymm3, ymm15)
	
        vmovapd(mem(rbx, 0*32), ymm0)
        vmovapd(mem(rbx, 1*32), ymm1)
	
        // iteration 2
        prefetch(0, mem(rax, 76*8))
	
        vbroadcastsd(mem(rax, 12*8), ymm2)
        vbroadcastsd(mem(rax, 13*8), ymm3)
        vfmadd231pd(ymm0, ymm2, ymm4)
        vfmadd231pd(ymm1, ymm2, ymm5)
        vfmadd231pd(ymm0, ymm3, ymm6)
        vfmadd231pd(ymm1, ymm3, ymm7)
	
        vbroadcastsd(mem(rax, 14*8), ymm2)
        vbroadcastsd(mem(rax, 15*8), ymm3)
        vfmadd231pd(ymm0, ymm2, ymm8)
        vfmadd231pd(ymm1, ymm2, ymm9)
        vfmadd231pd(ymm0, ymm3, ymm10)
        vfmadd231pd(ymm1, ymm3, ymm11)
	
        vbroadcastsd(mem(rax, 16*8), ymm2)
        vbroadcastsd(mem(rax, 17*8), ymm3)
        vfmadd231pd(ymm0, ymm2, ymm12)
        vfmadd231pd(ymm1, ymm2, ymm13)
        vfmadd231pd(ymm0, ymm3, ymm14)
        vfmadd231pd(ymm1, ymm3, ymm15)
	
        vmovapd(mem(rbx, 2*32), ymm0)
        vmovapd(mem(rbx, 3*32), ymm1)
	
        // iteration 3
        vbroadcastsd(mem(rax, 18*8), ymm2)
        vbroadcastsd(mem(rax, 19*8), ymm3)
        vfmadd231pd(ymm0, ymm2, ymm4)
        vfmadd231pd(ymm1, ymm2, ymm5)
        vfmadd231pd(ymm0, ymm3, ymm6)
        vfmadd231pd(ymm1, ymm3, ymm7)
	
        vbroadcastsd(mem(rax, 20*8), ymm2)
        vbroadcastsd(mem(rax, 21*8), ymm3)
        vfmadd231pd(ymm0, ymm2, ymm8)
        vfmadd231pd(ymm1, ymm2, ymm9)
        vfmadd231pd(ymm0, ymm3, ymm10)
        vfmadd231pd(ymm1, ymm3, ymm11)
	

        vbroadcastsd(mem(rax, 22*8), ymm2)
        vbroadcastsd(mem(rax, 23*8), ymm3)
        vfmadd231pd(ymm0, ymm2, ymm12)
        vfmadd231pd(ymm1, ymm2, ymm13)
        vfmadd231pd(ymm0, ymm3, ymm14)
        vfmadd231pd(ymm1, ymm3, ymm15)
	
        add(imm(4*6*8), rax) // a += 4*6 (unroll x mr)
        add(imm(4*8*8), rbx) // b += 4*8 (unroll x nr)
	
        vmovapd(mem(rbx, -4*32), ymm0)
        vmovapd(mem(rbx, -3*32), ymm1)
	
	
        dec(rsi) // i -= 1;
        jne(.DLOOPKITER) // iterate again if i != 0.


        label(.DCONSIDKLEFT)

        mov(var(k_left), rsi) // i = k_left;
        test(rsi, rsi) // check i via logical AND.
        je(.DPOSTACCUM) // if i == 0, we're done; jump to end.
        // else, we prepare to enter k_left loop.
	
	
        label(.DLOOPKLEFT) // EDGE LOOP

        prefetch(0, mem(rax, 64*8))
	
        vbroadcastsd(mem(rax, 0*8), ymm2)
        vbroadcastsd(mem(rax, 1*8), ymm3)
        vfmadd231pd(ymm0, ymm2, ymm4)
        vfmadd231pd(ymm1, ymm2, ymm5)
        vfmadd231pd(ymm0, ymm3, ymm6)
        vfmadd231pd(ymm1, ymm3, ymm7)
	
        vbroadcastsd(mem(rax, 2*8), ymm2)
        vbroadcastsd(mem(rax, 3*8), ymm3)
        vfmadd231pd(ymm0, ymm2, ymm8)
        vfmadd231pd(ymm1, ymm2, ymm9)
        vfmadd231pd(ymm0, ymm3, ymm10)
        vfmadd231pd(ymm1, ymm3, ymm11)
	
        vbroadcastsd(mem(rax, 4*8), ymm2)
        vbroadcastsd(mem(rax, 5*8), ymm3)
        vfmadd231pd(ymm0, ymm2, ymm12)
        vfmadd231pd(ymm1, ymm2, ymm13)
        vfmadd231pd(ymm0, ymm3, ymm14)
        vfmadd231pd(ymm1, ymm3, ymm15)
	
        add(imm(1*6*8), rax) // a += 1*6 (unroll x mr)
        add(imm(1*8*8), rbx) // b += 1*8 (unroll x nr)
	
        vmovapd(mem(rbx, -4*32), ymm0)
        vmovapd(mem(rbx, -3*32), ymm1)


        dec(rsi) // i -= 1;
        jne(.DLOOPKLEFT) // iterate again if i != 0.
	


	
        label(.DPOSTACCUM)


	
	
        mov(var(alpha), rax) // load address of alpha
        mov(var(beta), rbx) // load address of beta
        vbroadcastsd(mem(rax), ymm0) // load alpha and duplicate
        vbroadcastsd(mem(rbx), ymm3) // load beta and duplicate
	
        vmulpd(ymm0, ymm4, ymm4) // scale by alpha
        vmulpd(ymm0, ymm5, ymm5)
        vmulpd(ymm0, ymm6, ymm6)
        vmulpd(ymm0, ymm7, ymm7)
        vmulpd(ymm0, ymm8, ymm8)
        vmulpd(ymm0, ymm9, ymm9)
        vmulpd(ymm0, ymm10, ymm10)
        vmulpd(ymm0, ymm11, ymm11)
        vmulpd(ymm0, ymm12, ymm12)
        vmulpd(ymm0, ymm13, ymm13)
        vmulpd(ymm0, ymm14, ymm14)
        vmulpd(ymm0, ymm15, ymm15)
	

        vxorpd(ymm0, ymm0, ymm0) // set ymm0 to zero.
        vucomisd(xmm0, xmm3) // set ZF if beta == 0.
        je(.DBETAZERO) // if ZF = 1, jump to beta == 0 case

        mov(mem(rdi,0*8), rax)// [address row_sct+0*8] -> rsp

        vfmadd231pd(mem(rcx, rax, 8), ymm3, ymm4)
        vmovupd(ymm4, mem(rcx, rax, 8))

        vfmadd231pd(mem(rcx, rax, 8, 8*4), ymm3, ymm5)
        vmovupd(ymm5, mem(rcx, rax, 8, 8*4))

        mov(mem(rdi,1*8), rbx)// [address row_sct+0*8] -> rsp

        vfmadd231pd(mem(rcx, rbx, 8), ymm3, ymm6)
        vmovupd(ymm6, mem(rcx, rbx, 8))

        vfmadd231pd(mem(rcx, rbx, 8, 8*4), ymm3, ymm7)
        vmovupd(ymm7, mem(rcx, rbx, 8, 8*4))


        mov(mem(rdi,2*8), rax)// [address row_sct+0*8] -> rsp

        vfmadd231pd(mem(rcx, rax, 8), ymm3, ymm8)
        vmovupd(ymm8, mem(rcx, rax, 8))

        vfmadd231pd(mem(rcx, rax, 8, 8*4), ymm3, ymm9)
        vmovupd(ymm9, mem(rcx, rax, 8, 8*4))

        mov(mem(rdi,3*8), rbx)// [address row_sct+0*8] -> rsp

        vfmadd231pd(mem(rcx, rbx, 8), ymm3, ymm10)
        vmovupd(ymm10, mem(rcx, rbx, 8))

        vfmadd231pd(mem(rcx, rbx, 8, 8*4), ymm3, ymm11)
        vmovupd(ymm11, mem(rcx, rbx, 8, 8*4))

        
        mov(mem(rdi,4*8), rax)// [address row_sct+0*8] -> rsp

        vfmadd231pd(mem(rcx, rax, 8), ymm3, ymm12)
        vmovupd(ymm12, mem(rcx, rax, 8))

        vfmadd231pd(mem(rcx, rax, 8, 8*4), ymm3, ymm13)
        vmovupd(ymm13, mem(rcx, rax, 8, 8*4))

        mov(mem(rdi,5*8), rbx)// [address row_sct+0*8] -> rsp

        vfmadd231pd(mem(rcx, rbx, 8), ymm3, ymm14)
        vmovupd(ymm14, mem(rcx, rbx, 8))

        vfmadd231pd(mem(rcx, rbx, 8, 8*4), ymm3, ymm15)
        vmovupd(ymm15, mem(rcx, rbx, 8, 8*4))
        jmp(.DDONE) // jump to end.
                label(.DBETAZERO)
 	label(.DDONE)       
        end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter), // 0
      [k_left] "m" (k_left), // 1
      [a]      "m" (a),      // 2
      [b]      "m" (b),      // 3
      [alpha]  "m" (alpha),  // 4
      [beta]   "m" (beta),   // 5
      [c]      "m" (c),      // 6
    [rsct_c]   "m" (rsct_c)
       // 7
    //  [csct_c]   "m" (csct_c)/*,   // 8

	: // register clobber list
    "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	)        
}
#endif