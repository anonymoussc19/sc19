#include "ws_mapping_avx.h"
#include "../../tensor_kernels.h"
#include "../../scatter_kernels.h"
#include<omp.h>
#include"mpi.h"
int B_range_a = 48;
int A_range_b = 24;
int B_range_c = 32;
int B_range_d = 32;
int B_range_e = 48;

int A_range_f = 32;
int B_range_f = 32;
int C_range_f = 32;

int C_range_a = 48;
int C_range_b = 24;
int C_range_c = 32;
int C_range_d = 32;
int C_range_e = 48;


void compare_fun(double* A, double*B,double*C ){
    for(int a=0; a<C_range_a; a++)
    for(int b=0; b<C_range_b; b++)
    for(int c=0; c<C_range_c; c++)
    for(int d=0; d<C_range_d; d++)
    for(int e=0; e<C_range_e; e++)
    for(int f=0; f<C_range_f; f++)
    C[+a* C_range_b* C_range_c* C_range_d* C_range_e+b* C_range_c* C_range_d* C_range_e+c* C_range_d* C_range_e+d* C_range_e+e] += 
        A[+b* C_range_f+f] * 
        B[+e* C_range_f* C_range_c* C_range_a* C_range_d+f* C_range_c* C_range_a* C_range_d+c* C_range_a* C_range_d+a* C_range_d+d]; 
}

int main(int argc, char** argv){

    int Tk4, Tk3, Tk2, Tk1, Tk0;
    int Tm4, Tm3, Tm2, Tm1, Tm0;
    int Tn4, Tn3, Tn2, Tn1, Tn0;


    Tk4 = C_range_f;
    Tm4 = A_range_b;
    Tn4 = B_range_a * B_range_c * B_range_d*B_range_e;

    Tk0 = 1; Tm0 =6; Tn0=8;
    Tn3 = 512;
//    Tn3 = 20000/48*48;
    Tm3 = 24;
        
    Tn2 = 512;
    Tm2 = 6 ;
    
    Tn1 = 8;
    Tm1 =6;
    Tk1 = Tk2 = Tk3  = 32;
    printf("abcde-efcad-bf\n");
    double *A, *B, *C, *Abuf, *Bbuf;
    C = (double*)memalign(4096, sizeof(double) * C_range_b * C_range_c * C_range_a*C_range_d*C_range_e);
    double *C1 = (double*)memalign(4096, sizeof(double) * C_range_b * C_range_c * C_range_a*C_range_d*C_range_e);


    A = (double*)memalign(4096, sizeof(double) * A_range_b * A_range_f);
    B = (double*)memalign(4096, sizeof(double) * B_range_e * B_range_f * B_range_c * B_range_a * B_range_d);
    for(int i = 0; i < 1*A_range_b*A_range_f; i++){
        A[i] = rand()%1000;
    }
    for(int i = 0; i < 1*B_range_e*B_range_f*B_range_c*B_range_a*B_range_d; i++){
        B[i] = rand()%1000;
    }
    for(int i = 0; i < 1*C_range_a*C_range_b*C_range_c*C_range_d*C_range_e; i++){
        C1[i] = C[i] = rand()%1000;
    }
//====

    double alpha = 1.0;
    double beta = 1.0;
    static tsc_counter a,b;
    CPUID(); RDTSC(a); CPUID(); RDTSC(b);
    /* warm up */
    CPUID(); RDTSC(a); CPUID(); RDTSC(b);
    /* warm up */
    CPUID(); RDTSC(a); CPUID(); RDTSC(b);

    int tot_runtime = 5;
;MPI_Init(NULL,NULL);
MPI_Barrier(MPI_COMM_WORLD);
    RDTSC(a);
    double start = omp_get_wtime();
    for(int ccnt = 0; ccnt < tot_runtime; ccnt++){


//Abuf = (double*)memalign(4096, sizeof(double) * A_range_b * A_range_f);
        Abuf = (double*)memalign(4096, sizeof(double) * Tm3*Tk3);
        int ext_range_b = A_range_b % Tm0;
        int base_range_b = A_range_b - ext_range_b;
        int base_m_range = base_range_b;
        unsigned long long *Ascatter = (unsigned long long*)memalign(4096, sizeof(unsigned long long)*A_range_b);

        int A_stride_b = 1 * A_range_f;
        int A_stride_f = 1;

        int Fused_A_stride_b = 1;
        int Ext_A_stride_b = 1;

        for(int Im=0; Im < base_m_range;Im++){
            int Ib = Im / Fused_A_stride_b;
            Ascatter[Im] = +Ib*C_range_c*C_range_d*C_range_e;
        }
        for(int Im=base_m_range; Im < Tm4;Im++){
            int Ib = (Im - base_m_range) / Ext_A_stride_b;
            Ascatter[Im] = +Ib*C_range_c*C_range_d*C_range_e;
        }

//Bbuf = (double*)memalign(4096, sizeof(double) * B_range_e * B_range_f * B_range_c * B_range_a * B_range_d);
//Bbuf = (double*)memalign(4096, sizeof(double) * Tk3*Tn3);
        Bbuf = (double*)memalign(4096, sizeof(double) * Tk3*(Tn3+8*48-1)/(8*48)*(8*48));

        int ext_range_e = B_range_e % Tn0;
        int base_range_e = B_range_e - ext_range_e;
        int base_n_range = base_range_e * B_range_d * B_range_c * B_range_a;
        unsigned long long *Bscatter = (unsigned long long*)memalign(4096, sizeof(unsigned long long)*B_range_a*B_range_c*B_range_d*B_range_e);

        int B_stride_e = 1 * B_range_f * B_range_c * B_range_a * B_range_d;
        int B_stride_f = 1 * B_range_c * B_range_a * B_range_d;
        int B_stride_c = 1 * B_range_a * B_range_d;
        int B_stride_a = 1 * B_range_d;
        int B_stride_d = 1;

        int Fused_B_stride_a = 1 * B_range_c * B_range_d * base_range_e;
        int Fused_B_stride_c = 1 * B_range_d * base_range_e;
        int Fused_B_stride_d = 1 * base_range_e;
        int Fused_B_stride_e = 1;
        int Ext_B_stride_a = 1 * B_range_c * B_range_d * ext_range_e;
        int Ext_B_stride_c = 1 * B_range_d * ext_range_e;
        int Ext_B_stride_d = 1 * ext_range_e;
        int Ext_B_stride_e = 1;

        for(int In=0; In < base_n_range;In++){
            int Ia = In / Fused_B_stride_a;
            int Ic = In % Fused_B_stride_a / Fused_B_stride_c;
            int Id = In % Fused_B_stride_c / Fused_B_stride_d;
            int Ie = In % Fused_B_stride_d;
            Bscatter[In] = +Ia*C_range_b*C_range_c*C_range_d*C_range_e+Ic*C_range_d*C_range_e+Id*C_range_e+Ie;
        }
        for(int In=base_n_range; In < Tn4;In++){
            int Ia = (In - base_n_range) / Ext_B_stride_a;
            int Ic = (In - base_n_range)  % Ext_B_stride_a / Ext_B_stride_c;
            int Id = (In - base_n_range)  % Ext_B_stride_c / Ext_B_stride_d;
            int Ie = (In - base_n_range) % Ext_B_stride_d+ base_range_e;
            Bscatter[In] = +Ia*C_range_b*C_range_c*C_range_d*C_range_e+Ic*C_range_d*C_range_e+Id*C_range_e+Ie;
        }

/* for(int k4 = 0; k4 < Tk4; k4 += Tk3) */
/* { */
/* for(int m4 = 0; m4 < Tm4; m4 += Tm3) */
/* { */
/* for(int k3 = 0; k3 < Tk3; k3 += Tk2) */
/* { */
/* if(+k3+k4>=Tk4)break; */
/* for(int m3 = 0; m3 < Tm3; m3 += Tm2) */
/* { */
/* if(+m3+m4>=Tm4)break; */
/* for(int k2 = 0; k2 < Tk2; k2 += Tk1) */
/* { */
/* if(+k2+k3+k4>=Tk4)break; */
/* for(int m2 = 0; m2 < Tm2; m2 += Tm1) */
/* { */
/* if(+m2+m3+m4>=Tm4)break; */
/* for(int m1 = 0; m1 < Tm1; m1 += Tm0) */
/* { */
/* if(+m1+m2+m3+m4>=Tm4)break; */
/* for(int k1 = 0; k1 < Tk1; k1 += Tk0) */
/* { */
/* if(+k1+k2+k3+k4>=Tk4)break; */
/* for(int m0 = 0; m0 < Tm0; m0 += 1) */
/* { */
/* if(+m0+m1+m2+m3+m4>=Tm4)break; */
/*  int Im = m0+m1+m2+m3+m4; int If =  k1+k2+k3+k4; */
/* // begin pack body */
/*  int A_pack_offset =  + m0 + k1* Tm0 + m1* MIN(Tk1, Tk4-k2-k3-k4) + m2* MIN(Tk1, Tk4-k2-k3-k4) + k2* MIN(Tm2, Tm4-m3-m4) + m3* MIN(Tk2, Tk4-k3-k4) + k3* MIN(Tm3, Tm4-m4);// + m4* MIN(Tk3, Tk4-k4) + k4* MIN(Tm4, Tm4); */
/* if(Im< base_m_range){ */
/* int Ib = Im / Fused_A_stride_b; */
/* Abuf[A_pack_offset] = A[ +Ib*A_stride_b +If*A_stride_f]; */
/* } */
/* else{ */
/* int Ib = (Im - base_m_range) / Ext_A_stride_b; */
/* Abuf[A_pack_offset] = A[ +Ib*A_stride_b +If*A_stride_f]; */
/* } */
/* // end pack body */
/* } */
/* } */
/* } */
/* } */
/* } */
/* } */
/* } */
/* } */
/* } */

/* for(int k4 = 0; k4 < Tk4; k4 += Tk3) */
/* { */
/* for(int n4 = 0; n4 < Tn4; n4 += Tn3) */
/* { */
/* for(int k3 = 0; k3 < Tk3; k3 += Tk2) */
/* { */
/* if(+k3+k4>=Tk4)break; */
/* for(int n3 = 0; n3 < Tn3; n3 += Tn2) */
/* { */
/* if(+n3+n4>=Tn4)break; */
/* for(int k2 = 0; k2 < Tk2; k2 += Tk1) */
/* { */
/* if(+k2+k3+k4>=Tk4)break; */
/* for(int n2 = 0; n2 < Tn2; n2 += Tn1) */
/* { */
/* if(+n2+n3+n4>=Tn4)break; */
/* for(int n1 = 0; n1 < Tn1; n1 += Tn0) */
/* { */
/* if(+n1+n2+n3+n4>=Tn4)break; */
/* for(int k1 = 0; k1 < Tk1; k1 += Tk0) */
/* { */
/* if(+k1+k2+k3+k4>=Tk4)break; */
/* //for(int n0 = 0; n0 < Tn0; n0 += 1) */
/* { */
/* //if(+n0+n1+n2+n3+n4>=Tn4)break; */
/*     int In =  n1+n2+n3+n4; int If = k1+k2+k3+k4; */
/* // begin pack body */
/*     int B_pack_offset = // + n0 */
/*         + k1* Tn0 + n1* MIN(Tk1, Tk4-k2-k3-k4) + n2* MIN(Tk1, Tk4-k2-k3-k4) + k2* MIN(Tn2, Tn4-n3-n4) + n3* MIN(Tk2, Tk4-k3-k4) + k3* MIN(Tn3, Tn4-n4);// + n4* MIN(Tk3, Tk4-k4) + k4* MIN(Tn4, Tn4); */
/* if(In+8<= base_n_range){ */
/* int Ia = In / Fused_B_stride_a; */
/* int Ic = In % Fused_B_stride_a / Fused_B_stride_c; */
/* int Id = In % Fused_B_stride_c / Fused_B_stride_d; */
/* int Ie = In % Fused_B_stride_d; */
/* for(int jj = 0; jj< 8;jj++) */
/* Bbuf[B_pack_offset] = B[ +(Ie+jj)*B_stride_e +If*B_stride_f +Ic*B_stride_c +Ia*B_stride_a +Id*B_stride_d]; */
/* } */
/* else{ */
/* int Ia = (In - base_n_range) / Ext_B_stride_a; */
/* int Ic = (In - base_n_range)  % Ext_B_stride_a / Ext_B_stride_c; */
/* int Id = (In - base_n_range)  % Ext_B_stride_c / Ext_B_stride_d; */
/* int Ie = (In - base_n_range) % Ext_B_stride_d+ base_range_e; */
/* Bbuf[B_pack_offset] = B[ +Ie*B_stride_e +If*B_stride_f +Ic*B_stride_c +Ia*B_stride_a +Id*B_stride_d]; */
/* } */
/* // end pack body */
/* } */
/* } */
/* } */
/* } */
/* } */
/* } */
/* } */
/* } */
/* } */

        for(int k4 = 0; k4 < Tk4; k4 += Tk3)
        {
            for(int m4 = 0; m4 < Tm4; m4 += Tm3)
            {
                for(int k3 = 0; k3 < Tk3; k3 += Tk2)
                {
                    if(+k3+k4>=Tk4)break;
                    for(int m3 = 0; m3 < Tm3; m3 += Tm2)
                    {
                        if(+m3+m4>=Tm4)break;
                        for(int k2 = 0; k2 < Tk2; k2 += Tk1)
                        {
                            if(+k2+k3+k4>=Tk4)break;
                            for(int m2 = 0; m2 < Tm2; m2 += Tm1)
                            {
                                if(+m2+m3+m4>=Tm4)break;
                                for(int m1 = 0; m1 < Tm1; m1 += Tm0)
                                {
                                    if(+m1+m2+m3+m4>=Tm4)break;
                                    for(int k1 = 0; k1 < Tk1; k1 += Tk0)
                                    {
                                        if(+k1+k2+k3+k4>=Tk4)break;
                                        for(int m0 = 0; m0 < Tm0; m0 += 1)
                                        {
                                            if(+m0+m1+m2+m3+m4>=Tm4)break;
                                            int Im = m0+m1+m2+m3+m4; int If =  k1+k2+k3+k4;
// begin pack body
                                            int A_pack_offset =  + m0 + k1* Tm0 + m1* MIN(Tk1, Tk4-k2-k3-k4) + m2* MIN(Tk1, Tk4-k2-k3-k4) + k2* MIN(Tm2, Tm4-m3-m4) + m3* MIN(Tk2, Tk4-k3-k4) + k3* MIN(Tm3, Tm4-m4);// + m4* MIN(Tk3, Tk4-k4) + k4* MIN(Tm4, Tm4);
                                            if(Im< base_m_range){
                                                int Ib = Im / Fused_A_stride_b;
                                                Abuf[A_pack_offset] = A[ +Ib*A_stride_b +If*A_stride_f];
                                            }
                                            else{
                                                int Ib = (Im - base_m_range) / Ext_A_stride_b;
                                                Abuf[A_pack_offset] = A[ +Ib*A_stride_b +If*A_stride_f];
                                            }
// end pack body
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

    
                for(int n4 = 0; n4 < Tn4; n4 += Tn3)
                {

                    for(int k3 = 0; k3 < Tk3; k3 += Tk2)
                    {
                        if(+k3+k4>=Tk4)break;
                        for(int n3 = 0; n3 < Tn3; n3 += Tn2)
                        {
                            if(+n3+n4>=Tn4)break;
                            for(int k2 = 0; k2 < Tk2; k2 += Tk1)
                            {
                                if(+k2+k3+k4>=Tk4)break;
                                for(int n2 = 0; n2 < Tn2; n2 += Tn1)
                                {
                                    if(+n2+n3+n4>=Tn4)break;
                                    for(int n1 = 0; n1 < Tn1; n1 += Tn0)
                                    {
                                        if(+n1+n2+n3+n4>=Tn4)break;
                                        int In =  n1+n2+n3+n4; 
                                        int Ia = In / Fused_B_stride_a;
                                        int Ic = In % Fused_B_stride_a / Fused_B_stride_c;
                                        int Id = In % Fused_B_stride_c / Fused_B_stride_d;
                                        int Ie = In % Fused_B_stride_d;
                                        if(In+8<= base_n_range){
                                            int add_Id_pckoff[8];
                                            for(int iid = 1; iid < 8;  iid++){
                                                add_Id_pckoff[iid] =  (iid*Fused_B_stride_d %Tn1)* MIN(Tk1, Tk4-k2-k3-k4)
                                                    + (iid*Fused_B_stride_d %Tn2 -iid* Fused_B_stride_d %Tn1) * MIN(Tk1, Tk4-k2-k3-k4)
                                                    + (iid*Fused_B_stride_d %Tn3 -iid* Fused_B_stride_d %Tn2) * MIN(Tk2, Tk4-k3-k4);
                                            }

                                            int checkoff = n1* MIN(Tk1, Tk4-k2-k3-k4) + n2* MIN(Tk1, Tk4-k2-k3-k4) + k2* MIN(Tn2, Tn4-n3-n4) + n3* MIN(Tk2, Tk4-k3-k4) + k3* MIN(Tn3, Tn4-n4);

                                            if(Id%8!=0 && checkoff < add_Id_pckoff[Id%8]){
                                                for(int k1 = 0; k1 < Tk1; k1 += Tk0)
                                                {
                                                    if(+k1+k2+k3+k4>=Tk4)break;
                                                    int B_pack_offset = k1* Tn0 + checkoff;
                                                    int If = k1+k2+k3+k4;
                                                    for(int jj = 0; jj< 8;jj++){
                                                        Bbuf[B_pack_offset+jj] = B[ +(Ie+jj)*B_stride_e +If*B_stride_f +Ic*B_stride_c +Ia*B_stride_a +Id*B_stride_d];
                                                    }
                                                }
                                                continue;
                                            }
                                            if(Id % 8 !=0) continue;

                                            for(int k1 = 0; k1 < Tk1; k1 += Tk0)
                                            {
                                                if(+k1+k2+k3+k4>=Tk4)break;
                                                    int If = k1+k2+k3+k4;
// begin pack body
                                                    int B_pack_offset = // + n0
                                                        + k1* Tn0 + n1* MIN(Tk1, Tk4-k2-k3-k4) + n2* MIN(Tk1, Tk4-k2-k3-k4) + k2* MIN(Tn2, Tn4-n3-n4) + n3* MIN(Tk2, Tk4-k3-k4) + k3* MIN(Tn3, Tn4-n4);// + n4* MIN(Tk3, Tk4-k4) + k4* MIN(Tn4, Tn4);

    
/* for(int jj = 0; jj< 8;jj++){ */
/* Bbuf[B_pack_offset+jj] = B[ +(Ie+jj)*B_stride_e +If*B_stride_f +Ic*B_stride_c +Ia*B_stride_a +Id*B_stride_d]; */
/* Bbuf[B_pack_offset+jj + add_Id1_pckoff] = B[ +(Ie+jj)*B_stride_e +If*B_stride_f +Ic*B_stride_c +Ia*B_stride_a +(Id+1) * B_stride_d]; */
/* Bbuf[B_pack_offset+jj + add_Id2_pckoff ] = B[ +(Ie+jj)*B_stride_e +If*B_stride_f +Ic*B_stride_c +Ia*B_stride_a +(Id+2)*B_stride_d]; */
/* Bbuf[B_pack_offset+jj + add_Id3_pckoff] = B[ +(Ie+jj)*B_stride_e +If*B_stride_f +Ic*B_stride_c +Ia*B_stride_a +(Id+3)*B_stride_d]; */


/* Bbuf[B_pack_offset+jj + add_Id4_pckoff] = B[ +(Ie+jj)*B_stride_e +If*B_stride_f +Ic*B_stride_c +Ia*B_stride_a +(Id+4)*B_stride_d]; */

/* Bbuf[B_pack_offset+jj + add_Id5_pckoff] = B[ +(Ie+jj)*B_stride_e +If*B_stride_f +Ic*B_stride_c +Ia*B_stride_a +(Id+5)*B_stride_d]; */

/* Bbuf[B_pack_offset+jj + add_Id6_pckoff] = B[ +(Ie+jj)*B_stride_e +If*B_stride_f +Ic*B_stride_c +Ia*B_stride_a +(Id+6)*B_stride_d]; */
/* Bbuf[B_pack_offset+jj + add_Id7_pckoff] = B[ +(Ie+jj)*B_stride_e +If*B_stride_f +Ic*B_stride_c +Ia*B_stride_a +(Id+7)*B_stride_d]; */
/* }//end forjj */


                                                    WS_4xF64_T mymm0, mymm1, mymm2, mymm3,
                                                        mymm4, mymm5, mymm6, mymm7,
                                                        mymm8, mymm9, mymm10, mymm11,
                                                        mymm12, mymm13, mymm14, mymm15;
    
                                                    WS_MOVE_M4xF64_TO_R4xF64((double*) &B[+(Ie + 0)*B_stride_e +If*B_stride_f +Ic*B_stride_c +Ia*B_stride_a +Id*B_stride_d  ],mymm0);

                                                    WS_MOVE_M4xF64_TO_R4xF64((double*) &B[+(Ie + 1)*B_stride_e +If*B_stride_f +Ic*B_stride_c +Ia*B_stride_a +Id*B_stride_d  ],mymm1);

                                                    WS_MOVE_M4xF64_TO_R4xF64((double*) &B[+(Ie + 2)*B_stride_e +If*B_stride_f +Ic*B_stride_c +Ia*B_stride_a +Id*B_stride_d  ],mymm2);

                                                    WS_MOVE_M4xF64_TO_R4xF64((double*) &B[+(Ie + 3)*B_stride_e +If*B_stride_f +Ic*B_stride_c +Ia*B_stride_a +Id*B_stride_d  ],mymm3);

                                                    WS_MOVE_M4xF64_TO_R4xF64((double*) &B[+(Ie + 4)*B_stride_e +If*B_stride_f +Ic*B_stride_c +Ia*B_stride_a +Id*B_stride_d  ],mymm4);

                                                    WS_MOVE_M4xF64_TO_R4xF64((double*) &B[+(Ie + 5)*B_stride_e +If*B_stride_f +Ic*B_stride_c +Ia*B_stride_a +Id*B_stride_d  ],mymm5);

                                                    WS_MOVE_M4xF64_TO_R4xF64((double*) &B[+(Ie + 6)*B_stride_e +If*B_stride_f +Ic*B_stride_c +Ia*B_stride_a +Id*B_stride_d  ],mymm6);

                                                    WS_MOVE_M4xF64_TO_R4xF64((double*) &B[+(Ie + 7)*B_stride_e +If*B_stride_f +Ic*B_stride_c +Ia*B_stride_a +Id*B_stride_d  ],mymm7);
    

                                                    WS_PERMUTE_2F128(0x02, mymm0, mymm1,mymm8); //mask 2 0
                                                    WS_PERMUTE_2F128(0x13, mymm0, mymm1,mymm12); //mask  3 1  0011 0001

                                                    WS_PERMUTE_2F128(0x02, mymm2, mymm3,mymm9); 
                                                    WS_PERMUTE_2F128(0x13, mymm2, mymm3,mymm13);

                                                    WS_PERMUTE_2F128(0x02, mymm4, mymm5,mymm10); 
                                                    WS_PERMUTE_2F128(0x13, mymm4, mymm5,mymm14);

                                                    WS_PERMUTE_2F128(0x02, mymm6, mymm7,mymm11); 
                                                    WS_PERMUTE_2F128(0x13, mymm6, mymm7,mymm15);

                                                    WS_PERMUTE4X64_PD(0xD8, mymm8, mymm0); // mask  11 01 10 00
                                                    WS_PERMUTE4X64_PD(0xD8, mymm9, mymm1); 
                                                    WS_PERMUTE4X64_PD(0xD8, mymm10, mymm2);
                                                    WS_PERMUTE4X64_PD(0xD8, mymm11, mymm3);
                                                    WS_PERMUTE4X64_PD(0xD8, mymm12, mymm4);
                                                    WS_PERMUTE4X64_PD(0xD8, mymm13, mymm5);
                                                    WS_PERMUTE4X64_PD(0xD8, mymm14, mymm6);
                                                    WS_PERMUTE4X64_PD(0xD8, mymm15, mymm7);

                                                    WS_PERMUTE_2F128(0x02, mymm0, mymm1, mymm8);   //mask  0 2
                                                    WS_PERMUTE_2F128(0x02, mymm2, mymm3, mymm9);   //mask  0 2

                                                    WS_PERMUTE_2F128(0x13, mymm0, mymm1, mymm10);// mask  1 3
                                                    WS_PERMUTE_2F128(0x13, mymm2, mymm3, mymm11);// mask  1 3

                                                    WS_PERMUTE_2F128(0x02, mymm4, mymm5, mymm12);   //mask  0 2
                                                    WS_PERMUTE_2F128(0x02, mymm6, mymm7, mymm13);   //mask  0 2

                                                    WS_PERMUTE_2F128(0x13, mymm4, mymm5, mymm14);// mask  1 3
                                                    WS_PERMUTE_2F128(0x13, mymm6, mymm7, mymm15);// mask  1 3


                                                    WS_MOVE_R4xF64_TO_M4xF64(mymm8 ,(double*) &Bbuf[B_pack_offset ]);
                                                    WS_MOVE_R4xF64_TO_M4xF64(mymm9 ,(double*) &Bbuf[B_pack_offset + 4]);

                                                    WS_MOVE_R4xF64_TO_M4xF64(mymm10 ,(double*) &Bbuf[B_pack_offset + add_Id_pckoff[1]]);
                                                    WS_MOVE_R4xF64_TO_M4xF64(mymm11,(double*) &Bbuf[B_pack_offset + add_Id_pckoff[1]+ 4 ]);

                                                    WS_MOVE_R4xF64_TO_M4xF64(mymm12 ,(double*) &Bbuf[B_pack_offset + add_Id_pckoff[2] ]);
                                                    WS_MOVE_R4xF64_TO_M4xF64(mymm13 ,(double*) &Bbuf[B_pack_offset + add_Id_pckoff[2]+4 ]);
                                                    WS_MOVE_R4xF64_TO_M4xF64(mymm14 ,(double*) &Bbuf[B_pack_offset + add_Id_pckoff[3] ]);
                                                    WS_MOVE_R4xF64_TO_M4xF64(mymm15 ,(double*) &Bbuf[B_pack_offset + add_Id_pckoff[3]+4 ]);



/**
   8way second part
*/


                                                    WS_MOVE_M4xF64_TO_R4xF64((double*) &B[+(Ie + 0)*B_stride_e +If*B_stride_f +Ic*B_stride_c +Ia*B_stride_a +(Id+4)*B_stride_d  ],mymm0);

                                                    WS_MOVE_M4xF64_TO_R4xF64((double*) &B[+(Ie + 1)*B_stride_e +If*B_stride_f +Ic*B_stride_c +Ia*B_stride_a +(Id+4)*B_stride_d  ],mymm1);

                                                    WS_MOVE_M4xF64_TO_R4xF64((double*) &B[+(Ie + 2)*B_stride_e +If*B_stride_f +Ic*B_stride_c +Ia*B_stride_a +(Id+4)*B_stride_d  ],mymm2);

                                                    WS_MOVE_M4xF64_TO_R4xF64((double*) &B[+(Ie + 3)*B_stride_e +If*B_stride_f +Ic*B_stride_c +Ia*B_stride_a +(Id+4)*B_stride_d  ],mymm3);

                                                    WS_MOVE_M4xF64_TO_R4xF64((double*) &B[+(Ie + 4)*B_stride_e +If*B_stride_f +Ic*B_stride_c +Ia*B_stride_a +(Id+4)*B_stride_d  ],mymm4);

                                                    WS_MOVE_M4xF64_TO_R4xF64((double*) &B[+(Ie + 5)*B_stride_e +If*B_stride_f +Ic*B_stride_c +Ia*B_stride_a +(Id+4)*B_stride_d  ],mymm5);

                                                    WS_MOVE_M4xF64_TO_R4xF64((double*) &B[+(Ie + 6)*B_stride_e +If*B_stride_f +Ic*B_stride_c +Ia*B_stride_a +(Id+4)*B_stride_d  ],mymm6);

                                                    WS_MOVE_M4xF64_TO_R4xF64((double*) &B[+(Ie + 7)*B_stride_e +If*B_stride_f +Ic*B_stride_c +Ia*B_stride_a +(Id+4)*B_stride_d  ],mymm7);
    

                                                    WS_PERMUTE_2F128(0x02, mymm0, mymm1,mymm8); //mask 2 0
                                                    WS_PERMUTE_2F128(0x13, mymm0, mymm1,mymm12); //mask  3 1  0011 0001

                                                    WS_PERMUTE_2F128(0x02, mymm2, mymm3,mymm9); 
                                                    WS_PERMUTE_2F128(0x13, mymm2, mymm3,mymm13);

                                                    WS_PERMUTE_2F128(0x02, mymm4, mymm5,mymm10); 
                                                    WS_PERMUTE_2F128(0x13, mymm4, mymm5,mymm14);

                                                    WS_PERMUTE_2F128(0x02, mymm6, mymm7,mymm11); 
                                                    WS_PERMUTE_2F128(0x13, mymm6, mymm7,mymm15);

                                                    WS_PERMUTE4X64_PD(0xD8, mymm8, mymm0); // mask  11 01 10 00
                                                    WS_PERMUTE4X64_PD(0xD8, mymm9, mymm1); 
                                                    WS_PERMUTE4X64_PD(0xD8, mymm10, mymm2);
                                                    WS_PERMUTE4X64_PD(0xD8, mymm11, mymm3);
                                                    WS_PERMUTE4X64_PD(0xD8, mymm12, mymm4);
                                                    WS_PERMUTE4X64_PD(0xD8, mymm13, mymm5);
                                                    WS_PERMUTE4X64_PD(0xD8, mymm14, mymm6);
                                                    WS_PERMUTE4X64_PD(0xD8, mymm15, mymm7);

                                                    WS_PERMUTE_2F128(0x02, mymm0, mymm1, mymm8);   //mask  0 2
                                                    WS_PERMUTE_2F128(0x02, mymm2, mymm3, mymm9);   //mask  0 2

                                                    WS_PERMUTE_2F128(0x13, mymm0, mymm1, mymm10);// mask  1 3
                                                    WS_PERMUTE_2F128(0x13, mymm2, mymm3, mymm11);// mask  1 3

                                                    WS_PERMUTE_2F128(0x02, mymm4, mymm5, mymm12);   //mask  0 2
                                                    WS_PERMUTE_2F128(0x02, mymm6, mymm7, mymm13);   //mask  0 2

                                                    WS_PERMUTE_2F128(0x13, mymm4, mymm5, mymm14);// mask  1 3
                                                    WS_PERMUTE_2F128(0x13, mymm6, mymm7, mymm15);// mask  1 3


                                                    WS_MOVE_R4xF64_TO_M4xF64(mymm8 ,(double*) &Bbuf[B_pack_offset + add_Id_pckoff[4] ]);
                                                    WS_MOVE_R4xF64_TO_M4xF64(mymm9 ,(double*) &Bbuf[B_pack_offset + add_Id_pckoff[4]+ 4]);

                                                    WS_MOVE_R4xF64_TO_M4xF64(mymm10 ,(double*) &Bbuf[B_pack_offset + add_Id_pckoff[5]]);
                                                    WS_MOVE_R4xF64_TO_M4xF64(mymm11,(double*) &Bbuf[B_pack_offset + add_Id_pckoff[5]+ 4 ]);

                                                    WS_MOVE_R4xF64_TO_M4xF64(mymm12 ,(double*) &Bbuf[B_pack_offset + add_Id_pckoff[6] ]);
                                                    WS_MOVE_R4xF64_TO_M4xF64(mymm13 ,(double*) &Bbuf[B_pack_offset + add_Id_pckoff[6]+4 ]);
                                                    WS_MOVE_R4xF64_TO_M4xF64(mymm14 ,(double*) &Bbuf[B_pack_offset + add_Id_pckoff[7] ]);
                                                    WS_MOVE_R4xF64_TO_M4xF64(mymm15 ,(double*) &Bbuf[B_pack_offset + add_Id_pckoff[7]+4 ]);

                                            }
                    
    
                                        }//end if base_nrange
                                        else{

                                            
                                            exit(1);
                                                    /* int Ia = (In - base_n_range) / Ext_B_stride_a; */
                                                    /* int Ic = (In - base_n_range)  % Ext_B_stride_a / Ext_B_stride_c; */
                                                    /* int Id = (In - base_n_range)  % Ext_B_stride_c / Ext_B_stride_d; */
                                                    /* int Ie = (In - base_n_range) % Ext_B_stride_d+ base_range_e; */
                                                    /*                                                     int B_pack_offset = // + n0 */
                                                    /*     + k1* Tn0 + n1* MIN(Tk1, Tk4-k2-k3-k4) + n2* MIN(Tk1, Tk4-k2-k3-k4) + k2* MIN(Tn2, Tn4-n3-n4) + n3* MIN(Tk2, Tk4-k3-k4) + k3* MIN(Tn3, Tn4-n4);// + n4* MIN(Tk3, Tk4-k4) + k4* MIN(Tn4, Tn4); */
                                                    /* Bbuf[B_pack_offset] = B[ +Ie*B_stride_e +If*B_stride_f +Ic*B_stride_c +Ia*B_stride_a +Id*B_stride_d]; */
                                        }
// end pack body
                                    }
                                }
                            }
                        }
                    }
                
            

    
                    for(int k3 = 0; k3 < Tk3; k3 += Tk2)
                    {
                        if(+k3+k4>=Tk4)break;
                        for(int m3 = 0; m3 < Tm3; m3 += Tm2)
                        {
                            if(+m3+m4>=Tm4)break;
                            for(int n3 = 0; n3 < Tn3; n3 += Tn2)
                            {
                                if(+n3+n4>=Tn4)break;
                                for(int k2 = 0; k2 < Tk2; k2 += Tk1)
                                {
                                    if(+k2+k3+k4>=Tk4)break;
                                    for(int n2 = 0; n2 < Tn2; n2 += Tn1)
                                    {
                                        if(+n2+n3+n4>=Tn4)break;
                                        for(int m2 = 0; m2 < Tm2; m2 += Tm1)
                                        {
                                            if(+m2+m3+m4>=Tm4)break;
                                            for(int m1 = 0; m1 < Tm1; m1 += Tm0)
                                            {
                                                if(+m1+m2+m3+m4>=Tm4)break;
                                                for(int n1 = 0; n1 < Tn1; n1 += Tn0)
                                                {
                                                    if(+n1+n2+n3+n4>=Tn4)break;
                                                    int In =  n1+n2+n3+n4; int Im = m1+m2+m3+m4;
                                                    int A_pack_offset = // + m0 + k1* Tm0
                                                        + m1* MIN(Tk1, Tk4-k2-k3-k4) + m2* MIN(Tk1, Tk4-k2-k3-k4) + k2* MIN(Tm2, Tm4-m3-m4) + m3* MIN(Tk2, Tk4-k3-k4) + k3* MIN(Tm3, Tm4-m4) ;//+ m4* MIN(Tk3, Tk4-k4) + k4* MIN(Tm4, Tm4);
                                                    int use_k = MIN(Tk1, Tk4-k2-k3-k4);
                                                    int B_pack_offset =  //+ n0+ k1* Tn0
                                                        + n1* MIN(Tk1, Tk4-k2-k3-k4) + n2* MIN(Tk1, Tk4-k2-k3-k4) + k2* MIN(Tn2, Tn4-n3-n4) + n3* MIN(Tk2, Tk4-k3-k4) + k3* MIN(Tn3, Tn4-n4) ;//+ n4* MIN(Tk3, Tk4-k4) + k4* MIN(Tn4, Tn4);
// begin compute body
                                                    if( Im < base_m_range && In < base_n_range){
                                                        bli_dgemm_haswell_asm_6x8(use_k,  &alpha, Abuf+ A_pack_offset, Bbuf+B_pack_offset, &beta, C+ Ascatter[Im] + Bscatter[In], 1*C_range_c*C_range_d*C_range_e, 1, NULL, NULL);
                                                    }
                                                    else if(Im >= base_m_range && In < base_n_range){
                                                        sct_dgemm_haswell_asm_6x8_8vec(use_k, &alpha, Abuf+ A_pack_offset, Bbuf+B_pack_offset, &beta, C + Bscatter[In], Ascatter+Im);
                                                    }
                                                    else if(Im < base_m_range && In >= base_n_range){
                                                        sct_dgemm_haswell_asm_6x8(use_k, &alpha, Abuf+ A_pack_offset, Bbuf+B_pack_offset, &beta, C, Ascatter+Im, Bscatter+In);
                                                    }
                                                    else{
                                                    }
// end compute body
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

    }//end ccnt
MPI_Barrier(MPI_COMM_WORLD);
    RDTSC(b);
    double  runtime = omp_get_wtime() - start;
    double flop_count = 2.0 * C_range_c * C_range_a *C_range_b * C_range_d * C_range_e*C_range_f* tot_runtime;
    long long cycles_tuned = (long)(((double) (COUNTER_DIFF_SIMPLE(b,a))) / ((long long) 1));
     
    printf("gflops,  %lf\n", flop_count/1000/1000/1000/runtime);
    double flops_cycle_tuned = ((double)flop_count)/((double)cycles_tuned);
    
printf("fpc, %2.4f\n,", flops_cycle_tuned);

    for(int i=0;i<tot_runtime;i++)
    compare_fun(A, B, C1);

    for(int i=0; i< C_range_c * C_range_a *C_range_b * C_range_d*C_range_e; i++){
//        if(C[i] != C1[i]){
        if( C[i] != C1[i]){
            //printf("error at C[%d], C %lf, C1 %lf\n", i, C[i], C1[i]);
            return 0;
        }
    }
    printf("correct\n");
    return 0;
}