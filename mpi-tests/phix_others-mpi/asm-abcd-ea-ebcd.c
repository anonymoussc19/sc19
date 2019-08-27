#include "ws_mapping_avx.h"
#include "../../tensor_kernels.h"
#include "../../scatter_kernels.h"
#include<omp.h>
#include"mpi.h"


int A_range_e = 72;
int A_range_a = 72;

int B_range_b = 72;
int B_range_c = 72;
int B_range_d = 72;
int B_range_e = 72;

int C_range_a = 72;
int C_range_b = 72;
int C_range_c = 72;
int C_range_d = 72;
int C_range_e = 72;


void compare_fun(double* A, double*B,double*C ){
for(int a=0; a<C_range_a; a++)
for(int b=0; b<C_range_b; b++)
for(int c=0; c<C_range_c; c++)
for(int d=0; d<C_range_d; d++)
for(int e=0; e<C_range_e; e++)
C[+a* C_range_b* C_range_c* C_range_d+b* C_range_c* C_range_d+c* C_range_d+d] += 
A[+e* C_range_a+a] * 
B[+e* C_range_b* C_range_c* C_range_d+b* C_range_c* C_range_d+c* C_range_d+d]; 
}






int main(int argc, char** argv){

        int Tk4, Tk3, Tk2, Tk1, Tk0;
    int Tm4, Tm3, Tm2, Tm1, Tm0;
    int Tn4, Tn3, Tn2, Tn1, Tn0;
    Tk4 = C_range_e;
    Tm4 = A_range_a;
    Tn4 = B_range_b * B_range_c * B_range_d;

    

    Tn3 = 1042/24*24;
    Tm3 = 72;
        
    Tn2 = 24;
    Tm2 = 72;
    
    Tn1 = 24;
    Tm1 = 6;
    Tk1 = Tk2 = Tk3  = 72;
    
    Tk0 = 1; Tm0 =6; Tn0=8;
            double *A, *B, *C, *Abuf, *Bbuf;
C = (double*)memalign(4096, sizeof(double) * C_range_b * C_range_c * C_range_a*C_range_d);
double *C1 = (double*)memalign(4096, sizeof(double) * C_range_b * C_range_c * C_range_a*C_range_d);


A = (double*)memalign(4096, sizeof(double) * A_range_e * A_range_a);
B = (double*)memalign(4096, sizeof(double) * B_range_e * B_range_b * B_range_c * B_range_d);
for(int i = 0; i < 1*A_range_e*A_range_a; i++){
A[i] = rand()%1000;
}
for(int i = 0; i < 1*B_range_e*B_range_b*B_range_c*B_range_d; i++){
B[i] = rand()%1000;
}
for(int i = 0; i < 1*C_range_a*C_range_b*C_range_c*C_range_d; i++){
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
 MPI_Init(NULL,NULL);
  MPI_Barrier(MPI_COMM_WORLD);
double start = omp_get_wtime();
    RDTSC(a);
    for(int ccnt = 0; ccnt < tot_runtime; ccnt++){

//Abuf = (double*)memalign(4096, sizeof(double) * A_range_e * A_range_a);
        Abuf = (double*)memalign(4096, sizeof(double) * Tk3*Tm3);
int ext_range_a = A_range_a % Tm0;
int base_range_a = A_range_a - ext_range_a;
int base_m_range = base_range_a;
unsigned long long *Ascatter = (unsigned long long*)memalign(4096, sizeof(unsigned long long)*A_range_a);

int A_stride_e = 1 * A_range_a;
int A_stride_a = 1;

int Fused_A_stride_a = 1;
int Ext_A_stride_a = 1;

for(int Im=0; Im < base_m_range;Im++){
int Ia = Im / Fused_A_stride_a;
Ascatter[Im] = +Ia*C_range_b*C_range_c*C_range_d;
}
for(int Im=base_m_range; Im < Tm4;Im++){
int Ia = (Im - base_m_range) / Ext_A_stride_a;
Ascatter[Im] = +Ia*C_range_b*C_range_c*C_range_d;
}

//Bbuf = (double*)memalign(4096, sizeof(double) * B_range_e * B_range_b * B_range_c * B_range_d);
Bbuf = (double*)memalign(4096, sizeof(double) * Tn3*Tk3);
int ext_range_d = B_range_d % Tn0;
int base_range_d = B_range_d - ext_range_d;
int base_n_range = base_range_d * B_range_c * B_range_b;
unsigned long long *Bscatter = (unsigned long long*)memalign(4096, sizeof(unsigned long long)*B_range_b*B_range_c*B_range_d);

int B_stride_e = 1 * B_range_b * B_range_c * B_range_d;
int B_stride_b = 1 * B_range_c * B_range_d;
int B_stride_c = 1 * B_range_d;
int B_stride_d = 1;

int Fused_B_stride_b = 1 * B_range_c * base_range_d;
int Fused_B_stride_c = 1 * base_range_d;
int Fused_B_stride_d = 1;
int Ext_B_stride_b = 1 * B_range_c * ext_range_d;
int Ext_B_stride_c = 1 * ext_range_d;
int Ext_B_stride_d = 1;

for(int In=0; In < base_n_range;In++){
int Ib = In / Fused_B_stride_b;
int Ic = In % Fused_B_stride_b / Fused_B_stride_c;
int Id = In % Fused_B_stride_c;
Bscatter[In] = +Ib*C_range_c*C_range_d+Ic*C_range_d+Id;
}
for(int In=base_n_range; In < Tn4;In++){
int Ib = (In - base_n_range) / Ext_B_stride_b;
int Ic = (In - base_n_range)  % Ext_B_stride_b / Ext_B_stride_c;
int Id = (In - base_n_range) % Ext_B_stride_c+ base_range_d;
Bscatter[In] = +Ib*C_range_c*C_range_d+Ic*C_range_d+Id;
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
/* int Im = m0+m1+m2+m3+m4; int Ie =  k1+k2+k3+k4; */
/* // begin pack body */
/* int A_pack_offset =  + m0 + k1* Tm0 + m1* MIN(Tk1, Tk4-k2-k3-k4) + m2* MIN(Tk1, Tk4-k2-k3-k4) + k2* MIN(Tm2, Tm4-m3-m4) + m3* MIN(Tk2, Tk4-k3-k4) + k3* MIN(Tm3, Tm4-m4) + m4* MIN(Tk3, Tk4-k4) + k4* MIN(Tm4, Tm4); */
/* if(Im< base_m_range){ */
/* int Ia = Im / Fused_A_stride_a; */
/* Abuf[A_pack_offset] = A[ +Ie*A_stride_e +Ia*A_stride_a]; */
/* } */
/* else{ */
/* int Ia = (Im - base_m_range) / Ext_A_stride_a; */
/* Abuf[A_pack_offset] = A[ +Ie*A_stride_e +Ia*A_stride_a]; */
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
/* //int In = n0+ n1+n2+n3+n4; int Ie = k1+k2+k3+k4; */

/*     int In =  n1+n2+n3+n4; int Ie = k1+k2+k3+k4; */
/*     int B_pack_offset =  //+ n0 */
/*         + k1* Tn0 + n1* MIN(Tk1, Tk4-k2-k3-k4) + n2* MIN(Tk1, Tk4-k2-k3-k4) + k2* MIN(Tn2, Tn4-n3-n4) + n3* MIN(Tk2, Tk4-k3-k4) + k3* MIN(Tn3, Tn4-n4) + n4* MIN(Tk3, Tk4-k4) + k4* MIN(Tn4, Tn4); */
/* // begin pack body */

/* if(In+8<=base_n_range){ */
/* int Ib = In / Fused_B_stride_b; */
/* int Ic = In % Fused_B_stride_b / Fused_B_stride_c; */
/* int Id = In % Fused_B_stride_c; */

/* for(int jj = 0; jj< 8;jj++){ */
/*     Bbuf[B_pack_offset+jj] = B[ +Ie*B_stride_e +Ib*B_stride_b +Ic*B_stride_c +Id*B_stride_d +jj]; */
/* } */
/* } */
/* else{ */
/*     printf("exit\n");    exit(1); */
/* int Ib = (In - base_n_range) / Ext_B_stride_b; */
/* int Ic = (In - base_n_range)  % Ext_B_stride_b / Ext_B_stride_c; */
/* int Id = (In - base_n_range) % Ext_B_stride_c+ base_range_d; */
/* Bbuf[B_pack_offset] = B[ +Ie*B_stride_e +Ib*B_stride_b +Ic*B_stride_c +Id*B_stride_d]; */
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

//for(int m0 = 0; m0 < Tm0; m0 += 1)

//if(+m0+m1+m2+m3+m4>=Tm4)break;
int Im = m1+m2+m3+m4;
// begin pack body

if(Im< base_m_range){
    
int Ia = Im / Fused_A_stride_a;
for(int k1 = 0; k1 < Tk1; k1 += Tk0)
{
if(+k1+k2+k3+k4>=Tk4)break;
 int Ie =  k1+k2+k3+k4;
int A_pack_offset =//  + m0
    + k1* Tm0 + m1* MIN(Tk1, Tk4-k2-k3-k4) + m2* MIN(Tk1, Tk4-k2-k3-k4) + k2* MIN(Tm2, Tm4-m3-m4) + m3* MIN(Tk2, Tk4-k3-k4) + k3* MIN(Tm3, Tm4-m4);// + m4* MIN(Tk3, Tk4-k4) + k4* MIN(Tm4, Tm4);
for(int ii=0;ii<6;ii++){
    Abuf[A_pack_offset+ii] = A[ +Ie*A_stride_e +(Ia+ii)*A_stride_a ];
}
}
}
else{
    printf("exita\n")  ;  exit(1);
int Ia = (Im - base_m_range) / Ext_A_stride_a;
for(int k1 = 0; k1 < Tk1; k1 += Tk0)
{
if(+k1+k2+k3+k4>=Tk4)break;
 int Ie =  k1+k2+k3+k4;
 int A_pack_offset =//  + m0
    + k1* Tm0 + m1* MIN(Tk1, Tk4-k2-k3-k4) + m2* MIN(Tk1, Tk4-k2-k3-k4) + k2* MIN(Tm2, Tm4-m3-m4) + m3* MIN(Tk2, Tk4-k3-k4) + k3* MIN(Tm3, Tm4-m4);// + m4* MIN(Tk3, Tk4-k4) + k4* MIN(Tm4, Tm4);
Abuf[A_pack_offset] = A[ +Ie*A_stride_e +Ia*A_stride_a];
}
}
// end pack body

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

//for(int n0 = 0; n0 < Tn0; n0 += 1)

//if(+n0+n1+n2+n3+n4>=Tn4)break;
//int In = n0+ n1+n2+n3+n4; int Ie = k1+k2+k3+k4;

    int In =  n1+n2+n3+n4; 
 
// begin pack body

if(In+8<=base_n_range){
int Ib = In / Fused_B_stride_b;
int Ic = In % Fused_B_stride_b / Fused_B_stride_c;
int Id = In % Fused_B_stride_c;
for(int k1 = 0; k1 < Tk1; k1 += Tk0)
{
    int Ie = k1+k2+k3+k4;
if(Ie>=Tk4)break;

   int B_pack_offset =  //+ n0
        + k1* Tn0 + n1* MIN(Tk1, Tk4-k2-k3-k4) + n2* MIN(Tk1, Tk4-k2-k3-k4) + k2* MIN(Tn2, Tn4-n3-n4) + n3* MIN(Tk2, Tk4-k3-k4) + k3* MIN(Tn3, Tn4-n4) ;//+ n4* MIN(Tk3, Tk4-k4) + k4* MIN(Tn4, Tn4);
for(int jj = 0; jj< 8;jj++){
    Bbuf[B_pack_offset+jj] = B[ +Ie*B_stride_e +Ib*B_stride_b +Ic*B_stride_c +Id*B_stride_d +jj];
}
   /* WS_4xF64_T mymm0, mymm1; */
   /* WS_MOVE_M4xF64_TO_R4xF64((double*)&B[+Ie*B_stride_e +Ib*B_stride_b +Ic*B_stride_c +(Id+0)*B_stride_d ], mymm0); */
   /* WS_MOVE_M4xF64_TO_R4xF64((double*)&B[+Ie*B_stride_e +Ib*B_stride_b +Ic*B_stride_c +(Id+4)*B_stride_d ], mymm1); */
   /* WS_MOVE_R4xF64_TO_M4xF64(mymm0 ,(double*) &Bbuf[B_pack_offset ]); */
   /* WS_MOVE_R4xF64_TO_M4xF64(mymm1 ,(double*) &Bbuf[B_pack_offset +4 ]); */

   
}
}
else{
    printf("exit\n");    exit(1);
int Ib = (In - base_n_range) / Ext_B_stride_b;
int Ic = (In - base_n_range)  % Ext_B_stride_b / Ext_B_stride_c;
int Id = (In - base_n_range) % Ext_B_stride_c+ base_range_d;
for(int k1 = 0; k1 < Tk1; k1 += Tk0)
{
if(+k1+k2+k3+k4>=Tk4)break;
int Ie = k1+k2+k3+k4;
   int B_pack_offset =  //+ n0
        + k1* Tn0 + n1* MIN(Tk1, Tk4-k2-k3-k4) + n2* MIN(Tk1, Tk4-k2-k3-k4) + k2* MIN(Tn2, Tn4-n3-n4) + n3* MIN(Tk2, Tk4-k3-k4) + k3* MIN(Tn3, Tn4-n4) ;//+ n4* MIN(Tk3, Tk4-k4) + k4* MIN(Tn4, Tn4);
Bbuf[B_pack_offset] = B[ +Ie*B_stride_e +Ib*B_stride_b +Ic*B_stride_c +Id*B_stride_d];
}
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

int Im = m1+m2+m3+m4;
int In = n1+n2+n3+n4;
int A_pack_offset = // + m0 + k1* Tm0
    + m1* MIN(Tk1, Tk4-k2-k3-k4) + m2* MIN(Tk1, Tk4-k2-k3-k4) + k2* MIN(Tm2, Tm4-m3-m4) + m3* MIN(Tk2, Tk4-k3-k4) + k3* MIN(Tm3, Tm4-m4);// + m4* MIN(Tk3, Tk4-k4) + k4* MIN(Tm4, Tm4);
int B_pack_offset = // + n0 + k1* Tn0
    + n1* MIN(Tk1, Tk4-k2-k3-k4) + n2* MIN(Tk1, Tk4-k2-k3-k4) + k2* MIN(Tn2, Tn4-n3-n4) + n3* MIN(Tk2, Tk4-k3-k4) + k3* MIN(Tn3, Tn4-n4); //+ n4* MIN(Tk3, Tk4-k4) + k4* MIN(Tn4, Tn4);
// begin compute body
if( Im < base_m_range && In < base_n_range){
bli_dgemm_haswell_asm_6x8(Tk3,  &alpha, Abuf+ A_pack_offset, Bbuf+B_pack_offset, &beta, C+ Ascatter[Im] + Bscatter[In], 1*C_range_b*C_range_c*C_range_d, 1, NULL, NULL);
}
else if(Im >= base_m_range && In < base_n_range){
sct_dgemm_haswell_asm_6x8_8vec(Tk3, &alpha, Abuf+ A_pack_offset, Bbuf+B_pack_offset, &beta, C + Bscatter[In], Ascatter+Im);
}
else if(Im < base_m_range && In >= base_n_range){
sct_dgemm_haswell_asm_6x8(Tk3, &alpha, Abuf+ A_pack_offset, Bbuf+B_pack_offset, &beta, C, Ascatter+Im, Bscatter+In);
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
    double flop_count = 2.0 * C_range_c * C_range_a *C_range_b * C_range_d * A_range_e* tot_runtime;
        long long cycles_tuned = (long)(((double) (COUNTER_DIFF_SIMPLE(b,a))) / ((long long) 1)); double  runtime = omp_get_wtime() - start;
    printf("flopcnt = %lf, ",flop_count);
    double flops_cycle_tuned = ((double)flop_count)/((double)cycles_tuned);
    printf("tot cyc= %12ld,  cycles,    %2.4f, flops/cycle\n",cycles_tuned, flops_cycle_tuned);printf("gflops = %lf\n", flop_count/1000/1000/1000/runtime); printf("");

    for(int i=0;i<tot_runtime;i++)
    compare_fun(A, B, C1);

    for(int i=0; i< C_range_c * C_range_a *C_range_b * C_range_d; i++){
//        if(C[i] != C1[i]){
        if( C[i] != C1[i]){
            printf("error at C[%d], C %lf, C1 %lf\n", i, C[i], C1[i]);
                return 1;
        }
    }
    printf("correct\n");
    return 0;
}
