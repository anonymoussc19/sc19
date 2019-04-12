#include "../../tensor_kernels.h"
#include "../../scatter_kernels.h"
#include<omp.h>

int B_range_a= 72;
int A_range_b= 72;
int A_range_c= 72;
int B_range_d = 72;


int A_range_e= 72;
int A_range_f= 72;
int B_range_e= 72;
int B_range_f= 72;


int C_range_a= 72;
int C_range_b= 72;
int C_range_c= 72;
int C_range_d = 72;
int C_range_e= 72;
int C_range_f= 72;


void compare_fun(double* A, double*B,double*C ){
        return;
for(int a=0; a<C_range_a; a++)
for(int b=0; b<C_range_b; b++)
for(int c=0; c<C_range_c; c++)
for(int d=0; d<C_range_d; d++)
for(int e=0; e<C_range_e; e++)
for(int f=0; f<C_range_f; f++)
C[+a* C_range_b* C_range_c* C_range_d+b* C_range_c* C_range_d+c* C_range_d+d] += 
A[+f* C_range_b* C_range_e* C_range_c+b* C_range_e* C_range_c+e* C_range_c+c] * 
B[+e* C_range_a* C_range_f* C_range_d+a* C_range_f* C_range_d+f* C_range_d+d]; 
}

int main(int argc, char** argv){

  int Tk4, Tk3, Tk2, Tk1, Tk0;
    int Tm4, Tm3, Tm2, Tm1, Tm0;
    int Tn4, Tn3, Tn2, Tn1, Tn0;

    B_range_a =  C_range_a = atoi(argv[1]);
    A_range_b =  C_range_b = atoi(argv[2]);
    A_range_c =  C_range_c = atoi(argv[3]);
    B_range_d =  C_range_d = atoi(argv[4]);
   B_range_e  = A_range_e =  C_range_e = atoi(argv[5]);
    A_range_f=B_range_f =  C_range_f = atoi(argv[6]);

    
    printf("abcd-eafd-fbec\n");
    
                Tk4 =  C_range_f * C_range_e;;
    Tk3 = 256;
    Tk2 = Tk1 = 256;
//    Tk1 = 12;
    Tk0 = 1;
    Tm4 =  B_range_a *B_range_d;
    Tm3 = 60;
    Tm2 = 60;
    Tm1 = 6;
    Tm0 = 6;

    Tn4 = A_range_c * A_range_b;
    Tn3 = 3072;
    Tn2 = 16;
    Tn1 = 8;
    Tn0 = 8;

            double *A, *B, *C, *Abuf, *Bbuf;
C = (double*)memalign(4096, sizeof(double) * C_range_a * C_range_d * C_range_b * C_range_c);
double *C1 = (double*)memalign(4096, sizeof(double) * C_range_a * C_range_d * C_range_b * C_range_c);



A = (double*)memalign(4096, sizeof(double) * A_range_f * A_range_b * A_range_e * A_range_c);
B = (double*)memalign(4096, sizeof(double) * B_range_e * B_range_a * B_range_f * B_range_d);
for(int i = 0; i < 1*A_range_f*A_range_b*A_range_e*A_range_c; i++){
A[i] = rand()%1000;
}
for(int i = 0; i < 1*B_range_e*B_range_a*B_range_f*B_range_d; i++){
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
    RDTSC(a);
    clock_t begin = clock();
    for(int ccnt = 0; ccnt < tot_runtime; ccnt++){


//Abuf = (double*)memalign(4096, sizeof(double) * A_range_f * A_range_b * A_range_e * A_range_c);
Abuf = (double*)memalign(4096, sizeof(double) * Tm3*Tk3);
int ext_range_c = A_range_c % Tm0;
int base_range_c = A_range_c - ext_range_c;
int base_m_range = base_range_c * A_range_b;
unsigned long long *Ascatter = (unsigned long long*)memalign(4096, sizeof(unsigned long long)*A_range_b*A_range_c);

int A_stride_f = 1 * A_range_b * A_range_e * A_range_c;
int A_stride_b = 1 * A_range_e * A_range_c;
int A_stride_e = 1 * A_range_c;
int A_stride_c = 1;

int Fused_A_stride_b = 1 * base_range_c;
int Fused_A_stride_c = 1;
int Ext_A_stride_b = 1 * ext_range_c;
int Ext_A_stride_c = 1;

for(int Im=0; Im < base_m_range;Im++){
int Ib = Im / Fused_A_stride_b;
int Ic = Im % Fused_A_stride_b;
Ascatter[Im] = +Ib*C_range_c*C_range_d+Ic*C_range_d;
}
for(int Im=base_m_range; Im < Tm4;Im++){
int Ib = (Im - base_m_range) / Ext_A_stride_b;
int Ic = (Im - base_m_range) % Ext_A_stride_b+ base_range_c;
Ascatter[Im] = +Ib*C_range_c*C_range_d+Ic*C_range_d;
}

//Bbuf = (double*)memalign(4096, sizeof(double) * B_range_e * B_range_a * B_range_f * B_range_d);
Bbuf = (double*)memalign(4096, sizeof(double) * Tk3*Tn3);
int ext_range_d = B_range_d % Tn0;
int base_range_d = B_range_d - ext_range_d;
int base_n_range = base_range_d * B_range_a;
unsigned long long *Bscatter = (unsigned long long*)memalign(4096, sizeof(unsigned long long)*B_range_a*B_range_d);

int B_stride_e = 1 * B_range_a * B_range_f * B_range_d;
int B_stride_a = 1 * B_range_f * B_range_d;
int B_stride_f = 1 * B_range_d;
int B_stride_d = 1;

int Fused_B_stride_a = 1 * base_range_d;
int Fused_B_stride_d = 1;
int Ext_B_stride_a = 1 * ext_range_d;
int Ext_B_stride_d = 1;

for(int In=0; In < base_n_range;In++){
int Ia = In / Fused_B_stride_a;
int Id = In % Fused_B_stride_a;
Bscatter[In] = +Ia*C_range_b*C_range_c*C_range_d+Id;
}
for(int In=base_n_range; In < Tn4;In++){
int Ia = (In - base_n_range) / Ext_B_stride_a;
int Id = (In - base_n_range) % Ext_B_stride_a+ base_range_d;
Bscatter[In] = +Ia*C_range_b*C_range_c*C_range_d+Id;
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
/* int Im  = +m0+m1+m2+m3+m4; */
/* int Ie = (k1+k2+k3+k4)/C_range_e; */
/* int If = (k1+k2+k3+k4)%C_range_e; */
/* // begin pack body */
/* int A_pack_offset =  + m0 + k1* Tm0 + m1* MIN(Tk1, Tk4-k2-k3-k4) + m2* MIN(Tk1, Tk4-k2-k3-k4) + k2* MIN(Tm2, Tm4-m3-m4) + m3* MIN(Tk2, Tk4-k3-k4) + k3* MIN(Tm3, Tm4-m4) + m4* MIN(Tk3, Tk4-k4) + k4* MIN(Tm4, Tm4); */
/* if(Im< base_m_range){ */
/* int Ib = Im / Fused_A_stride_b; */
/* int Ic = Im % Fused_A_stride_b; */
/* Abuf[A_pack_offset] = A[ +If*A_stride_f +Ib*A_stride_b +Ie*A_stride_e +Ic*A_stride_c]; */
/* } */
/* else{ */
/* int Ib = (Im - base_m_range) / Ext_A_stride_b; */
/* int Ic = (Im - base_m_range) % Ext_A_stride_b+ base_range_c; */
/* Abuf[A_pack_offset] = A[ +If*A_stride_f +Ib*A_stride_b +Ie*A_stride_e +Ic*A_stride_c]; */
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
/* for(int n0 = 0; n0 < Tn0; n0 += 1) */
/* { */
/* if(+n0+n1+n2+n3+n4>=Tn4)break; */
/* int In = +n0+n1+n2+n3+n4; */
/* int Ie = (k1+k2+k3+k4)/C_range_e; */
/* int If = (k1+k2+k3+k4)%C_range_e; */
/* // begin pack body */
/* int B_pack_offset =  + n0 + k1* Tn0 + n1* MIN(Tk1, Tk4-k2-k3-k4) + n2* MIN(Tk1, Tk4-k2-k3-k4) + k2* MIN(Tn2, Tn4-n3-n4) + n3* MIN(Tk2, Tk4-k3-k4) + k3* MIN(Tn3, Tn4-n4) + n4* MIN(Tk3, Tk4-k4) + k4* MIN(Tn4, Tn4); */
/* if(In< base_n_range){ */
/* int Ia = In / Fused_B_stride_a; */
/* int Id = In % Fused_B_stride_a; */
/* Bbuf[B_pack_offset] = B[ +Ie*B_stride_e +Ia*B_stride_a +If*B_stride_f +Id*B_stride_d]; */
/* } */
/* else{ */
/* int Ia = (In - base_n_range) / Ext_B_stride_a; */
/* int Id = (In - base_n_range) % Ext_B_stride_a+ base_range_d; */
/* Bbuf[B_pack_offset] = B[ +Ie*B_stride_e +Ia*B_stride_a +If*B_stride_f +Id*B_stride_d]; */
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
for(int k1 = 0; k1 < Tk1; k1 += Tk0)
{
if(+k1+k2+k3+k4>=Tk4)break;
for(int n0 = 0; n0 < Tn0; n0 += 1)
{
if(+n0+n1+n2+n3+n4>=Tn4)break;
int In = +n0+n1+n2+n3+n4;
int Ie = (k1+k2+k3+k4)/C_range_e;
int If = (k1+k2+k3+k4)%C_range_e;
// begin pack body
int B_pack_offset =  + n0 + k1* Tn0 + n1* MIN(Tk1, Tk4-k2-k3-k4) + n2* MIN(Tk1, Tk4-k2-k3-k4) + k2* MIN(Tn2, Tn4-n3-n4) + n3* MIN(Tk2, Tk4-k3-k4) + k3* MIN(Tn3, Tn4-n4) ;//+ n4* MIN(Tk3, Tk4-k4) + k4* MIN(Tn4, Tn4);
if(In< base_n_range){
int Ia = In / Fused_B_stride_a;
int Id = In % Fused_B_stride_a;
Bbuf[B_pack_offset] = B[ +Ie*B_stride_e +Ia*B_stride_a +If*B_stride_f +Id*B_stride_d];
}
else{
int Ia = (In - base_n_range) / Ext_B_stride_a;
int Id = (In - base_n_range) % Ext_B_stride_a+ base_range_d;
Bbuf[B_pack_offset] = B[ +Ie*B_stride_e +Ia*B_stride_a +If*B_stride_f +Id*B_stride_d];
}
// end pack body
}
}
}
}
}
}
}
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
int Im  = +m0+m1+m2+m3+m4;
int Ie = (k1+k2+k3+k4)/C_range_e;
int If = (k1+k2+k3+k4)%C_range_e;
// begin pack body
int A_pack_offset =  + m0 + k1* Tm0 + m1* MIN(Tk1, Tk4-k2-k3-k4) + m2* MIN(Tk1, Tk4-k2-k3-k4) + k2* MIN(Tm2, Tm4-m3-m4) + m3* MIN(Tk2, Tk4-k3-k4) + k3* MIN(Tm3, Tm4-m4) ;//+ m4* MIN(Tk3, Tk4-k4) + k4* MIN(Tm4, Tm4);
if(Im< base_m_range){
int Ib = Im / Fused_A_stride_b;
int Ic = Im % Fused_A_stride_b;
Abuf[A_pack_offset] = A[ +If*A_stride_f +Ib*A_stride_b +Ie*A_stride_e +Ic*A_stride_c];
}
else{
int Ib = (Im - base_m_range) / Ext_A_stride_b;
int Ic = (Im - base_m_range) % Ext_A_stride_b+ base_range_c;
Abuf[A_pack_offset] = A[ +If*A_stride_f +Ib*A_stride_b +Ie*A_stride_e +Ic*A_stride_c];
}
// end pack body
}
}
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
    int A_pack_offset =  //+ m0 + k1* Tm0
        + m1* MIN(Tk1, Tk4-k2-k3-k4) + m2* MIN(Tk1, Tk4-k2-k3-k4) + k2* MIN(Tm2, Tm4-m3-m4) + m3* MIN(Tk2, Tk4-k3-k4) + k3* MIN(Tm3, Tm4-m4) ;//+ m4* MIN(Tk3, Tk4-k4) + k4* MIN(Tm4, Tm4);
    int B_pack_offset =  //+ n0 + k1* Tn0
        + n1* MIN(Tk1, Tk4-k2-k3-k4) + n2* MIN(Tk1, Tk4-k2-k3-k4) + k2* MIN(Tn2, Tn4-n3-n4) + n3* MIN(Tk2, Tk4-k3-k4) + k3* MIN(Tn3, Tn4-n4) ;//+ n4* MIN(Tk3, Tk4-k4) + k4* MIN(Tn4, Tn4);
    int use_k = MIN(Tk1, Tk4-k2-k3-k4);
// begin compute body
if( Im < base_m_range && In < base_n_range){
bli_dgemm_haswell_asm_6x8(use_k,  &alpha, Abuf+ A_pack_offset, Bbuf+B_pack_offset, &beta, C+ Ascatter[Im] + Bscatter[In], 1*C_range_d, 1, NULL, NULL);
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
        RDTSC(b);        double runtime = (double)(clock()-begin)/CLOCKS_PER_SEC;
    double flop_count = 2.0 * C_range_a * C_range_b *C_range_c
            *C_range_d * C_range_e * C_range_f *tot_runtime;
    long long cycles_tuned = (long)(((double) (COUNTER_DIFF_SIMPLE(b,a))) / ((long long) 1));
    printf("flopcnt = %lf, ",flop_count);
    double flops_cycle_tuned = ((double)flop_count)/((double)cycles_tuned);
    printf("tot cyc= %12ld,  cycles,    %2.4f, flops/cycle\n",cycles_tuned, flops_cycle_tuned);
    printf("gflops = %lf\n", flop_count/1000/1000/1000/runtime);
    for(int i=0;i<tot_runtime;i++)
    compare_fun(A, B, C1);

    for(int i=0; i< C_range_a * C_range_b *C_range_c
            *C_range_d  ;i++){
//        if(C[i] != C1[i]){
        if( C[i] != C1[i]){
            printf("error at C[%d], C %lf, C1 %lf\n", i, C[i], C1[i]);
                return 1;
        }
    }
    printf("correct\n");

return 0;
}