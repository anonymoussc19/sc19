#include<stdio.h>
#include<time.h>
#include"tblis.h"
#include<malloc.h>
#include "rdtsc.h"

int B_range_a=312;
int A_range_b=312;
int B_range_c=312;

int A_range_d=312;
int B_range_d=312;

int C_range_a=312;
int C_range_b=312;
int C_range_c=312;
int C_range_d=312;


void compare_fun(double* A, double*B,double*C ){
for(int a=0; a<C_range_a; a++)
for(int b=0; b<C_range_b; b++)
for(int c=0; c<C_range_c; c++)
for(int d=0; d<C_range_d; d++)
C[+a* C_range_b* C_range_c+b* C_range_c+c] += 
A[+d* C_range_b+b] * 
B[+a* C_range_c* C_range_d+c* C_range_d+d]; 
}

int main(int argc, char** argv)
{
//    int m=3072, n=3072, k=3072;
    /* int m,n,k; */
    /* m = atoi(argv[1]); */
    /* n = atoi(argv[2]); */
    /* k = atoi(argv[3]); */

    int a4,b4,c4,d4;
//    a4=b4=c4=d4=e4=f4=24;
//    a4=c4=19; f4=d4=19;
//    d4=e4=f4=17;
//    a4=b4=e4=23;
//    g4=24;

//    a4=b4=d4=312; c4=24;
    a4 = C_range_a;
    b4 = C_range_b;
    c4 = C_range_c;
    d4 = C_range_d;
//cgab, gefd
    double* data_A = (double*)memalign(4096,b4*d4*sizeof(double));
    tblis_tensor A;


    len_type Altp[2] = {d4,b4,}; // m*k
    stride_type Astp[2] = {   b4, 1};
    len_type* Alentp = &Altp;

    stride_type*  Asttp = &Astp;
    for(int i=0;i<b4*d4;i++)data_A[i] = rand()%1000;


    tblis_init_tensor_d(&A, 2, Alentp, data_A, Asttp);

    double* data_B = (double*)memalign(4096,a4*c4*d4*sizeof(double));
    tblis_tensor B;
    len_type Bltp[3] = {a4,c4, d4 };// n*k

    
//    stride_type Bstp[4] = {e4*f4*d4, f4*d4, d4,1};
    stride_type Bstp[3] = {c4*d4,d4,1};
    len_type* Blentp = &Bltp;
    stride_type*  Bsttp = &Bstp;
    for(int i=0;i<a4*c4*d4;i++)data_B[i] = rand()%1000;
    tblis_init_tensor_d(&B, 3, Blentp, data_B, Bsttp);

    double* data_C = (double*)memalign(4096,a4*b4*c4*sizeof(double));
    double* data_C2 = (double*)memalign(4096,a4*b4*c4*sizeof(double));
    for(int i = 0; i < a4*b4*c4; i++){
        data_C[i] = data_C2[i]=0;
    }

    tblis_tensor C;
    len_type Cltp[3] = {a4,b4,c4}; // m*n
//    stride_type Cstp[6] = {24*24*24*24*24, 24*24*24*24,24*24*24,24*24,24,1};
    stride_type Cstp[3] = {b4*c4,c4, 1};
    len_type* Clentp = &Cltp;
    stride_type*  Csttp =&Cstp;

    tblis_init_tensor_d(&C, 3, Clentp, data_C, Csttp);
//    tblis_tensor_mult(NULL, NULL, &A, "cgab", &B, "gefd", &C, "abcdef");
//    data_A[0] = rand()/1000;

    //tblis_tensor_mult(NULL, NULL, &A, "cgab", &B, "gefd", &C, "abcdef");
    
    for(int i = 0; i < a4*b4*c4; i++){
        data_C2[i]=0;
    }
    clock_t begin = clock();   
    static tsc_counter a,b;
                // timing
    /* warm up, according to Intel manual */
    CPUID(); RDTSC(a); CPUID(); RDTSC(b);
    /* warm up */
    CPUID(); RDTSC(a); CPUID(); RDTSC(b);
    /* warm up */
    CPUID(); RDTSC(a); CPUID(); RDTSC(b);
//    tblis_tensor_mult(tblis_single, NULL, &A, "cgab", &B, "gefd", &C, "abcdef");
    int ccnt = 1;
    long long cycles_tuned=0;

    RDTSC(a);
    for(int i=0;i<ccnt;i++){
    tblis_tensor_mult(tblis_single, NULL, &A, "db", &B, "acd", &C, "abc");
//        tblis_tensor_mult(tblis_single, NULL, &A, "cgab", &B, "gefd", &C, "abcdef");
//        tblis_tensor_mult(tblis_single, NULL, &A, "bcag", &B, "fdeg", &C, "abcdef");
//        tblis_tensor_mult(tblis_single, NULL, &A, "gcba", &B, "gfed", &C, "adbecf");
//    tblis_tensor_mult(tblis_single, NULL, &A, "cabg", &B, "efdg", &C, "abcdef");
  

        /* for(int i = 0; i < a4*b4*c4*d4*e4*f4; i++){ */
        /*     data_C2[i]+=1100; */
        /* } */


//        printf ("cnt\n");
    }
        RDTSC(b);
        cycles_tuned += (long)(((double) (COUNTER_DIFF_SIMPLE(b,a))) / ((long long) 1));

    clock_t end = clock();
    double elapsed_secs = (double)(end-begin)/CLOCKS_PER_SEC;
    double flop_count = 2.0*a4*b4*c4*d4*ccnt;

    double flops_cycle_tuned = ((double)flop_count)/((double)cycles_tuned);
    printf("tot cyc= %12ld,  cycles,    %2.4f, flops/cycle\n",cycles_tuned, flops_cycle_tuned);
    printf("GFLOPS = %lf\n", 1.0*flop_count/(end-begin)*CLOCKS_PER_SEC/1000/1000/1000);
    printf("runtime = %lf\n",elapsed_secs);


    compare_fun(data_A, data_B, data_C2);
    printf("fin c2\n");


    for(int i=0; i< a4*b4*c4 ;i++){
        if(data_C[i] != data_C2[i] || data_C[i]*data_C2[i]==0){
            printf("error at C[%d], C %lf, C1 %lf\n", i, data_C[i], data_C2[i]);
                return 1;
        }
    }
    printf("fin compare\n");    
//tblis_init_tensor_d(A, 4.0, (len_type*){10, 9, 2, 5},
//                    data_A, (stride_type*){1, 10, 90, 180});

/* double data_B[7*5*9*8]; */
/* tblis_tensor B; */
/* tblis_init_tensor_d(&B, 4, (len_type*){7, 5, 9, 8}, */
/*                     data_B, (stride_type*){1, 7, 35, 315}); */

/* double data_C[7*2*10*8]; */
/* tblis_tensor C; */
/* tblis_init_tensor_d(&C, 4, (len_type*){7, 2, 10, 8}, */
/*                     data_C, (stride_type*){1, 7, 14, 140}); */

// initialize data_A and data_B...

// this computes C[abcd] += A[cebf] B[afed]
//tblis_tensor_mult(NULL, NULL, &A, "cebf", &B, "afed", &C, "abcd");
    return 0;
}