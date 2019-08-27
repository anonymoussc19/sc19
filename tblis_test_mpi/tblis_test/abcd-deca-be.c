#include<stdio.h>
#include<time.h>
#include"tblis.h"
#include "mpi.h"
#include<malloc.h>
#include "rdtsc.h"
#include <omp.h>


int B_range_a = 72;
int A_range_b = 24;
int B_range_c = 72;
int B_range_d = 72;

int A_range_e = 72;
int B_range_e = 72;

int C_range_a = 72;
int C_range_b = 24;
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
A[+b* C_range_e+e] * 
B[+d* C_range_e* C_range_c* C_range_a+e* C_range_c* C_range_a+c* C_range_a+a]; 
}


int main(int argc, char** argv)
{
//    int m=3072, n=3072, k=3072;
    /* int m,n,k; */
    /* m = atoi(argv[1]); */
    /* n = atoi(argv[2]); */
    /* k = atoi(argv[3]); */
    /* A_range_a = C_range_a = atoi(argv[1]); */
    /* A_range_b = C_range_b = atoi(argv[2]); */
    /* B_range_c = C_range_c = atoi(argv[3]); */
    /* C_range_d =     A_range_d = B_range_d = atoi(argv[4]); */

    int a4,b4,c4,d4,e4;
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
    e4 = C_range_e;
    
//cgab, gefd
    double* data_A = (double*)memalign(4096,b4*e4*sizeof(double));
    tblis_tensor A;


    len_type Altp[2] = {b4, e4}; // m*k
    stride_type Astp[2] = {   e4, 1};
    len_type* Alentp = &Altp;

    stride_type*  Asttp = &Astp;
    for(int i=0;i<b4*e4;i++)data_A[i] = rand()%1000;


    tblis_init_tensor_d(&A, 2, Alentp, data_A, Asttp);

    double* data_B = (double*)memalign(4096,a4*e4*c4*d4*sizeof(double));
    tblis_tensor B;
    len_type Bltp[4] = {d4,e4,c4,a4 };// n*k

    
//    stride_type Bstp[4] = {e4*f4*d4, f4*d4, d4,1};
    stride_type Bstp[4] = {e4*c4*a4, c4*a4,a4,1};
    len_type* Blentp = &Bltp;
    stride_type*  Bsttp = &Bstp;
    for(int i=0;i<d4*c4*e4*a4;i++)data_B[i] = rand()%1000;
    tblis_init_tensor_d(&B, 4, Blentp, data_B, Bsttp);

    double* data_C = (double*)memalign(4096,a4*b4*c4*d4*sizeof(double));
    double* data_C2 = (double*)memalign(4096,a4*b4*c4*d4*sizeof(double));
    for(int i = 0; i < a4*b4*c4*d4; i++){
        data_C[i] = data_C2[i]=0;
    }

    tblis_tensor C;
    len_type Cltp[4] = {a4,b4,c4,d4}; // m*n
//    stride_type Cstp[6] = {24*24*24*24*24, 24*24*24*24,24*24*24,24*24,24,1};
    stride_type Cstp[4] = {b4*c4*d4,c4*d4,d4, 1};
    len_type* Clentp = &Cltp;
    stride_type*  Csttp =&Cstp;

    tblis_init_tensor_d(&C, 4, Clentp, data_C, Csttp);
//    tblis_tensor_mult(NULL, NULL, &A, "cgab", &B, "gefd", &C, "abcdef");
//    data_A[0] = rand()/1000;

    //tblis_tensor_mult(NULL, NULL, &A, "cgab", &B, "gefd", &C, "abcdef");
    
    for(int i = 0; i < a4*b4*c4; i++){
        data_C2[i]=0;
    }
//        tblis_tensor_mult(tblis_single, NULL, &A, "bda", &B, "dc", &C, "abc");
    double start = omp_get_wtime();   
    static tsc_counter a,b;
                // timing
    /* warm up, according to Intel manual */
    CPUID(); RDTSC(a); CPUID(); RDTSC(b);
    /* warm up */
    CPUID(); RDTSC(a); CPUID(); RDTSC(b);
    /* warm up */
    CPUID(); RDTSC(a); CPUID(); RDTSC(b);
//    tblis_tensor_mult(tblis_single, NULL, &A, "cgab", &B, "gefd", &C, "abcdef");
    MPI_Init(NULL,NULL);
MPI_Barrier(MPI_COMM_WORLD);
int ccnt = 5;
start = omp_get_wtime();

    long long cycles_tuned=0;

    RDTSC(a);
    for(int i=0;i<ccnt;i++){
    tblis_tensor_mult(tblis_single, NULL, &A, "be", &B, "deca", &C, "abcd");
//        tblis_tensor_mult(tblis_single, NULL, &A, "cgab", &B, "gefd", &C, "abcdef");
//        tblis_tensor_mult(tblis_single, NULL, &A, "bcag", &B, "fdeg", &C, "abcdef");
//        tblis_tensor_mult(tblis_single, NULL, &A, "gcba", &B, "gfed", &C, "adbecf");
//    tblis_tensor_mult(tblis_single, NULL, &A, "cabg", &B, "efdg", &C, "abcdef");
  

        /* for(int i = 0; i < a4*b4*c4*d4*e4*f4; i++){ */
        /*     data_C2[i]+=1100; */
        /* } */


//        printf ("cnt\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
        RDTSC(b);
        cycles_tuned += (long)(((double) (COUNTER_DIFF_SIMPLE(b,a))) / ((long long) 1));

    double end = omp_get_wtime();
    double elapsed_secs = (double)(end-start)/1.0;

double flop_count = 2.0*a4*b4*c4*d4*e4*ccnt;

    double flops_cycle_tuned = ((double)flop_count)/((double)cycles_tuned);
    printf("fpc, %2.4f,", flops_cycle_tuned);
    printf("gflops,  %lf\n", 1.0*flop_count/(end-start)*1.0/1000/1000/1000);
    printf("runtime = %lf\n",elapsed_secs);

    for(int i=0; i<ccnt;i++)
    compare_fun(data_A, data_B, data_C2);
    printf("fin c2\n");


    for(int i=0; i< a4*b4*c4*d4 ;i++){
        if(data_C[i] != data_C2[i] || data_C[i]*data_C2[i]==0){
            //printf("error at C[%d], C %lf, C1 %lf\n", i, data_C[i], data_C2[i]);
                return 0;
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