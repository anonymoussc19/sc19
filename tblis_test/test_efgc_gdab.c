#include<stdio.h>
#include<time.h>
#include"tblis.h"
#include<malloc.h>
#include "rdtsc.h"
int a4 = 24;
int b4 = 16;
int c4 = 16;
int d4 = 24;
int e4 = 16;
int f4 = 16;
int g4 = 24;

void compare_fun(double* A, double*B,double*C ){


//    a4=c4=19; f4=d4=;
    for(int a = 0; a<a4;a++)
    for(int b = 0; b<b4;b++)
    for(int c = 0; c<c4;c++)
    for(int d = 0; d<d4;d++)
    for(int e = 0; e<e4;e++)
    for(int f = 0; f<f4;f++){
    for(int g = 0; g<g4;g++){
        //cgab A,   gefd B
        C[a* b4*c4*d4*e4*f4 +
          b*  c4*d4*e4*f4 +
          c * d4*e4*f4 +
          d * e4*f4+
          e * f4+
          f]  +=
//dega gfbc
            A[e *f4*g4*c4 + f *g4*c4 + g* c4+c *1]
            *
            B[g* d4*a4*b4 + d *a4*b4 + a*b4+ b];


    }

    }
}

int main(int argc, char** argv)
{
//    int m=3072, n=3072, k=3072;
    /* int m,n,k; */
    /* m = atoi(argv[1]); */
    /* n = atoi(argv[2]); */
    /* k = atoi(argv[3]); */



    

    double* data_A = (double*)memalign(4096,e4*f4*g4*c4*sizeof(double));
    tblis_tensor A;
    len_type Altp[4] = {e4,f4,g4,c4}; // m*k
    stride_type Astp[4] = {f4*g4*c4 , g4*c4 ,c4 ,1};
    len_type* Alentp = &Altp;
    stride_type*  Asttp = &Astp;
    for(int i=0;i<e4*f4*g4*c4;i++)data_A[i] = rand()/1000;
    tblis_init_tensor_d(&A, 4, Alentp, data_A, Asttp);




    
    double* data_B = (double*)memalign(4096,g4*d4*a4*b4*sizeof(double));
    tblis_tensor B;
    len_type Bltp[4] = {g4,d4,a4,b4 };// n*k
    stride_type Bstp[4] = {a4*d4*b4, a4*b4, b4,1};
    len_type* Blentp = &Bltp;
    stride_type*  Bsttp = &Bstp;
    for(int i=0;i<g4*d4*a4*b4;i++)data_B[i] = rand()/1000;
    tblis_init_tensor_d(&B, 4, Blentp, data_B, Bsttp);



    
    double* data_C = (double*)memalign(4096,a4*b4*c4*d4*e4*f4*sizeof(double));
    double* data_C2 = (double*)memalign(4096,a4*b4*c4*d4*e4*f4*sizeof(double));
    for(int i = 0; i < a4*b4*c4*d4*e4*f4; i++){
        data_C[i] = data_C2[i]=rand()%1000;
    }

    tblis_tensor C;
    len_type Cltp[6] = {a4,b4,c4,d4,e4,f4}; // m*n
//    stride_type Cstp[6] = {24*24*24*24*24, 24*24*24*24,24*24*24,24*24,24,1};
    stride_type Cstp[6] = {b4*c4*d4*e4*f4,c4*d4*e4*f4, d4*e4*f4,e4*f4, f4 ,1};
    len_type* Clentp = &Cltp;
    stride_type*  Csttp =&Cstp;

    tblis_init_tensor_d(&C, 6, Clentp, data_C, Csttp);
//    tblis_tensor_mult(NULL, NULL, &A, "cgab", &B, "gefd", &C, "abcdef");
//    data_A[0] = rand()/1000;

    //tblis_tensor_mult(NULL, NULL, &A, "cgab", &B, "gefd", &C, "abcdef");
    

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
    int ccnt = 5;
    long long cycles_tuned=0;

    RDTSC(a);
    for(int i=0;i<ccnt;i++){
    tblis_tensor_mult(tblis_single, NULL, &A, "efgc", &B, "gdab", &C, "abcdef");
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
    double flop_count = 2.0*a4*b4*c4*d4*e4*f4*g4*ccnt;

    double flops_cycle_tuned = ((double)flop_count)/((double)cycles_tuned);
    printf("tot cyc= %12ld,  cycles,    %2.4f, flops/cycle\n",cycles_tuned, flops_cycle_tuned);
    printf("GFLOPS = %lf\n", 1.0*flop_count/(end-begin)*CLOCKS_PER_SEC/1000/1000/1000);
    printf("runtime = %lf\n",elapsed_secs);
    for(int i=0;i<ccnt;i++)
    compare_fun(data_A, data_B, data_C2);
    printf("fin compare\n");
    for(int i=0; i< a4*b4*c4*d4*e4*f4 ;i++){
        if(data_C[i] != data_C2[i]){
            printf("error at C[%d], C %lf, C1 %lf\n", i, data_C[i], data_C2[i]);
                return 1;
        }
    }
    
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