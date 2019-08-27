#include<stdio.h>
#include<time.h>
#include"tblis.h"
#include "mpi.h"
#include "omp.h"

#include<malloc.h>
#include "rdtsc.h"
#include "papi.h"

void handle_err(int err, int ln){

	if(err!=PAPI_OK)
	{
		printf("papi error %d %d\n",err,ln);
		exit(1);
	}
}
#define handle_error(x) handle_err(x,__LINE__)
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
            A[d *e4*g4*a4 + e *g4*a4 + g* a4+a *1]
            *
            B[g* f4*b4*c4 + f *b4*c4 + b*c4+ c];


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


    int numEvents = 3;
	long long values[3];
	int events=PAPI_NULL;
    PAPI_library_init(PAPI_VER_CURRENT);
    handle_error(PAPI_create_eventset(&events));
	handle_error(PAPI_add_event(events,PAPI_L1_TCM));
	handle_error(PAPI_add_event(events,PAPI_L2_TCM));
	handle_error(PAPI_add_event(events,PAPI_L3_TCM));

    
//cgab, gefd
    double* data_A = (double*)memalign(4096,d4*e4*g4*a4*sizeof(double));
    tblis_tensor A;


    len_type Altp[4] = {d4,e4,g4,a4}; // m*k
//    stride_type Astp[4] = {g4*a4*b4, a4*b4, b4,1};
    //gabc
    stride_type Astp[4] = {e4*g4*a4 , g4*a4 ,a4 ,1};
    len_type* Alentp = &Altp;
    stride_type*  Asttp = &Astp;
    for(int i=0;i<d4*e4*g4*a4;i++)data_A[i] = rand()/1000;
    tblis_init_tensor_d(&A, 4, Alentp, data_A, Asttp);

    
    double* data_B = (double*)memalign(4096,g4*f4*b4*c4*sizeof(double));
    tblis_tensor B;
    len_type Bltp[4] = {g4,f4,b4,c4 };// n*k
//    stride_type Bstp[4] = {e4*f4*d4, f4*d4, d4,1};
    stride_type Bstp[4] = {b4*f4*c4, b4*c4, c4,1};
    len_type* Blentp = &Bltp;
    stride_type*  Bsttp = &Bstp;
    for(int i=0;i<g4*f4*b4*c4;i++)data_B[i] = rand()/1000;
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
    

    for(int i=0;i<10;i++)
    tblis_tensor_mult(tblis_single, NULL, &A, "dega", &B, "gfbc", &C, "abcdef");
    
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
double begin = omp_get_wtime();   
int ccnt = 10;
    long long cycles_tuned=0;

    RDTSC(a);
    for(int i=0;i<ccnt;i++){
        #ifdef RUNPAPI
		handle_error(PAPI_start(events));
#endif
    tblis_tensor_mult(tblis_single, NULL, &A, "dega", &B, "gfbc", &C, "abcdef");
//        tblis_tensor_mult(tblis_single, NULL, &A, "cgab", &B, "gefd", &C, "abcdef");
//        tblis_tensor_mult(tblis_single, NULL, &A, "bcag", &B, "fdeg", &C, "abcdef");
//        tblis_tensor_mult(tblis_single, NULL, &A, "gcba", &B, "gfed", &C, "adbecf");
//    tblis_tensor_mult(tblis_single, NULL, &A, "cabg", &B, "efdg", &C, "abcdef");
  

        /* for(int i = 0; i < a4*b4*c4*d4*e4*f4; i++){ */
        /*     data_C2[i]+=1100; */
        /* } */
#ifdef RUNPAPI
	handle_error( PAPI_stop(events,values)) ;
	printf("event L1 =   %lld\n",values[0]);
	printf("event L2 =   %lld\n",values[1]);
	printf("event L3 =   %lld\n",values[2]);
#endif


//        printf ("cnt\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
        RDTSC(b);
        cycles_tuned += (long)(((double) (COUNTER_DIFF_SIMPLE(b,a))) / ((long long) 1));

    double end = omp_get_wtime();
    double elapsed_secs = (double)(end-begin)/1.0;

double flop_count = 2.0*a4*b4*c4*d4*e4*f4*g4*ccnt;

    double flops_cycle_tuned = ((double)flop_count)/((double)cycles_tuned);
    printf("fpc, %2.4f,", flops_cycle_tuned);
    printf("gflops,  %lf\n", 1.0*flop_count/(end-begin)*1.0/1000/1000/1000);
    printf("runtime = %lf\n",elapsed_secs);
    for(int i=0;i<ccnt;i++)
    compare_fun(data_A, data_B, data_C2);
    printf("fin compare\n");
    for(int i=0; i< a4*b4*c4*d4*e4*f4 ;i++){
        if(data_C[i] != data_C2[i]){
            //printf("error at C[%d], C %lf, C1 %lf\n", i, data_C[i], data_C2[i]);
                return 0;
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