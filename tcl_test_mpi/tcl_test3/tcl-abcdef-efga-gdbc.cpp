/*
 *   Copyright (C) 2017  Paul Springer (springer@aices.rwth-aachen.de)
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU Lesser General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <malloc.h>
#include <memory>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <tcl.h>
#include "rdtsc.h"
 #include<omp.h>
#include"mpi.h"

int B_range_a= 24;
int A_range_b= 16;
int A_range_c= 16;
int A_range_d = 24;
int B_range_e= 16;
int B_range_f= 16;

int A_range_g= 24;
int B_range_g= 24;
int C_range_g= 24;

int C_range_a= 24;
int C_range_b= 16;
int C_range_c= 16;
int C_range_d = 24;
int C_range_e= 16;
int C_range_f= 16;


void compare_fun(double* A, double*B,double*C ){
for(int a=0; a<C_range_a; a++)
for(int b=0; b<C_range_b; b++)
for(int c=0; c<C_range_c; c++)
for(int d=0; d<C_range_d; d++)
for(int e=0; e<C_range_e; e++)
for(int f=0; f<C_range_f; f++)
for(int g=0; g<C_range_g; g++)
C[+a* C_range_b* C_range_c* C_range_d* C_range_e* C_range_f+b* C_range_c* C_range_d* C_range_e* C_range_f+c* C_range_d* C_range_e* C_range_f+d* C_range_e* C_range_f+e* C_range_f+f] += 
B[+g* C_range_d* C_range_b* C_range_c+d* C_range_b* C_range_c+b* C_range_c+c] * 
A[+e* C_range_f* C_range_g* C_range_a+f* C_range_g* C_range_a+g* C_range_a+a]; 
}


int main(int argc, char** argv)
{
   tcl::sizeType ta = 24;
   tcl::sizeType tb = 16;
   tcl::sizeType tc = 16;
   tcl::sizeType td = 24;
   tcl::sizeType te = 16;
   tcl::sizeType tf = 16;
   tcl::sizeType tg = 24;

   double *dataA, *dataB, *dataC, *C1;
   posix_memalign((void**) &dataA, 4096, sizeof(double) * ((size_t)te)*tf*tg*ta);
   posix_memalign((void**) &dataB, 4096, sizeof(double) * ((size_t)tg)*td*tb*tc);
   posix_memalign((void**) &dataC, 4096, sizeof(double) * ((size_t)ta)*tb*tc*td*te*tf);
   C1 = (double*)memalign( 4096, sizeof(double) * ((size_t)ta)*tb*tc*td*te*tf);

   // Initialize tensors (data is not owned by the tensors)
   tcl::Tensor<double> A({ta,tg,tf,te}, dataA);
   tcl::Tensor<double> B({tc,tb,td,tg}, dataB);
   tcl::Tensor<double> C({tf,te,td,tc,tb,ta}, dataC);

   // Data initialization
//#pragma omp parallel for
   for(int i=0; i < A.getTotalSize(); ++i)
   dataA[i] = rand()% 1000;
//#pragma omp parallel for
   for(int i=0; i < B.getTotalSize(); ++i)
   dataB[i] = rand()% 1000;
//#pragma omp parallel for
   for(int i=0; i < C.getTotalSize(); ++i)
   dataC[i] = C1[i]=rand()% 1000;

   double alpha = 1.0;
   double beta = 1.0;
     static tsc_counter atsc,btsc;
    CPUID(); RDTSC(atsc); CPUID(); RDTSC(btsc);
    /* warm up */
    CPUID(); RDTSC(atsc); CPUID(); RDTSC(btsc);
 /* warm up */
    CPUID(); RDTSC(atsc); CPUID(); RDTSC(btsc);
    int tot_runtime = 5;
;MPI_Init(NULL,NULL);
MPI_Barrier(MPI_COMM_WORLD);  double start = omp_get_wtime();
    RDTSC(atsc);
    for(int ccnt = 0; ccnt < tot_runtime; ccnt++){
   // tensor contarction: C_{m,n} = alpha * A_{k2,m,k1} * B_{n,k2,k1} + beta * C_{m,n}
   auto err = tcl::tensorMult<double>( alpha, A["ta,tg,tf,te"], B["tc,tb,td,tg"], beta, C["tf,te,td,tc,tb,ta"] );
      if( err != tcl::SUCCESS ){
      printf("ERROR: %s\n", tcl::getErrorString(err));
      exit(-1);
   }
    }

   MPI_Barrier(MPI_COMM_WORLD);
   RDTSC(btsc);
double flop_count = 2.0*ta*tb*tc*td*te*tf*tg * tot_runtime;

   long long cycles_tuned = (long)(((double) (COUNTER_DIFF_SIMPLE(btsc,atsc))) / ((long long) 1));double  runtime = omp_get_wtime() - start;
     
    double flops_cycle_tuned = ((double)flop_count)/((double)cycles_tuned);
    printf("fpc, %2.4f,", flops_cycle_tuned);printf("gflops,  %lf     ,,,", flop_count/1000/1000/1000/runtime);

   


   // for(int i=0; i < m; ++i){
   //    for(int j=0; j < n; ++j)
   //       std::cout<< dataC[j * m + i] << " ";
   //    std::cout<< "\n";
   // }

      for(int i=0;i<tot_runtime;i++)
      compare_fun(dataA, dataB, C1);

    for(int i=0; i< C_range_a * C_range_b *C_range_c
            *C_range_d * C_range_e * C_range_f ;i++){
//        if(C[i] != C1[i]){
        if( dataC[i] != C1[i]){
            //printf("error at C[%d], C %lf, C1 %lf\n", i, dataC[i], C1[i]);
                return 0;
        }
    }
    printf("correct\n");
   return 0;
}
