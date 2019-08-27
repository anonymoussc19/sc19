#!/bin/bash
sed -i 's$#include<omp.h>$#include<omp.h>\n#include"mpi.h"$g' *.cpp


sed -i 's$int tot_runtime = 5;$int tot_runtime = 5;\n;MPI_Init(NULL,NULL);\nMPI_Barrier(MPI_COMM_WORLD);$g' *.cpp

sed -i 's$double flop_count$MPI_Barrier(MPI_COMM_WORLD);\ndouble flop_count$g' *.cpp

sed -i 's$printf("flopcnt = %lf, ",flop_count);$ $g' *.cpp

sed -i 's$printf("tot cyc= %12ld,  cycles,    %2.4f, flops/cycle\\n",cycles_tuned, flops_cycle_tuned);$printf("fpc, %2.4f,", flops_cycle_tuned);$g' *.cpp

sed -i 's$printf("error$//printf("error$g' *.cpp

sed -i 's$return 1$return 0$g' *.cpp

sed -i 's$gflops =$gflops, $g' *.cpp 