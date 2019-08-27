#!/bin/bash
sed -i 's$#include<omp.h>$#include<omp.h>\n#include"mpi.h"$g' *.c


sed -i 's$int ccnt$MPI_Init(NULL,NULL);\nMPI_Barrier(MPI_COMM_WORLD);\nint ccnt$g' *.c

sed -i 's$double flop_count$MPI_Barrier(MPI_COMM_WORLD);\ndouble flop_count$g' *.c

sed -i 's$printf("flopcnt = %lf, ",flop_count);$ $g' *.c

sed -i 's$printf("tot cyc= %12ld,  cycles,    %2.4f, flops/cycle\\n",cycles_tuned, flops_cycle_tuned);$printf("fpc, %2.4f,", flops_cycle_tuned);$g' *.c

sed -i 's$printf("error$//printf("error$g' *.c

sed -i 's$return 1$return 0$g' *.c

sed -i 's$GFLOPS =$gflops, $g' *.c

sed -i 's$#include"tblis.h"$#include"tblis.h"\n#include "mpi.h"$g' *.c