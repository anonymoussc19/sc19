#!/bin/bash



sed -i 's%#include<omp.h>%#include<omp.h>\n#include"mpi.h"%g' *.c
sed -i 's#tot_runtime = 1;#tot_runtime = 5;#g' *.c
sed -i 's#tot_runtime = 5;#tot_runtime = 5;\n MPI_Init(NULL,NULL);\n  MPI_Barrier(MPI_COMM_WORLD);\n#g' *.c
sed -i 's#end ccnt#end ccnt\n MPI_Barrier(MPI_COMM_WORLD);#g' *.c 
