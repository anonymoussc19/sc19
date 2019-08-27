#!/bin/bash
module load gcc/5.3.0
#module load mvapich2-2.2/gcc
module load openmpi/1.4.3-gcc
#export MV2_SMP_USE_CMA=0
 for i in $( find . -name "a*.c" ); do
     mpicc -O3 -std=c99 -mavx2 -fopenmp  $i -o "$i.exe"
     echo -ne "$i  ,   " 

 done

for i in $( find . -name "*.exe" ); do
    echo  "$i" 
    mpirun -np 4 ./$i 
done

