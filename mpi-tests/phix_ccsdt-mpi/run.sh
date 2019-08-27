#!/bin/bash

module load gcc/5.3.0
#module load mvapich2-2.2/gcc
module load openmpi/1.4.3-gcc

 for i in $( find . -name "abcdef*.c" ); do
     mpicc -O3 -fopenmp  $i -o "$i.exe"
     echo -ne "$i  ,   " 

 done

for i in $( find . -name "*.exe" ); do
    echo  "$i" 
    mpirun -np 4 ./$i 24 16 16 24 16 16 24
done

