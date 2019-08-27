#!/bin/bash
module load gcc/5.3.0
#module load mvapich2-2.2/gcc
module load openmpi/1.4.3-gcc

 for i in $( find . -name "abcd-*.c" ); do
     mpicc -O3 -fopenmp --std=c99  $i -o "$i.exe"
     echo -ne "$i  ,   " 

 done

for i in $( find . -name "*.exe" ); do
    echo  "$i" 
    mpirun -np 4 ./$i 72 72 72 72 72 72
done

