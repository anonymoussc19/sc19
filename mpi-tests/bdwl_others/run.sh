#!/bin/bash
export MV2_SMP_USE_CMA=0
module load gcc/7.3.0
module load mvapich2-2.2/gcc
for i in $( find . -name "a*.c" ); do
    mpicc -O3 -std=c99 -mavx2 -fopenmp -I /home/li23/majorrev/sc19/  $i -o "$i.exe"
     echo -ne "$i  ,   " 

 done

for i in $( find . -name "*.exe" ); do
    for j in 1  14  28
    do
        echo " "
        echo " "
        echo  "00000000    $i, np $j"
        echo  "$i" 
        mpirun -np $j ./$i
    done
done

