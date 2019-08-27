#!/bin/bash
module load gcc/7.3.0
module load mvapich2-2.2/gcc
export MV2_SMP_USE_CMA=0
 for i in $( find . -name "abcdef*.c" ); do
     mpicc -O3 -std=c99 -mavx2 -fopenmp -I /home/li23/majorrev/sc19/  $i -o "$i.exe"
     echo -ne "$i  ,   " 

 done
echo "start"

for i in $( find . -name "*.exe" ); do
    for j in 1  14  28
    do
        echo " "
        echo " "
        echo  "00000000    $i, np $j"
        mpirun -np $j ./$i 24 16 16 24 16 16 24
    done
done

