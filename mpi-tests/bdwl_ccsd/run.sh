#!/bin/bash
module load gcc/7.3.0
module load mvapich2-2.2/gcc
export MV2_SMP_USE_CMA=0
 for i in $( find . -name "abcd-*.c" ); do
     mpicc -O3 -fopenmp --std=c99  $i  -I /home/li23/majorrev/sc19/ -o "$i.exe"
     echo -ne "$i  ,   " 

 done
echo "start"
for i in $( find . -name "*.exe" ); do
    for j in  28
    do
        echo " "
        echo " "
        echo  "00000000    $i, np $j"

        mpirun -np $j ./$i 72 72 72 72 72 72
    done
done

