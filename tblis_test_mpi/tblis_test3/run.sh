#!/bin/bash
module load gcc/7.3.0
module load mvapich2-2.2/gcc
export TBLIS_ROOT=/home/li23/tblis/install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TBLIS_ROOT
export MV2_SMP_USE_CMA=0
for i in $( find . -name "*.c" ); do
    mpicc -O3 -std=c99 -I $TBLIS_ROOT/include/ -I /home/li23/majorrev/sc19/ -I $TBLIS_ROOT/include/tblis/ -L $TBLIS_ROOT/lib/ $TBLIS_ROOT/lib/libtblis.so $i -fopenmp
    echo  "$i  ,   "
    for j in 1 2 4 8 14 16 28;
    do
        echo " "
        echo " "
        echo  "xxxxxxxx    $i, np $j"
        mpirun -np $j ./a.out
        echo " "
    done

    

done