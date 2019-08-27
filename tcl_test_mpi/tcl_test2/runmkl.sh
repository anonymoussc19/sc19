#!/bin/bash
export MV2_SMP_USE_CMA=0
#change according to your mkl.
module load gcc/5.3.0
module load mvapich2-2.2/gcc
#source /opt/software/intel/parallel_studio_xe_2018_cluster_edition/compilers_and_libraries_2018/linux/bin/compilervars.sh intel64
#source /opt/intel/compilers_and_libraries_2018/linux/bin/compilervars.sh intel64
source /opt/intel/compilers_and_libraries_2019.0.117/linux/bin/compilervars.sh intel64
export OMP_NUM_THREADS=1
#export HPTT_ROOT= #YOUR HPTT INSTALLATION DIR
#export TCL_ROOT= #YOUR TCL INSTALLATION DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HPTT_ROOT/lib:$TCL_ROOT/lib

for i in $( find . -name "*.cpp" ); do
    echo  "$i, "
    echo " 0000000 "
    mpicxx $i -O0 -I $TCL_ROOT/include/ -g --std=c++11   -L $HPTT_ROOT/lib -lhptt -L $TCL_ROOT/lib/ -ltcl   -fopenmp   -lpthread -lm -ldl

    for j in 1 2 4 8 14 16 28;
    do
        echo " "
        echo " "
        echo  "xxxxxxxx    $i, np $j"
        mpirun -np $j ./a.out
        echo " "
    done
        

done