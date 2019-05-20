#!/bin/bash
#change according to your mkl.
source /opt/software/intel/parallel_studio_xe_2018_cluster_edition/compilers_and_libraries_2018/linux/bin/compilervars.sh intel64
export OMP_NUM_THREADS=1
export HPTT_ROOT= #YOUR HPTT INSTALLATION DIR
export TCL_ROOT= #YOUR TCL INSTALLATION DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HPTT_ROOT/lib:$TCL_ROOT/lib

for i in $( find . -name "*.cpp" ); do
    echo -ne "$i, "
    g++ $i -O0 -I $TCL_ROOT/include/ -g --std=c++11   -L $HPTT_ROOT/lib -lhptt -L $TCL_ROOT/lib/ -ltcl -fopenmp
    ./a.out
done