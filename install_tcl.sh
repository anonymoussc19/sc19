#!/bin/bash

git clone https://github.com/springer13/hptt.git
git clone https://github.com/springer13/tcl.git

cd hptt
make avx
echo $HPTT_ROOT=`pwd`
cd ..
cd tcl
sed -i 's/#BLAS_LIB/BLAS_LIB/g' Makefile
sed -i 's/#INCLUDE_FLAGS/INCLUDE_FLAGS/g' Makefile
sed -i 's/BLAS_LIB=-L${BLIS/#BLAS_LIB=-L${BLIS/g' Makefile

# Change according to you MKL
source /opt/software/intel/parallel_studio_xe_2018_cluster_edition/compilers_and_libraries_2018/linux/bin/compilervars.sh intel64 
make 
echo $TCL_ROOT=`pwd`



