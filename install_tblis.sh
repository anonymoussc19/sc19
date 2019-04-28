#!/bin/bash
module load gcc/5.3.0
git clone https://github.com/devinamatthews/tblis.git
cd tblis
mkdir install
./configure --prefix=YOUR_INSTALL_DIR --disable-thread-model

make && make install