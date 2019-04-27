#!/bin/bash
#change module load according to your environment
module load gcc/5.3.0
module load cmake/3.9.0
wget https://ampl.com/demo/ampl.linux64.tgz
tar -zxf ampl.linux64.tgz
export PATH=$PATH:`pwd`/ampl.linux64
wget https://ampl.com/dl/open/couenne/couenne-linux64.zip
unzip couenne-linux64.zip
mv couenne ./ampl.linux64

cd ./tool
mkdir build
cd build 
cmake ../src
make 

mkdir bdwl_tiles
mkdir skyl_tiles

./Tool ../src/tileinput.txt ccsd broadwell > bdwl_tiles/ccsd.out
./Tool ../src/tileinput.txt ccsdt broadwell > bdwl_tiles/ccsdt.out

./Tool ../src/tileinput.txt abc-ad-312 broadwell > bdwl_tiles/abc-ad-312.out
./Tool ../src/tileinput.txt ab312 broadwell > bdwl_tiles/ab312.out

./Tool ../src/tileinput.txt abcd-ea broadwell > bdwl_tiles/abcd-ea.out
./Tool ../src/tileinput.txt abc-dc-24 broadwell > bdwl_tiles/abc-dc-24.out

./Tool ../src/tileinput.txt abcd-ebad broadwell > bdwl_tiles/abcd-ebad.out
./Tool ../src/tileinput.txt abcde broadwell > bdwl_tiles/abcde.out



./Tool ../src/tileinput.txt ccsd skylake > skyl_tiles/ccsd.out
./Tool ../src/tileinput.txt ccsdt skylake > skyl_tiles/ccsdt.out

./Tool ../src/tileinput.txt abc-ad-312 skylake > skyl_tiles/abc-ad-312.out
./Tool ../src/tileinput.txt ab312 skylake > skyl_tiles/ab312.out

./Tool ../src/tileinput.txt abcd-ea skylake > skyl_tiles/abcd-ea.out
./Tool ../src/tileinput.txt abc-dc-24 skylake > skyl_tiles/abc-dc-24.out

./Tool ../src/tileinput.txt abcd-ebad skylake > skyl_tiles/abcd-ebad.out
./Tool ../src/tileinput.txt abcde skylake > skyl_tiles/abcde.out



