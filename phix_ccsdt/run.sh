#!/bin/bash
module load gcc/7.3.0
 for i in $( find . -name "abcdef*.c" ); do
     gcc -O3 -fopenmp  $i -o "$i.exe"
     echo -ne "$i  ,   " 

 done

for i in $( find . -name "*.exe" ); do
    echo  "$i" 
    ./$i 24 16 16 24 16 16 24
done

