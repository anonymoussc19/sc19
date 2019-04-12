#!/bin/bash
module load gcc/5.3.0
 for i in $( find . -name "abcd-*.c" ); do
     gcc -O3 -fopenmp --std=c99  $i -o "$i.exe"
     echo -ne "$i  ,   " 

 done

for i in $( find . -name "*.exe" ); do
    echo  "$i" 
    ./$i 72 72 72 72 72 72
done

