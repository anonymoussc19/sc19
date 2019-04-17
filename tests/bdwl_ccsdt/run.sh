#!/bin/bash

 for i in $( find . -name "abcdef*.c" ); do
     gcc -O3 -std=c99 -mavx2 -fopenmp  $i -o "$i.exe"
     echo -ne "$i  ,   " 

 done

for i in $( find . -name "*.exe" ); do
    echo  "$i" 
    ./$i 24 16 16 24 16 16 24
done

