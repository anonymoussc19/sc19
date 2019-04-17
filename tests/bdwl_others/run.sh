#!/bin/bash

 for i in $( find . -name "a*.c" ); do
     gcc -O3 -std=c99 -mavx2 -fopenmp  $i -o "$i.exe"
     echo -ne "$i  ,   " 

 done

for i in $( find . -name "*.exe" ); do
    echo  "$i" 
    ./$i 
done

