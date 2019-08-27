module load gcc/9.1.0
gcc -mavx2 -O3 -std=c99 -fopenmp   -DMYTHREAD=$mythread ./omp-degb-split.c
for i in 1 2 3 4 5; do
    ./a.out 24 16 16 24 16 16 24
done