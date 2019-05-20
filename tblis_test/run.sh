module load gcc/5.3.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TBLIS_ROOT

for i in $( find . -name "*.c" ); do
    gcc -O3 -std=c99 -I $TBLIS_ROOT/include/ -I $TBLIS_ROOT/include/tblis/ -L $TBLIS_ROOT/lib/ $TBLIS_ROOT/lib/libtblis.so $i
    echo -ne "$i  ,   " 
    ./a.out
done