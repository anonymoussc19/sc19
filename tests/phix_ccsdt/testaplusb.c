#include "papi.h"
#include<malloc.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
void handle_error(int err){
    exit(1);
    printf("papi error\n");
}

int main(int argc, char** argv){

    int N = 1000000;

    double * A = (double*)memalign(4096, sizeof(double)*N);
    double * B = (double*)memalign(4096, sizeof(double)*N);
    double * C = (double*)memalign(4096, sizeof(double)*N);

    for(int i=0;i<N; i++ ){
        A[i] = rand()%1000;
        B[i] = rand()%1000;
        C[i] = rand()%1000;
    }


    int numEvents = 1;
    long long values[1];
    int events[1] ;
    events[0] = 0x80000000 + atoi(argv[0]);
    
        if (PAPI_start_counters(events, numEvents) != PAPI_OK)
    handle_error(1);        
        for(int i=0;i<N; i++ ){
            double tmp = A[i] + B[i];
            if(tmp == 42.42){
                C[i]= tmp;
            }
        }
    if ( PAPI_stop_counters(values, numEvents) != PAPI_OK)
    handle_error(1);


    
    printf("event 0x%08x\n =   %d\n",events[0],values[0]);

    for(int i=0;i<N; i++ )
    C[0]+= C[i];
    
    return C[0];
}
