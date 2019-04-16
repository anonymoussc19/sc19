#include "blis.h"
#include <time.h>
#include <stdlib.h>
#include<stdio.h>
//#include "papi.h"
#include <malloc.h>
//#include <cstdlib>
#include <time.h>
#include<math.h>
#define BLIS_ASM_SYNTAX_ATT
#define kkk 24

#define MIN(a,b) ((a) < (b) ? (a) : (b))

#include "bli_x86_asm_macros.h"
#include "rdtsc.h"

int main(){
    int a0,b0,c0,d0,e0,f0,g0;
    int a1,b1,c1,d1,e1,f1,g1;
    int a2,b2,c2,d2,e2,f2,g2;
    int a3,b3,c3,d3,e3,f3,g3;
    int a4,b4,c4,d4,e4,f4,g4;
    int packcntA=0;
    int packcntB = 0;

    for( d4 = 0; d4 < 0 + 24; d4 += 24)
    for( e4 = 0; e4 < 0 + 24; e4 += 24)
    for( f4 = 0; f4 < 0 + 24; f4 += 24)
    for( g4 = 0; g4 < 0 + 24; g4 += 24)
    for( a4 = 0; a4 < 0 + 24; a4 += 6)
    for( b4 = 0; b4 < 0 + 24; b4 += 1)
    for( c4 = 0; c4 < 0 + 24; c4 += 1)
    
    for( g3 = g4; g3 < g4 + 24; g3 += 24)
    for( a3 = a4; a3 < a4 + 6; a3 += 6)
    for( b3 = b4; b3 < b4 + 1; b3 += 1)
    for( c3 = c4; c3 < c4 + 1; c3 += 1)
    for( d3 = d4; d3 < d4 + 24; d3 += 2)
    for( e3 = e4; e3 < e4 + 24; e3 += 2)
    for( f3 = f4; f3 < f4 + 24; f3 += 8)
    
    for( d2 = d3; d2 < d3 + 2; d2 += 2)
    for( e2 = e3; e2 < e3 + 2; e2 += 2)
    for( f2 = f3; f2 < f3 + 8; f2 += 8)
    for( g2 = g3; g2 < g3 + 24; g2 += 24)
    for( a2 = a3; a2 < a3 + 6; a2 += 6)
    for( b2 = b3; b2 < b3 + 1; b2 += 1)
    for( c2 = c3; c2 < c3 + 1; c2 += 1)
    
    for( b1 = b2; b1 <MIN(24, b2 + 1); b1 += 1)
    for( c1 = c2; c1 <MIN(24, c2 + 1); c1 += 1)
    for( a1 = a2; a1 <MIN(24, a2 + 6); a1 += 6)
    for( d1 = d2; d1 < MIN(24, d2 + 2); d1 += 1)
    for( e1 = e2; e1 < MIN(24, e2 + 2); e1 += 1)
    for( f1 = f2; f1 < MIN(24, f2 + 8); f1 += 8) 
// for( g1 = g2; g1 < g2 + 24; g1 += 1) 
 {

if(a1 + 6 < 24 || b1+8 < 24){// mr < 6 or nr < 8
Cmrnr_tmp = memalign(256, 6*8*sizeof(double));
// A B has been padde
     int offsetA =  + g4*24*24*24 + a4*24*24*24 + b4*6*24*24 + c4*6*1*24 + (g3-g4)*6*1*1 + (a3-a4)*1*1*24 + (b3-b4)*6*1*24 + (c3-c4)*6*1*24 + (g2-g3)*6*1*1 + (a2-a3)*1*1*24 + (b2-b3)*6*1*24 + (c2-c3)*6*1*24 +(b1-b2)*6*1*24 + (c1-c2)*6*24 + (a1-a2)*24;
//     printf("offA: %d\n", offsetA);
     int offsetB =  + d4*24*24*24 + e4*24*24*24 + f4*24*24*24 + g4*24*24*24 + (g3-g4)*24*24*24 + (d3-d4)*24*24*24 + (e3-e4)*2*24*24 + (f3-f4)*2*2*24 + (d2-d3)*2*8*24 + (e2-e3)*2*8*24 + (f2-f3)*2*2*24 + (g2-g3)*2*2*8 + (d1-d2)*2*8*24 + (e1-e2)*8*24 + (f1-f2)*24;


}
else{
     int offsetA =  + g4*24*24*24 + a4*24*24*24 + b4*6*24*24 + c4*6*1*24 + (g3-g4)*6*1*1 + (a3-a4)*1*1*24 + (b3-b4)*6*1*24 + (c3-c4)*6*1*24 + (g2-g3)*6*1*1 + (a2-a3)*1*1*24 + (b2-b3)*6*1*24 + (c2-c3)*6*1*24 +(b1-b2)*6*1*24 + (c1-c2)*6*24 + (a1-a2)*24;
//     printf("offA: %d\n", offsetA);
     int offsetB =  + d4*24*24*24 + e4*24*24*24 + f4*24*24*24 + g4*24*24*24 + (g3-g4)*24*24*24 + (d3-d4)*24*24*24 + (e3-e4)*2*24*24 + (f3-f4)*2*2*24 + (d2-d3)*2*8*24 + (e2-e3)*2*8*24 + (f2-f3)*2*2*24 + (g2-g3)*2*2*8 + (d1-d2)*2*8*24 + (e1-e2)*8*24 + (f1-f2)*24;

     bli_dgemm_haswell_asm_6x8(
         kkk, &alpha, Abuf+offsetA, Bbuf+offsetB,&beta,
         C + a1 * 24*24*24*24*24 +
         b1 * 24*24*24*24 +
         c1 * 24*24*24 +
         d1 * 24*24 + 
         e1 * 24 + f1,
// a b c d e f
         24*24*24*24*24, 1,   // row stride(a), col stride(e)
         NULL,NULL);
}
 }
    return 0;
}