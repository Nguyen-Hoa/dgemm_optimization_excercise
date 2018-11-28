#include "xmmintrin.h"
#include "stdio.h"
#include "string.h"
#include "stdlib.h"
/*
 works cited

matrixcalc.org, because hand calculating matrix products is too long

SSE examples
  zybooks, Computer Organization and Design; chapters 3, 4, 5, 6
  https://www.scss.tcd.ie/David.Gregg/cs3014/notes/lecture16-sse1.pdf

Loop Unrolling examples
  https://github.com/WillCh/cs267/blob/master/hw1/hw1/dgemm-sse.c
*/

void dbg_printmat(int m, int n, float* D);

void dgemm( int m, int n, float *A, float *C )
{

    /* consider padding matrix if dimmensions are not optimal */
    int new_m = m;
    while(new_m % 16 != 0){
        new_m++;
    }

   int new_n = n;
   while(new_n % 4 != 0){
       new_n++;
   }

    float *result = (float*) malloc( new_m * sizeof(float) ); //temporary matrix to store results, size of one row
    float *Apad = (float*) malloc (new_m * new_n * (sizeof(float)) ); //padded matrix
    //float *ATpad = (float*) malloc (new_m * new_n * (sizeof(float)) ); //transposed padded matrix
    memset(result, 0, sizeof(float) * new_m);
    memset(Apad, 0, new_m * new_n * sizeof(float) );
    //memset(ATpad, 0, new_n * new_m * sizeof(float) );

    for(int icpy = 0; icpy < n; icpy++){
        memcpy(&Apad[icpy*new_m], &A[icpy*m], sizeof(float) * m);
    }

//    for(int icpy = 0; icpy < n; icpy++){
//        memcpy(&result[0], &A[icpy*m], sizeof(float) * m); //copy row of A[] into result
//        for(int ycpy = 0; ycpy < m; ycpy++){
//            memcpy(&ATpad[icpy+ycpy*new_m], &result[ycpy], sizeof(float)); //transpose
//        }
//    }


//    printf("\nPadded A matrix:");
//    dbg_printmat(new_m, new_n, Apad);
//    printf("\nTransposed Padded A matrix:");
//    dbg_printmat(new_n, new_m, ATpad);
//    printf("\n");

    for(int i = 0; i < m; i++){
        memset(result, 0, sizeof(float) * new_m);
       //dbg_printmat(m, n, C);
        for(int k = 0; k < new_n; k+=4){
            /*
            unrolling the k loop, this represents A[i+k*m] in naive.
            */

            int kp = k + 1;
            int kpp = k + 2;
            int kppp = k + 3;
            __m128 k0 = _mm_load1_ps(&Apad[i+k*new_m]);
            __m128 k1 = _mm_load1_ps(&Apad[i+kp*new_m]);
            __m128 k2 = _mm_load1_ps(&Apad[i+kpp*new_m]);
            __m128 k3 = _mm_load1_ps(&Apad[i+kppp*new_m]);

            for(int j = 0; j < new_m; j+=16){
              /*
              unrolling j loop, this represents A[j+k*m] in naive, which is summed with the previous result
                after each k iteration.
              */

                /* load previous calculations */
               __m128 vC = _mm_loadu_ps(&result[j]);
               __m128 v1 = _mm_loadu_ps(&result[j+4]);
               __m128 v2 = _mm_loadu_ps(&result[j+8]);
               __m128 v3 = _mm_loadu_ps(&result[j+12]);

                /* sum the products of A matrices */
               vC = _mm_add_ps(vC, _mm_mul_ps(k0,
                              _mm_loadu_ps(&Apad[j+k*new_m] )));
               vC = _mm_add_ps(vC, _mm_mul_ps(k1,
                              _mm_loadu_ps(&Apad[j+kp*new_m] )));
               vC = _mm_add_ps(vC, _mm_mul_ps(k2,
                              _mm_loadu_ps(&Apad[j+kpp*new_m] )));
               vC = _mm_add_ps(vC, _mm_mul_ps(k3,
                              _mm_loadu_ps(&Apad[j+kppp*new_m] )));

               v1 = _mm_add_ps(v1, _mm_mul_ps(k0,
                              _mm_loadu_ps(&Apad[j+4+k*new_m] )));
               v1 = _mm_add_ps(v1, _mm_mul_ps(k1,
                              _mm_loadu_ps(&Apad[j+4+kp*new_m] )));
               v1 = _mm_add_ps(v1, _mm_mul_ps(k2,
                              _mm_loadu_ps(&Apad[j+4+kpp*new_m] )));
               v1 = _mm_add_ps(v1, _mm_mul_ps(k3,
                              _mm_loadu_ps(&Apad[j+4+kppp*new_m] )));

               v2 = _mm_add_ps(v2, _mm_mul_ps(k0,
                              _mm_loadu_ps(&Apad[j+8+k*new_m] )));
               v2 = _mm_add_ps(v2, _mm_mul_ps(k1,
                              _mm_loadu_ps(&Apad[j+8+kp*new_m] )));
               v2 = _mm_add_ps(v2, _mm_mul_ps(k2,
                              _mm_loadu_ps(&Apad[j+8+kpp*new_m] )));
               v2 = _mm_add_ps(v2, _mm_mul_ps(k3,
                              _mm_loadu_ps(&Apad[j+8+kppp*new_m] )));

               v3 = _mm_add_ps(v3, _mm_mul_ps(k0,
                              _mm_loadu_ps(&Apad[j+12+k*new_m] )));
               v3 = _mm_add_ps(v3, _mm_mul_ps(k1,
                              _mm_loadu_ps(&Apad[j+12+kp*new_m] )));
               v3 = _mm_add_ps(v3, _mm_mul_ps(k2,
                              _mm_loadu_ps(&Apad[j+12+kpp*new_m] )));
               v3 = _mm_add_ps(v3, _mm_mul_ps(k3,
                              _mm_loadu_ps(&Apad[j+12+kppp*new_m] )));

                /* store as a row, into result[] */
               _mm_storeu_ps(&result[j], vC);
               _mm_storeu_ps(&result[j+4], v1);
               _mm_storeu_ps(&result[j+8], v2);
               _mm_storeu_ps(&result[j+12], v3);
            }
            /*
            store transpose of result[] into C[]. This is equivalent to C[i+j*m] from naive. I plaed it outside of the J loop because it is a waste of resources to calculate the same after each iteration of j, when it can be done once after each k iteration!
            */
            int w = 0;
            int y = 0;
            while (y < m){
                C[i+w*m] = result[y]; //this notation is slightly faster!
                //memcpy would be faster in the case of a row to row copy, such as in padding matrix above. 
                //memcpy(&C[i+w*m], &result[y], sizeof(float)); //different notation, maybe this fixes memory problem?
                y++;
                w++;
            }
        }
    }
    free(result);
    free(Apad);
    //free(ATpad);
}

/*

initial thoughts:
 SSE: probably uses one less for loop? Doing 4 computations at once, so in a 4x4, thats only 4 execs,
 but in a 4x8 that twice (8 execs). vs 8x4 ( 8 execs). False! It just does less calculations, the amount of for loops
 should be the same.

 Why is A an mxn matrix and C an mxm matrix?

 I don't understant the notation: " C[i+j*m] += A[i+k*m] * A[j+k*m]; "

 Padding: Since SSE is loading 4 elements at a time, it makes sense to pad matrix so that it's dimmension is a multiple of four. Probably only neccesary for one dimmension, as in the SSE example above, as long as the row is a multiple of four, there can be as many columns.
 if(x % 4 != 0) -> needs padding!
 To pad, add additional spaces, intialized to zero.
 How would this affect product calculation?
    So far, none

 Loop Unrolling: Load blocks of 4, until end of row; making sure to update iterator accordingly.

 Pre-transposing, increase spatial locality? elements no longer are accesed off the row!

 */
