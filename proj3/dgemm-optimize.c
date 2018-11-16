#include "xmmintrin.h"
/*
 works cited

 SSE examples
 zybooks, Computer Organization and Design; chapters 3, 4, 5, 6
 https://www.scss.tcd.ie/David.Gregg/cs3014/notes/lecture16-sse1.pdf

 Loop Unrolling examples

 Padding examples
 */

/*
 C : m x m
 A : m x n
 A (transpose) : n x m
 A * A (transpose) is always a square matrix!

 m : height
 n : width (padding!)

 */
void dbg_printmat(int m, int n, float* D);

void dgemm_optimize( int m, int n, float *A, float *C )
{
    /* consider padding matrix if dimmensions are not optimal */
    int new_m = m;
    while(new_m % 4 != 0){
        new_m++;
    }

    /* padded matrix(es) */
        /* might as well pre transpose matrix during padding phase */
    float *Apad = (float*) malloc (new_m * n * (sizeof(float)) );
    memset(Apad, 0, new_m * n * sizeof(float) );
    for(int icpy = 0; icpy < n; icpy++){
        memcpy(&Apad[icpy*new_m], &A[icpy*m], sizeof(float) * m);
    }
    //printf("\nPadded A matrix:");
    //dbg_printmat(new_m, n, Apad);

    /* temp result */
    float *result = (float*) malloc( m * sizeof(float) );
    memset(result, 0, sizeof(float) * m);

    for(int i = 0; i < m; i++){
        memset(result, 0, sizeof(float) * m);
        for(int k = 0; k < n; k++){

            /* loop unrolling goes here, load as much of the row as possible
            check that the row fits, update iterator
            same i and k values, constant j values!
            With padding, this should always be a multiple of four!
            */
            //__m128 v1;
            //__m128 v2;
            //__m128 v3;
            //__m128 v4;

            for(int j = 0; j < m; j+=4){
                /* load previous calculations */
                __m128 vC = _mm_loadu_ps(&result[j]);

                /* sum the products of A matrices */
                vC = _mm_add_ps(vC, _mm_mul_ps(_mm_loadu_ps(&A[j+k*m]),
                                               _mm_load1_ps(&A[i+k*m]) ) );

                /* store as a row, into result[] */
                _mm_storeu_ps(&result[j], vC);

                /* store transpose of result[] into C[] */
                int w = j;
                int y = 0;
                while (y < m){
                    C[i+w*m] = result[y];
                    y++;
                    w++;
                }

            }
        }
    }
    free(result);
    free(Apad);
}

/*
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
