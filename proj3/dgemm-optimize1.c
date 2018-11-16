#include "xmmintrin.h"
#include <string.h>
#include <stdio.h>
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
n : width

*/
void dgemm( int m, int n, float *A, float *C )
{

  //consider padding matrix
  int new_m = m;
  while(new_m % 4 != 0){
    new_m++;
  }
  //padded matrix is a copy of A[]
  float *Apad = (float*) malloc( new_m * new_m * sizeof(float) );
  memset(Apad, 0, sizeof(float) * new_m * new_m);
  memcpy(Apad, A, sizeof(float) * m * m);

  /* temporary result */
  /* result only stores one row */
  float *result = (float*) malloc( new_m * new_m * sizeof(float) );
  memset(result, 0, sizeof(float) * new_m * new_m);

  for(int i = 0; i < m; i++){
    //zero out result for next column
    memset(result, 0, sizeof(float) * m * m);

    for(int k = 0; k < n; k++){
      //j += 4 because SSE loads 4 elements in a ROW
      for(int j = 0; j < m; j+=4){
        __m128 vC = _mm_loadu_ps(&result[j]);

        vC = _mm_add_ps(vC, _mm_mul_ps(_mm_loadu_ps(&A[j+k*m]),
        _mm_load1_ps(&A[i+k*m]) ) );

        _mm_storeu_ps(&result[j], vC);

        //store the transpose of result into C[]
        int w = j;
        int y = 0;
        while (y < m){
          C[i+w*m] = result[y];
          y++;
          w++;
        }

      //The problem comes from storing the result contiguously!
      //possible fix: transpose the matrix after last for loop?
      }
    }
  }
  /* transpose */
  /*
  for(int w = 0; w < m * m; w++){
  for(int y = 0; y < m * m; y++){
    C[y+w*m] = result[w+y*m];
  }
  }
  */

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

*/
