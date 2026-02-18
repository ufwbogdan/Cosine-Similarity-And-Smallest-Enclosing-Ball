#include <immintrin.h>
#include <stdio.h>

// to compile: click on the gear on the top right corner and add "-mavx" as Extra Compiler Flag

void print(__m256 a, const char* label = nullptr) {
    if(label) printf("%s = ", label);
    float output[8];
    _mm256_storeu_ps(output, a);
    printf("[ ");
    for (int i = 8; i > 4; i--) printf("%2.0f ", output[i-1]);
    printf("] [ ");
    for (int i = 4; i > 0; i--) printf("%2.0f ", output[i-1]);
    printf("]\n");
}

int main() {
    
    __m256 A = _mm256_set_ps(1, 2, 3, 4, 5, 6, 7, 8);
    __m256 B = _mm256_set_ps(9, 10, 11, 12, 13, 14, 15, 16);
    __m256 C = _mm256_permute2f128_ps(A, B, _MM_SHUFFLE(0,2,0,1));
    
    /*
    
    A and B are devided in 4 chunks:
    
    let [  5  6  7  8 ] be referred as chunk 0 
    let [  1  2  3  4 ] be referred as chunk 1
    let [ 13 14 15 16 ] be referred as chunk 2
    let [  9 10 11 12 ] be referred as chunk 3
    
    _MM_SHUFFLE(k1,k2,k3,k4) usage :
    
    k1) if it equals 2 or 3 then the LEFT HALF of the result gets zero (independently from k2),
    k2) it defines the chunk to be copied into the LEFT HALF of the result,
    k3) if it equals 2 or 3 then the RIGHT HALF of the result gets zero (independently from k4),
    k4) it defines the chunk to be copied into the RIGHT HALF of the result
    
    */
    
    print(A, "A");
    print(B, "B");
    print(C, "C");
    return 0;
}