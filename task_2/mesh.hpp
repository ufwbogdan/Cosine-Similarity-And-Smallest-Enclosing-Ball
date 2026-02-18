#pragma once // Ensure this header is included only once

#include "common.hpp" // definitions of point, ball, etc.
#include <cassert>	  // for assert()
#include <cstdlib>	  // for malloc() and free()
#include <cmath>	  // for sqrtf()
#include <immintrin.h> // for SIMD instrinsics

// return square of x
static inline float sq(const float &x)
{
	return x * x;
}

// return squared distance between two d-dimensional points
template <int d>
static inline float sqdst(const point<d> &a, const point<d> &b)
{
	float sum = 0.0f;
	for (int i = 0; i < d; ++i)
		sum += sq(a.v[i] - b.v[i]);
	return sum;
}

// mesh of n d-dimensional points (SoA layout)
template <const int d>
struct mesh
{
	// constructor: allocate memory from the n points to d dimensions
	mesh(const int n) : n(n)
	{
		assert(n > 0 && n % 16 == 0); // size must be positive and multiple of 16
		for(int di = 0; di < d; di++){
			#if defined(__AVX) || defined(__SSE__)
            	data[di] = (float*)_mm_malloc(sizeof(float) * n, 32); // allocate memory
            #else 
				data[di] = (float*)malloc(sizeof(float) * n); // allocate raw memory
			#endif
			assert(data[di]);
		}
		assert(data); //ensure allocation succeeded
	}

	// destructor: free allocated memory
	~mesh()
	{
		for(int di = 0; di < d; di++){
			#if defined (__AVX__) || defined (__SSE__)
           		_mm_free(data[di]);
			#else
				free(data[di]);
			#endif
        }
	}

	// set the i-th point in the mesh
	void set(const point<d> p, int i)
	{
		assert(i >= 0 && i < n);
		for(int di = 0; di < d; di++){
            data[di][i] = p.v[di];
        }
	}

	// compute enclosing ball: center = midpoint of min/max, radius = max distance to center
	ball<d> calc_ball()
	{
		ball<d> b;
		point<d> min;
		point<d> max;

		// AVX implementation case
		#if defined(__AVX__)
			std::cout << "Using AVX\n";
			float tmp[8];
			// compute min/max per coordinate to find center
			for(int di = 0; di < d; di++){
				__m256 vmin = _mm256_set1_ps(data[di][0]);
				__m256 vmax = _mm256_set1_ps(data[di][0]);
				int i = 0;
				for(; i <= n - 8; i += 8){
					__m256 v = _mm256_load_ps(&data[di][i]);
					vmin = _mm256_min_ps(vmin, v);
					vmax = _mm256_max_ps(vmax, v);
				}
				// horizontal operation in order to get the min and max values from the registers
				
				_mm256_store_ps(tmp, vmin);
				min.v[di] = tmp[0];
				for(int k = 1; k < 8; k++){
					if(tmp[k] < min.v[di]){
						min.v[di] = tmp[k];
					}
				}
				_mm256_store_ps(tmp, vmax);
				max.v[di] = tmp[0];
				for(int k = 1; k < 8; k++){
					if(tmp[k] > max.v[di]){
						max.v[di] = tmp[k];
					}
				}
				b.center.v[di] = 0.5f * (min.v[di] + max.v[di]);
			}

			// compute radius = max distance from center
			float maxsqdst = 0.0f;
			int i = 0;
			for (; i <= n - 8; i += 8)
			{
				__m256 vsum = _mm256_setzero_ps();
				for(int di = 0; di < d; di++){
					__m256 vdiff = _mm256_sub_ps(_mm256_load_ps(&data[di][i]), _mm256_set1_ps(b.center.v[di]));
					vsum = _mm256_add_ps(_mm256_mul_ps(vdiff, vdiff), vsum);
				}
				
				_mm256_store_ps(tmp, vsum);
				for(int k = 0; k < 8; k++){
					maxsqdst = std::max(maxsqdst, tmp[k]);
				}
			}
			// sqrt only once for efficiency
			b.radius = sqrtf(maxsqdst);

			return b; // return enclosing ball

		// SSE implementation case
		#elif defined (__SSE__)
			std::cout<<"Using SSE\n";
			float tmp[4];
			// compute min/max per coordinate to find center
			for(int di = 0; di < d; di++){
				__m128 vmin = _mm_set1_ps(data[di][0]);
				__m128 vmax = _mm_set1_ps(data[di][0]);
				int i = 0;
				for(; i <= n - 4; i += 4){
					__m128 v = _mm_load_ps(&data[di][i]);
					vmin = _mm_min_ps(vmin, v);
					vmax = _mm_max_ps(vmax, v);
				}
				// horizontal operation in order to get the min and max values from the registers
				
				_mm_store_ps(tmp, vmin);
				min.v[di] = tmp[0];
				for(int k = 1; k < 4; k++){
					if(tmp[k] < min.v[di]){
						min.v[di] = tmp[k];
					}
				}
				_mm_store_ps(tmp, vmax);
				max.v[di] = tmp[0];
				for(int k = 1; k < 4; k++){
					if(tmp[k] > max.v[di]){
						max.v[di] = tmp[k];
					}
				}
				b.center.v[di] = 0.5f * (min.v[di] + max.v[di]);
			}

			// compute radius = max distance from center
			float maxsqdst = 0.0f;
			int i = 0;
			for (; i <= n - 4; i += 4)
			{
				__m128 vsum = _mm_setzero_ps();
				for(int di = 0; di < d; di++){
					__m128 vdiff = _mm_sub_ps(_mm_load_ps(&data[di][i]), _mm_set1_ps(b.center.v[di]));
					vsum = _mm_add_ps(_mm_mul_ps(vdiff, vdiff), vsum);
				}
				_mm_store_ps(tmp, vsum);
				for(int k = 0; k < 4; k++){
					maxsqdst = std::max(maxsqdst, tmp[k]);
				}
			}
			// sqrt only once for efficiency
			b.radius = sqrtf(maxsqdst);

			return b; // return enclosing ball
		#else
			std::cout<<"Using standard implementation\n";
			// compute min/max per coordinate to find center
			for(int di = 0; di < d; di++){ 
				min.v[di] = data[di][0]; 
			} 
			 for(int di = 0; di < d; di++){ 
				max.v[di] = data[di][0]; 
			} 
			for(int di = 0; di < d; di++){
				 for(int i = 1; i < n; i++){
					 min.v[di] = data[di][i] < min.v[di] ? data[di][i] : min.v[di];
					 max.v[di] = data[di][i] > max.v[di] ? data[di][i] : max.v[di];
				}
			} 
			 // center = midpoint of min/max
			for (int i = 0; i < d; ++i) 
				b.center.v[i] = (max.v[i] - min.v[i]) * 0.5f + min.v[i];
			// compute radius = max distance from center
			float maxsqdst = 0.0f;
			for (int i = 0; i < n; ++i) { 
				float sum = 0.0f;
				for(int di = 0; di < d; di++){ 
					float diff = data[di][i] - b.center.v[di];
					sum += sq(diff); 
				}
				if(sum > maxsqdst) 
					 maxsqdst = sum; 
			} 
			// sqrt only once for efficiency
			b.radius = sqrtf(maxsqdst);
			return b; // return enclosing ball
		#endif
	}

	// return index of the farthest point from given point p
int farthest(point<d> p)
	{
		int argmax = 0;
		float maxsqdst = -1.0f;

		#if defined(__AVX__)
			// AVX farthest implementation
			std::cout << "Using AVX in farthest\n";
			float tmp[8];
			for (int i = 0; i <= n - 8; i += 8) {
				__m256 vsum = _mm256_setzero_ps();
				for (int di = 0; di < d; ++di) {
					__m256 v = _mm256_load_ps(&data[di][i]);                 
					__m256 pc = _mm256_set1_ps(p.v[di]);                    
					__m256 diff = _mm256_sub_ps(v, pc);
					vsum = _mm256_add_ps(vsum, _mm256_mul_ps(diff, diff));  
				}

				_mm256_storeu_ps(tmp, vsum); 
				for (int k = 0; k < 8; ++k) {
					if (tmp[k] > maxsqdst) {
						maxsqdst = tmp[k];
						argmax = i + k;
					}
				}
			}

			return argmax;

		#elif defined(__SSE__)
			// SSE farthest implementation
			std::cout << "Using SSE in farthest\n";
			float tmp[4];
			for (int i = 0; i <= n - 4; i += 4) {
				__m128 vsum = _mm_setzero_ps();
				for (int di = 0; di < d; ++di) {
					__m128 v = _mm_load_ps(&data[di][i]);                
					__m128 pc = _mm_set1_ps(p.v[di]);                   
					__m128 diff = _mm_sub_ps(v, pc);
					vsum = _mm_add_ps(vsum, _mm_mul_ps(diff, diff));     
				}
				_mm_storeu_ps(tmp, vsum);
				for (int k = 0; k < 4; ++k) {
					if (tmp[k] > maxsqdst) {
						maxsqdst = tmp[k];
						argmax = i + k;
					}
				}
			}

			return argmax;

		#else
			{
				// scalar farthest implementation
				std::cout << "Using scalar in farthest\n";
				for (int i = 0; i < n; ++i) {
					float sum = 0.0f;
					for (int di = 0; di < d; ++di) {
						float diff = data[di][i] - p.v[di];
						sum += diff * diff;
					}
					if (sum > maxsqdst) {
						maxsqdst = sum;
						argmax = i;
					}
				}
				return argmax;
			}
		#endif
	}


private:
	const int n = 0;		  // number of points
    float* data[d];
};
