#pragma once // Ensures this header file is only included once per compilation unit

#include <cassert> // For runtime assertions (debug checks)
#include <cstdlib> // For malloc() and free()
#include <cmath>   // For sqrtf()
#include <immintrin.h> // for intrinsics AVX & SSE2
#include <iostream> // for couts

// Inline helper function to compute the square of a number
static inline double sq(double x)
{
	return x * x;
}

// A simple dynamically allocated vector of doubles
// Provides basic functionality and a cosine similarity operation
struct vector
{
	// Constructor: allocates memory for n elements
	// Requires n to be positive and a multiple of 8 (likely for alignment/SIMD use)
	vector(const int n) : n(n)
	{
		assert(n > 0);								 // Validate size
		#if defined(__AVX__) || defined(__SSE__)
			data = (double *)_mm_malloc(sizeof(double) * n, 32); // Allocate aligned memory for AVX/SSE2
		#else
			data = (double *)malloc(sizeof(double) * n); // Allocate raw memory
		#endif
		assert(data);								 // Ensure allocation succeeded
	}

	// Destructor: frees allocated memory
	// Invoked automatically when a vector object goes out of scope or is deleted
	~vector()
	{
		#if defined(__AVX__) || defined(__SSE__)
			_mm_free(data); // Free aligned memory
		#else
			free(data); // Free raw memory
		#endif
	}

	// Sets the value of the i-th element in the vector
	void set(double x, int i)
	{
		assert(i >= 0 && i < n); // Bounds check
		data[i] = x;
	}

	// Computes the cosine similarity between two vectors:
	static double cosine_similarity(const vector &a, const vector &b)
	{
		assert(a.n == b.n); // Vectors must be of the same dimension
		const int n = a.n;
		
		// AVX vectorization case
		int i = 0;
		#if defined(__AVX__)
			std::cout << "Using AVX " << std::endl;
			__m256d vab = _mm256_setzero_pd(); // Dot product accumulator
			__m256d vaa = _mm256_setzero_pd(); // Magnitude accumulator for vector a (va)
			__m256d vbb = _mm256_setzero_pd(); // Magnitude accumulator for vector b (vb)

			for (i = 0; i <= n-4; i+=4) {
				// loading the elements packed with i in from both vectors a & b
				__m256d va = _mm256_load_pd(&a.data[i]); 
				__m256d vb = _mm256_load_pd(&b.data[i]);

				#if defined(__FMA__) // FMA merged implementation case
					vaa = _mm256_fmadd_pd(va, va, vaa);
					vbb = _mm256_fmadd_pd(vb, vb, vbb);
					vab = _mm256_fmadd_pd(va, vb, vab);
				#else // if FMA is not possible, compute standard way AVX
					vaa = _mm256_add_pd(vaa, _mm256_mul_pd(va, va));
					vbb = _mm256_add_pd(vbb, _mm256_mul_pd(vb, vb));
					vab = _mm256_add_pd(vab, _mm256_mul_pd(va, vb));
				#endif
			}

			// Tree-structured horizontal reduction on registers
			// basically summing up the high and low parts of the 256-bit registers
			__m128d low_ab  = _mm256_castpd256_pd128(vab);
			__m128d high_ab = _mm256_extractf128_pd(vab, 1);
			__m128d sum_ab  = _mm_add_pd(low_ab, high_ab);  
			__m128d shuf_ab = _mm_shuffle_pd(sum_ab, sum_ab, 1);
			__m128d final_ab = _mm_add_sd(sum_ab, shuf_ab);
			double ab = _mm_cvtsd_f64(final_ab);

			__m128d low_aa  = _mm256_castpd256_pd128(vaa);
			__m128d high_aa = _mm256_extractf128_pd(vaa, 1);
			__m128d sum_aa  = _mm_add_pd(low_aa, high_aa);
			__m128d shuf_aa = _mm_shuffle_pd(sum_aa, sum_aa, 1);
			__m128d final_aa = _mm_add_sd(sum_aa, shuf_aa);
			double aa = _mm_cvtsd_f64(final_aa);

			__m128d low_bb  = _mm256_castpd256_pd128(vbb);
			__m128d high_bb = _mm256_extractf128_pd(vbb, 1);
			__m128d sum_bb  = _mm_add_pd(low_bb, high_bb);
			__m128d shuf_bb = _mm_shuffle_pd(sum_bb, sum_bb, 1);
			__m128d final_bb = _mm_add_sd(sum_bb, shuf_bb);
			double bb = _mm_cvtsd_f64(final_bb);

			// take remaining elements if the n is not divisible by 4
			for (; i < n; ++i) {
				aa += sq(a.data[i]);
				bb += sq(b.data[i]);
				ab += a.data[i] * b.data[i];
			}

			return ab / (sqrt(aa) * sqrt(bb));

		#elif defined (__SSE__) 
			// SSE implementation case
			std::cout << "Using SSE" << std::endl;
			__m128d vab = _mm_setzero_pd(); // Dot product accumulator
			__m128d vaa = _mm_setzero_pd(); // Magnitude accumulator for vector a (va)
			__m128d vbb = _mm_setzero_pd(); // Magnitude accumulator for vector b (vb)

			for (i = 0; i <= n-2; i+=2) {
				// loading the elements packed with i in from both vectors a & b
				__m128d va = _mm_load_pd(&a.data[i]); 
				__m128d vb = _mm_load_pd(&b.data[i]);

				vaa = _mm_add_pd(vaa, _mm_mul_pd(va, va));
				vbb = _mm_add_pd(vbb, _mm_mul_pd(vb, vb));
				vab = _mm_add_pd(vab, _mm_mul_pd(va, vb));
			}

			// Tree-structured horizontal reduction on registers
			// summing up the high and low halves of the 128-bit registers
			__m128d shuf_ab = _mm_shuffle_pd(vab, vab, 1);  
			__m128d sum_ab = _mm_add_sd(vab, shuf_ab);      
			double ab = _mm_cvtsd_f64(sum_ab);

			__m128d shuf_aa = _mm_shuffle_pd(vaa, vaa, 1);
			__m128d sum_aa = _mm_add_sd(vaa, shuf_aa);
			double aa = _mm_cvtsd_f64(sum_aa);

			__m128d shuf_bb = _mm_shuffle_pd(vbb, vbb, 1);
			__m128d sum_bb = _mm_add_sd(vbb, shuf_bb);
			double bb = _mm_cvtsd_f64(sum_bb);

			// take remaining elements if the n is not divisible by 2
			for (; i < n; ++i) {
				aa += sq(a.data[i]);
				bb += sq(b.data[i]);
				ab += a.data[i]*b.data[i];
			}

			return ab / (sqrt(aa) * sqrt(bb));

		#else
			// standard implementation case
			double ab = 0.0; // Dot product accumulator
			double aa = 0.0; // Magnitude accumulator for vector a
			double bb = 0.0; // Magnitude accumulator for vector b

			for (int i = 0; i < n; ++i)
			{
				aa += sq(a.data[i]);
				bb += sq(b.data[i]);
				ab += a.data[i] * b.data[i];
			}

			// Normalize by magnitudes; potential division by zero if aa or bb == 0
			return ab / (sqrtf(aa) * sqrtf(bb));
		#endif
	}

private:
	const int n = 0;		// Dimension (size) of the vector
	double *data = nullptr; // Pointer to dynamically allocated array
};
