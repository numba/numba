/* -*- mode: c; tab-width: 4; c-basic-offset: 4; -*- */ 

#include "vector-machine.h"

#include <immintrin.h>
#include <stdio.h>

namespace // nameless
{
	template <typename T>
    inline T* ptr_offset_bytes(T* addr, ptrdiff_t offset)
	{
		return reinterpret_cast<T*>( reinterpret_cast<intptr_t>(addr) + offset);
    } 
  
	// return the number of iterations of a loop computing "per_iter_count" operations at a time when
	// at least count operations are required 
	inline size_t compute_loop_iters(size_t count, size_t per_iter_count)
	{
		return ((count + per_iter_count - 1) / per_iter_count);
	}
}

const void* vvm_load(vvm_register* target, const void* base, size_t element_size, size_t count)
{
	size_t total_size = element_size * count;
	__builtin_memcpy(target, base, total_size);
  
	return ptr_offset_bytes(base, total_size);
}

void* vvm_store(const vvm_register* source, void* base, size_t element_size, size_t count)
{
	size_t total_size = element_size * count;
	__builtin_memcpy(base, source, total_size);
	return ptr_offset_bytes(base, total_size);
} 

void vvm_add_float_single(const vvm_register* srcA_in, const vvm_register* srcB_in, vvm_register* dst_out, size_t count)
{
	static const size_t simd_iter = 4;
	static const size_t per_iter_count = simd_iter * sizeof(__m256) / sizeof(float);
	const __m256* srcA = reinterpret_cast<const __m256*>(srcA_in);
	const __m256* srcB = reinterpret_cast<const __m256*>(srcB_in);
	__m256* __restrict dst = reinterpret_cast<__m256* __restrict>(dst_out);
	
	size_t loop_iterations = compute_loop_iters(count, per_iter_count);
	do
	{
		for (size_t i = 0; i < simd_iter; i++)
		{
			dst[i] = _mm256_add_ps(srcA[i], srcB[i]);
		} 
      
		dst  += simd_iter;
		srcA += simd_iter;
		srcB += simd_iter;
    } while (--loop_iterations);
}


#if defined(__APPLE__) && defined(__MACH__)

#include <Accelerate/Accelerate.h>

void vvm_sin_float_single(const vvm_register* src_in, vvm_register* dst_out, size_t count)
{
	const int icount = static_cast<int>(count);
	vvsinf((float*)dst_out, (const float*) src_in, &icount);
}

#else
#endif
