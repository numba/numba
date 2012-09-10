/* -*- mode: c; tab-width: 4; c-basic-offset: 4; -*- */ 

#include "vector-machine.h"

#include <immintrin.h>
#include <stdio.h>

#define RESTRICT __restrict

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

	template <typename TYPE>
	inline const void* internal_load_stream(void* target, const void* base, ptrdiff_t stride, size_t count)
	{
		TYPE* RESTRICT out = reinterpret_cast<TYPE*>(target);
		const TYPE* in = reinterpret_cast<const TYPE*>(base);
		do
		{
			*out = *in;
			out++;
			in = ptr_offset_bytes(in, stride);
		} while (--count);

		return reinterpret_cast<const void*>(in);
	}

	template <typename TYPE>
	inline void* internal_store_stream(const void* source, void* target, ptrdiff_t stride, size_t count)
	{
		TYPE* RESTRICT out = reinterpret_cast<TYPE*>(target);
		const TYPE* in = reinterpret_cast<const TYPE*>(source);
		do
		{
			*out = *in;
			out = ptr_offset_bytes(out, stride);
			in ++ ;
		} while (--count);

		return reinterpret_cast<void*>(out);
	}

	template <uintptr_t VALUE>
	inline bool a_kind_of_magic(const void* ptrA, const void* ptrB)
	{
		uintptr_t a = reinterpret_cast<uintptr_t>(ptrA);
		uintptr_t b = reinterpret_cast<uintptr_t>(ptrB);
		return 0 ==  ((a ^ b) & ~(VALUE - 1));
	}
}

const void* vvm_load(vvm_register* target, const void* base, size_t element_size, size_t count)
{
	size_t total_size = element_size * count;
	__builtin_memcpy(target, base, total_size);
  
	return ptr_offset_bytes(base, total_size);
}

const void* vvm_load_size4_stream_plain_c(vvm_register* target, const void* base, ptrdiff_t stride, size_t count)
{
	const uint32_t* src = reinterpret_cast<const uint32_t*>(base);
	uint32_t* RESTRICT dst = reinterpret_cast<uint32_t* RESTRICT>(target);
	/* assume count > 0 */
	do
	{
		*dst = *src;
		dst++;
		src = ptr_offset_bytes(src, stride);
	} while (--count);

	return src;
}

const void* vvm_load_size4_stream_unroll4_c(vvm_register* target, const void* base, ptrdiff_t stride, size_t count)
{
	const uint32_t* src = reinterpret_cast<const uint32_t*>(base);
	uint32_t* RESTRICT dst = reinterpret_cast<uint32_t* RESTRICT>(target);
	/* assume count > 0 and multiple of 4*/
	do 
	{
		for (size_t i = 0; i < 4; i ++)
		{
			dst[i] = *ptr_offset_bytes(src, stride * i);
		}
		dst += 4;
		src = ptr_offset_bytes(src, stride * 4);
		count -= 4;
	} while (count);

	return src;
}

const void* vvm_load_size4_stream_sse_v1(vvm_register* target, const void* base, ptrdiff_t stride, size_t count)
{
	const int32_t* src = reinterpret_cast<const int32_t*>(base);
	__m128i* RESTRICT dst = reinterpret_cast<__m128i* RESTRICT>(target);
	/* based on example 1 on intel optimization manual (example 3-31) */
	do 
	{
		__m128i el0 = _mm_cvtsi32_si128(*src);
		__m128i el1 = _mm_cvtsi32_si128(*(ptr_offset_bytes(src, stride)));
		__m128i el2 = _mm_cvtsi32_si128(*(ptr_offset_bytes(src, 2*stride)));
		__m128i el3 = _mm_cvtsi32_si128(*(ptr_offset_bytes(src, 3*stride)));

		__m128i lo = _mm_unpacklo_epi32(el0, el1);
		__m128i hi = _mm_unpacklo_epi32(el2, el3);
		__m128i res = _mm_unpacklo_epi32(lo, hi);

		_mm_store_si128(dst, res);

		src = ptr_offset_bytes(src, 4*stride);
		dst += 1;
		count -= 4;
	} while (count);

	return reinterpret_cast<const void*>(src);	
}

const void*  vvm_load_size4_stream_sse_v2(vvm_register* target, const void* base, ptrdiff_t stride, size_t count)
{
	const int32_t* src = reinterpret_cast<const int32_t*>(base);
	__m128i* RESTRICT dst = reinterpret_cast<__m128i* RESTRICT>(target);
	/* based on example 2 on intel optimization manual (example 3-31) */
	do 
	{
		__m128i el0 = _mm_cvtsi32_si128(*src);
		__m128i el1 = _mm_cvtsi32_si128(*(ptr_offset_bytes(src, stride)));
		__m128i el2 = _mm_cvtsi32_si128(*(ptr_offset_bytes(src, 2*stride)));
		__m128i el3 = _mm_cvtsi32_si128(*(ptr_offset_bytes(src, 3*stride)));

		__m128i el1s = _mm_slli_epi64(el1,32);
		__m128i el3s = _mm_slli_epi64(el3,32);
		__m128i el01 = _mm_or_ps(el0, el1s);
		__m128i el23 = _mm_or_ps(el2, el3s);
		__m128i res  = _mm_movelh_ps(el01, el23);

		_mm_store_si128(dst, res);

		src = ptr_offset_bytes(src, 4*stride);
		dst += 1;
		count -= 4;
	} while (count);

	return reinterpret_cast<const void*>(src);	
}

const void* vvm_load_size4_stream_sse_v3(vvm_register* target, const void* base, ptrdiff_t stride, size_t count)
{
	const int32_t* src = reinterpret_cast<const int32_t*>(base);
	__m128i* RESTRICT dst = reinterpret_cast<__m128i* RESTRICT>(target);
	/* based on example 3 on intel optimization manual (example 3-31) */
	do 
	{
		__m128i el0 = _mm_cvtsi32_si128(*src);
		__m128i el1 = _mm_cvtsi32_si128(*(ptr_offset_bytes(src, stride)));
		__m128i el2 = _mm_cvtsi32_si128(*(ptr_offset_bytes(src, 2*stride)));
		__m128i el3 = _mm_cvtsi32_si128(*(ptr_offset_bytes(src, 3*stride)));

		__m128i el13 = _mm_movelh_ps(el1, el3);
		__m128i el13s= _mm_slli_epi64(el13, 32);
		__m128i el02 = _mm_movelh_ps(el0, el2);
		__m128i res  = _mm_or_ps(el13s, el02);

		_mm_store_si128(dst, res);

		src = ptr_offset_bytes(src, 4*stride);
		dst += 1;
		count -= 4;
	} while (count);

	return reinterpret_cast<const void*>(src);	
}

// same but using streaming store...
const void* vvm_load_size4_stream_sse_v4(vvm_register* target, const void* base, ptrdiff_t stride, size_t count)
{
	const int32_t* src = reinterpret_cast<const int32_t*>(base);
	__m128i* RESTRICT dst = reinterpret_cast<__m128i* RESTRICT>(target);
	/* based on example 1 on intel optimization manual (example 3-31) */
	do 
	{
		__m128i el0 = _mm_cvtsi32_si128(*src);
		__m128i el1 = _mm_cvtsi32_si128(*(ptr_offset_bytes(src, stride)));
		__m128i el2 = _mm_cvtsi32_si128(*(ptr_offset_bytes(src, 2*stride)));
		__m128i el3 = _mm_cvtsi32_si128(*(ptr_offset_bytes(src, 3*stride)));

		__m128i el4 = _mm_cvtsi32_si128(*(ptr_offset_bytes(src, 4*stride)));
		__m128i el5 = _mm_cvtsi32_si128(*(ptr_offset_bytes(src, 5*stride)));
		__m128i el6 = _mm_cvtsi32_si128(*(ptr_offset_bytes(src, 6*stride)));
		__m128i el7 = _mm_cvtsi32_si128(*(ptr_offset_bytes(src, 7*stride)));

		__m128i lo0 = _mm_unpacklo_epi32(el0, el1);
		__m128i hi0 = _mm_unpacklo_epi32(el2, el3);
		__m128i res0 = _mm_unpacklo_epi32(lo0, hi0);

		__m128i lo1 = _mm_unpacklo_epi32(el4, el5);
		__m128i hi1 = _mm_unpacklo_epi32(el6, el7);
		__m128i res1 = _mm_unpacklo_epi32(lo1, hi1);

		_mm_store_si128(dst, res0);
		_mm_store_si128(dst + 1, res1);

		src = ptr_offset_bytes(src, 8*stride);
		dst += 2;
		count -= 8;
	} while (count);

	return reinterpret_cast<const void*>(src);	
}

const void* vvm_load_size4_stream_sse_v5(vvm_register* target, const void* base, ptrdiff_t stride, size_t count)
{
	const int32_t* src = reinterpret_cast<const int32_t*>(base);
	__m128i* RESTRICT dst = reinterpret_cast<__m128i* RESTRICT>(target);
	/* based on example 1 on intel optimization manual (example 3-31) */
	do 
	{
		__m128i el0 = _mm_cvtsi32_si128(*src);
		__m128i el1 = _mm_cvtsi32_si128(*(ptr_offset_bytes(src, stride)));
		__m128i el2 = _mm_cvtsi32_si128(*(ptr_offset_bytes(src, 2*stride)));
		__m128i el3 = _mm_cvtsi32_si128(*(ptr_offset_bytes(src, 3*stride)));

		__m128i el4 = _mm_cvtsi32_si128(*(ptr_offset_bytes(src, 4*stride)));
		__m128i el5 = _mm_cvtsi32_si128(*(ptr_offset_bytes(src, 5*stride)));
		__m128i el6 = _mm_cvtsi32_si128(*(ptr_offset_bytes(src, 6*stride)));
		__m128i el7 = _mm_cvtsi32_si128(*(ptr_offset_bytes(src, 7*stride)));

		__m128i el1s = _mm_slli_epi64(el1,32);
		__m128i el3s = _mm_slli_epi64(el3,32);
		__m128i el01 = _mm_or_ps(el0, el1s);
		__m128i el23 = _mm_or_ps(el2, el3s);
		__m128i res0 = _mm_movelh_ps(el01, el23);

		__m128i el5s = _mm_slli_epi64(el5,32);
		__m128i el7s = _mm_slli_epi64(el7,32);
		__m128i el45 = _mm_or_ps(el4, el5s);
		__m128i el67 = _mm_or_ps(el6, el7s);
		__m128i res1 = _mm_movelh_ps(el45, el67);

		_mm_store_si128(dst, res0);
		_mm_store_si128(dst + 1, res1);

		src = ptr_offset_bytes(src, 8*stride);
		dst += 2;
		count -= 8;
	} while (count);

	return reinterpret_cast<const void*>(src);	
}

const void*  vvm_load_size4_stream_sse_v6(vvm_register* target, const void* base, ptrdiff_t stride, size_t count)
{
	const int32_t* src = reinterpret_cast<const int32_t*>(base);
	__m128i* RESTRICT dst = reinterpret_cast<__m128i* RESTRICT>(target);
	/* based on example 1 on intel optimization manual (example 3-31) */
	do 
	{
		__m128i el0 = _mm_cvtsi32_si128(*src);
		__m128i el1 = _mm_cvtsi32_si128(*(ptr_offset_bytes(src, stride)));
		__m128i el2 = _mm_cvtsi32_si128(*(ptr_offset_bytes(src, 2*stride)));
		__m128i el3 = _mm_cvtsi32_si128(*(ptr_offset_bytes(src, 3*stride)));

		__m128i el4 = _mm_cvtsi32_si128(*(ptr_offset_bytes(src, 4*stride)));
		__m128i el5 = _mm_cvtsi32_si128(*(ptr_offset_bytes(src, 5*stride)));
		__m128i el6 = _mm_cvtsi32_si128(*(ptr_offset_bytes(src, 6*stride)));
		__m128i el7 = _mm_cvtsi32_si128(*(ptr_offset_bytes(src, 7*stride)));

		__m128i el13 = _mm_movelh_ps(el1, el3);
		__m128i el13s= _mm_slli_epi64(el13, 32);
		__m128i el02 = _mm_movelh_ps(el0, el2);
		__m128i res0 = _mm_or_ps(el13s, el02);

		__m128i el57 = _mm_movelh_ps(el5, el7);
		__m128i el57s= _mm_slli_epi64(el57, 32);
		__m128i el46 = _mm_movelh_ps(el4, el6);
		__m128i res1  = _mm_or_ps(el57s, el46);

		_mm_store_si128(dst, res0);
		_mm_store_si128(dst + 1, res1);

		src = ptr_offset_bytes(src, 8*stride);
		dst += 2;
		count -= 8;
	} while (count);

	return reinterpret_cast<const void*>(src);	
}




// TODO: only issue a prefetch only when changing cache line, issue a load
//       when changing page
void vvm_prefetch_stream(const void* base, ptrdiff_t stride, size_t count)
{
#   define CACHELINE_SIZE 64
#   define PAGE_SIZE 4096
	uintptr_t test = 0;
	do
	{
		test = test ^ reinterpret_cast<uintptr_t>(base);
        if (test >= CACHELINE_SIZE)
	    {
			// needs prefetch
			if (test < PAGE_SIZE)
			{
				// just a line
				__asm("prefetcht0 (%0)" : :"c" (base));
			}
			else
			{
				// a page
				char foo = 0;
				__asm volatile("movb (%1), %0" : "=c" (foo) : "c" (base) );
			}	
		}

		test = reinterpret_cast<uintptr_t>(base);
		base = ptr_offset_bytes(base, stride);
	} while (--count);
#   undef PAGE_SIZE
#   undef CACHELINE_SIZE
}

const void* vvm_load_stream(vvm_register* target, const void* base, size_t element_size, ptrdiff_t stride, size_t count)
{
	switch(stride)
	{
	case 1:
		return internal_load_stream<uint8_t>((void*) target, base, stride, count);
	case 2:
		return internal_load_stream<uint16_t>((void*) target, base, stride, count);
	case 4:
		return internal_load_stream<uint32_t>((void*) target, base, stride, count);
	case 8:
		return internal_load_stream<uint64_t>((void*) target, base, stride, count);
	default:
		return base;
	}
}

void* vvm_store(const vvm_register* source, void* base, size_t element_size, size_t count)
{
	size_t total_size = element_size * count;
	__builtin_memcpy(base, source, total_size);
	return ptr_offset_bytes(base, total_size);
} 

void* vvm_store_stream(const vvm_register* target, void* base, size_t element_size, ptrdiff_t stride, size_t count)
{
	switch(stride)
	{
	case 1:
		return internal_store_stream<uint8_t>((const void*) target, base, stride, count);
	case 2:
		return internal_store_stream<uint16_t>((const void*) target, base, stride, count);
	case 4:
		return internal_store_stream<uint32_t>((const void*) target, base, stride, count);
	case 8:
		return internal_store_stream<uint64_t>((const void*) target, base, stride, count);
	default:
		return base;
	}
}

void* vvm_store_size4_stream_plain_c(const vvm_register* source, void* target, ptrdiff_t stride, size_t count)
{
	int32_t* RESTRICT out = reinterpret_cast<int32_t*>(target);
	const int32_t* in = reinterpret_cast<const int32_t*>(source);
	do
	{
		*out = *in;
		out = ptr_offset_bytes(out, stride);
		in ++ ;
	} while (--count);
	
	return reinterpret_cast<void*>(out);	
}

void* vvm_store_size4_stream_unroll4_c(const vvm_register* source, void* target, ptrdiff_t stride, size_t count)
{
	int32_t* RESTRICT out = reinterpret_cast<int32_t*>(target);
	const int32_t* in = reinterpret_cast<const int32_t*>(source);
	// assume count as a multiple of 4
	do
	{
		for (size_t i = 0; i < 4; i++)
		{
			*ptr_offset_bytes(out, 0*stride) = in[0];
			*ptr_offset_bytes(out, 1*stride) = in[1];
			*ptr_offset_bytes(out, 2*stride) = in[2];
			*ptr_offset_bytes(out, 3*stride) = in[3];
		}
		out = ptr_offset_bytes(out, 4*stride);
		in += 4;
	} while (--count);
	
	return reinterpret_cast<void*>(out);	
}

void* vvm_store_size4_stream_unroll4_nt(const vvm_register* source, void* target, ptrdiff_t stride, size_t count)
{
	int32_t* RESTRICT out = reinterpret_cast<int32_t*>(target);
	const int32_t* in = reinterpret_cast<const int32_t*>(source);
	// assume count as a multiple of 4
	do
	{
		for (size_t i = 0; i < 4; i++)
		{
			_mm_stream_si32(ptr_offset_bytes(out, 0*stride), in[0]);
			_mm_stream_si32(ptr_offset_bytes(out, 1*stride), in[1]);
			_mm_stream_si32(ptr_offset_bytes(out, 2*stride), in[2]);
			_mm_stream_si32(ptr_offset_bytes(out, 3*stride), in[3]);
		}
		out = ptr_offset_bytes(out, 4*stride);
		in += 4;
	} while (--count);
	
	return reinterpret_cast<void*>(out);	
}

void* vvm_store_size4_stream_seq(const vvm_register* source, void* target, ptrdiff_t stride, size_t count)
{
	__m128* RESTRICT out = reinterpret_cast<__m128*>(target);
	const __m128* in = reinterpret_cast<const __m128*>(source);
	// assume multiple of a cacheline... (64b)
	do
	{
		__m128 r0 = in[0];
		__m128 r1 = in[1];
		__m128 r2 = in[2];
		__m128 r3 = in[3];
		
		out[0] = r0;
		out[1] = r1;
		out[2] = r2;
		out[3] = r3;
		in  += 4;
		out += 4;
		count -= 16;
	} while (count);
	
	return reinterpret_cast<void*>(out);	
}

void* vvm_store_size4_stream_seq_nt(const vvm_register* source, void* target, ptrdiff_t stride, size_t count)
{
	double* RESTRICT out = reinterpret_cast<double*>(target);
	const __m128* in = reinterpret_cast<const __m128*>(source);
	// assume multiple of a cacheline... (64b)
	do
	{
		__m128 r0 = in[0];
		__m128 r1 = in[1];
		__m128 r2 = in[2];
		__m128 r3 = in[3];
		
		_mm_stream_pd(out, r0);
		_mm_stream_pd(out + 2, r1);
		_mm_stream_pd(out + 4, r2);
		_mm_stream_pd(out + 6, r3);

		in  += 4;
		out += 8;
		count -= 16;
	} while (count);
	
	return reinterpret_cast<void*>(out);	
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
