/* -*- mode: c; tab-width: 4; c-basic-offset: 4; -*- */ 


#include "Python.h"

#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

#include <inttypes.h>
#include <stddef.h>
#include <immintrin.h>

//#define SIMDTEST_VERBOSE

#if defined(SIMDTEST_VERBOSE)
#   include <stdio.h>
#   define VERBOSE_LOG(X) do { X; } while(0)
#else // !defined(SIMDTEST_VERBOSE)
#   define VERBOSE_LOG(X) do {} while(0)
#endif // SIMDTEST_VERBOSE



/*
 * Some simple test on kernels written using simd intrinsics for the intel
 * processor.
 */

static PyMethodDef simdtest_methods[] = 
{
	{NULL, NULL, 0, NULL}
};

/* The loop definition must precede the PyMODINIT_FUNC. */
/* Original
static void double_logit(char **args, npy_intp *dimensions,
                            npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    double tmp;

    for (i = 0; i < n; i++) {
        // BEGIN main ufunc computation
        tmp = *(double *)in;
        tmp /= 1-tmp;
        *((double *)out) = log(tmp);
        // END main ufunc computation

        in += in_step;
        out += out_step;
    }
}
*/


/* returns new position in the stream */
static inline const float* 
linearize_floats(const float* src,
				 ptrdiff_t stride,
				 size_t count,
				 float* dst)
{
	__builtin_memcpy(dst, src, count * sizeof(float));
	return src+count;
	/*
	for (size_t i = 0; i < count; ++i)
	{
		dst[i] = *src;
		src = (const float*)(((const char*) src) + stride);
	}

	return src;
	*/
}

static inline float* 
delinearize_floats(const float* src,
				   size_t count,
				   float* dst,
				   ptrdiff_t stride)
{
	__builtin_memcpy(dst, src, count * sizeof(float));
	return dst + count;
	/*
	for (size_t i = 0; i < count; ++i)
	{
		*dst = src[i];
		dst = (float*)(((char*) dst) + stride);
	}

	return dst;
	*/
}

static inline void 
simd_add_floats(const float* restrict src0,
				const float* restrict src1,
		    	float* restrict dst,
				size_t count) 
{
	size_t simd_count = (count + 31) / 32; // round up the number of iterations... will just make some extra ops for free
	VERBOSE_LOG(printf("count: %zd simd_count: %zd\n", count, simd_count));
	do
	{
		__m256 arg0_0   = _mm256_load_ps (src0);
		__m256 arg0_1   = _mm256_load_ps (src0 + 8);
		__m256 arg0_2   = _mm256_load_ps (src0 + 16);
		__m256 arg0_3   = _mm256_load_ps (src0 + 24);

		__m256 arg1_0   = _mm256_load_ps (src1);
		__m256 arg1_1   = _mm256_load_ps (src1 + 8);
		__m256 arg1_2   = _mm256_load_ps (src1 + 16);
		__m256 arg1_3   = _mm256_load_ps (src1 + 24);

		__m256 result_0 = _mm256_add_ps  (arg0_0, arg1_0);
		__m256 result_1 = _mm256_add_ps  (arg0_1, arg1_1);
		__m256 result_2 = _mm256_add_ps  (arg0_2, arg1_2);
		__m256 result_3 = _mm256_add_ps  (arg0_3, arg1_3);

		                  _mm256_store_ps(dst,    result_0);
		                  _mm256_store_ps(dst+8,  result_1);
		                  _mm256_store_ps(dst+16, result_2);
		                  _mm256_store_ps(dst+24, result_3);

		src0 += 32;
		src1 += 32;
	    dst  += 32;
		simd_count -= 4;
	} while (simd_count);
}


#define KERNEL(SRC0,SRC1,DST) \
	{ \
	__m256 arg0   = _mm256_loadu_ps(SRC0); \
	__m256 arg1   = _mm256_loadu_ps(SRC1); \
	__m256 result = _mm256_add_ps(arg0, arg1); \
	                _mm256_storeu_ps(DST, result); \
	SRC0 += 8; \
	SRC1 += 8; \
    DST += 8; \
	}
static inline void 
simd_add_floats_u(const float* restrict src0,
				const float* restrict src1,
		    	float* restrict dst,
				size_t count) 
{
	size_t simd_count = (count + 63) / 64; // round up the number of iterations... will just make some extra ops for free
	VERBOSE_LOG(printf("count: %zd simd_count: %zd\n", count, simd_count));
	do
	{
		KERNEL(src0, src1, dst);
		KERNEL(src0, src1, dst);
		KERNEL(src0, src1, dst);
		KERNEL(src0, src1, dst);

		KERNEL(src0, src1, dst);
		KERNEL(src0, src1, dst);
		KERNEL(src0, src1, dst);
		KERNEL(src0, src1, dst);
		/*		__m256 arg0_0   = _mm256_loadu_ps (src0);
		__m256 arg0_1   = _mm256_loadu_ps (src0 + 8);
		__m256 arg0_2   = _mm256_loadu_ps (src0 + 16);
		__m256 arg0_3   = _mm256_loadu_ps (src0 + 24);

		__m256 arg1_0   = _mm256_loadu_ps (src1);
		__m256 arg1_1   = _mm256_loadu_ps (src1 + 8);
		__m256 arg1_2   = _mm256_loadu_ps (src1 + 16);
		__m256 arg1_3   = _mm256_loadu_ps (src1 + 32);

		__m256 result_0 = _mm256_add_ps   (arg0_0, arg1_0);
		__m256 result_1 = _mm256_add_ps   (arg0_1, arg1_1);
		__m256 result_2 = _mm256_add_ps   (arg0_2, arg1_2);
		__m256 result_3 = _mm256_add_ps   (arg0_3, arg1_3);

		                  _mm256_storeu_ps(dst, result_0);
		                  _mm256_storeu_ps(dst+8, result_1);
		                  _mm256_storeu_ps(dst+16, result_2);
		                  _mm256_storeu_ps(dst+24, result_3);

		src0 += 32;
		src1 += 32;
	    dst  += 32;
		*/
		simd_count --;
	} while (simd_count);
}
#undef KERNEL

static void 
simd_add_f__f_f (char**    args, 
				 npy_intp* dimensions,
				 npy_intp* strides,
				 void*     data) __attribute__((flatten));

#define CHUNK_SIZE_IN_BYTES (1024)
#define FLOATS_PER_CHUNK    (CHUNK_SIZE_IN_BYTES / sizeof(float))

static void simd_add_f__f_f (char**    args, 
							npy_intp* dimensions,
							npy_intp* strides,
							void*     data)
{
    npy_intp count = dimensions[0];
	if (0 == count)
		return;

	const float* restrict arg0_pivot = (const float*) args[0];
	const float* restrict arg1_pivot = (const float*) args[1];
	float* restrict result_pivot = (float*) args[2];
	ptrdiff_t arg0_stride = strides[0];
	ptrdiff_t arg1_stride = strides[1];
	ptrdiff_t result_stride = strides[2];

	__m256 workbuffer_arg0[CHUNK_SIZE_IN_BYTES/sizeof(__m256)];
	__m256 workbuffer_arg1[CHUNK_SIZE_IN_BYTES/sizeof(__m256)];
	__m256 workbuffer_result[CHUNK_SIZE_IN_BYTES/sizeof(__m256)];

	do
	{
		size_t this_iter_count = (FLOATS_PER_CHUNK < count) ? FLOATS_PER_CHUNK : count;
		arg0_pivot = linearize_floats(arg0_pivot, arg0_stride, this_iter_count, (float*)workbuffer_arg0);
		
		arg1_pivot = linearize_floats(arg1_pivot, arg1_stride, this_iter_count, (float*)workbuffer_arg1);
		simd_add_floats((const float*)workbuffer_arg0,
						(const float*)workbuffer_arg1,
						(float*)workbuffer_result,
						this_iter_count);
		result_pivot = delinearize_floats((const float*)workbuffer_result, this_iter_count, result_pivot, result_stride);
        
		count -= this_iter_count;
	} while (count);
}


static void rsimd_add_f__f_f (char**    args, 
							  npy_intp* dimensions,
							  npy_intp* strides,
							  void*     data)
{
    npy_intp count = dimensions[0];
	if (0 == count)
		return;

	const float* restrict arg0_pivot = (const float*) args[0];
	const float* restrict arg1_pivot = (const float*) args[1];
	float* restrict result_pivot = (float*) args[2];
	ptrdiff_t arg0_stride = strides[0];
	ptrdiff_t arg1_stride = strides[1];
	ptrdiff_t result_stride = strides[2];

	__m256 workbuffer_arg0[CHUNK_SIZE_IN_BYTES/sizeof(__m256)];
	__m256 workbuffer_arg1[CHUNK_SIZE_IN_BYTES/sizeof(__m256)];
	__m256 workbuffer_result[CHUNK_SIZE_IN_BYTES/sizeof(__m256)];

	do
	{
		size_t this_iter_count = (FLOATS_PER_CHUNK < count) ? FLOATS_PER_CHUNK : count;
		arg0_pivot = linearize_floats(arg0_pivot, arg0_stride, this_iter_count, (float*)workbuffer_arg0);
		
		arg1_pivot = linearize_floats(arg1_pivot, arg1_stride, this_iter_count, (float*)workbuffer_arg1);
		for (size_t iter=0; iter < 10; iter++)
		{
			simd_add_floats((const float*)workbuffer_arg0,
							(const float*)workbuffer_arg1,
							(float*)workbuffer_result,
							this_iter_count);
		}
		result_pivot = delinearize_floats((const float*)workbuffer_result, this_iter_count, result_pivot, result_stride);
        
		count -= this_iter_count;
	} while (count);
}


static void faith_add_f__f_f (char**    args, 
				    		   npy_intp* dimensions,
							   npy_intp* steps,
							   void*     data)
{
	simd_add_floats_u((const float*)(args[0]), (const float*)(args[1]), (float*)(args[2]), dimensions[0]);
}



static void scalar_add_f__f_f (char**    args, 
							   npy_intp* dimensions,
							   npy_intp* steps,
							   void*     data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in0 = args[0];
	char *in1 = args[1];
	char *out = args[2];
    npy_intp in0_step = steps[0];
	npy_intp in1_step = steps[1];
	npy_intp out_step = steps[2];

    for (i = 0; i < n; i++) {
        *((float *)out) = *((float*) in0) + *((float*) in1);

        in0 += in0_step;
        in1 += in1_step;
        out += out_step;
    }
}

/* These are the input and return dtypes of logit.*/

PyUFuncGenericFunction test_simdadd[]   = { &simd_add_f__f_f };
PyUFuncGenericFunction test_rsimdadd[]  = { &rsimd_add_f__f_f };
PyUFuncGenericFunction test_scalaradd[] = { &scalar_add_f__f_f };
PyUFuncGenericFunction test_faithadd[]  = { &faith_add_f__f_f };
static char test_binaryop_signature[]   = { NPY_FLOAT, NPY_FLOAT, NPY_FLOAT };
static void *test_data[] = { NULL };


struct {
	PyUFuncGenericFunction* func_impl;
	char* name;
} funcs_to_register[] =
{
	{ test_scalaradd, "scalar_add" },
	{ test_simdadd,   "simd_add" },
	{ test_rsimdadd,  "rsimd_add" },
	{ test_faithadd,  "faith_add" },
};


void register_functions(PyObject* m)
{
    PyObject* d = PyModule_GetDict(m);

	for (size_t i = 0; i < sizeof(funcs_to_register)/sizeof(funcs_to_register[0]); ++i)
	{
		PyObject* f = PyUFunc_FromFuncAndData(funcs_to_register[i].func_impl,
											  test_data,
											  test_binaryop_signature,
											  1, 2, 1,
											  PyUFunc_None, 
											  funcs_to_register[i].name,
											  "docstring placeholder",
											  0);
		PyDict_SetItemString(d, funcs_to_register[i].name, f);
		Py_DECREF(f);
	}

}

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "simdtest",
    NULL,
    -1,
    simdtest_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyObject *PyInit_simdtest(void)
{
    PyObject* m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    import_array();
    import_umath();

	register_functions(m);
    
    return m;
}
#else
PyMODINIT_FUNC initsimdtest(void)
{
    PyObject* m = Py_InitModule("simdtest", simdtest_methods);
    if (!m) {
        return;
    }

    import_array();
    import_umath();

	register_functions(m);
}
#endif


