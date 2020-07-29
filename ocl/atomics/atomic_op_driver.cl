int   numba_dppl_atomic_add_i32(volatile __generic int *p, int val, int type);
int   numba_dppl_atomic_sub_i32(volatile __generic int *p, int val, int type);
float numba_dppl_atomic_add_f32(volatile __generic float *p, float val, int type);
float numba_dppl_atomic_sub_f32(volatile __generic float *p, float val, int type);


__kernel void atomic_add_float(__global float *input, __global float *val)
{
    numba_dppl_atomic_add_f32(input, *val, 0);
}

__kernel void atomic_sub_float(__global float *input, __global float *val)
{
    numba_dppl_atomic_sub_f32(input, *val, 0);
}

__kernel void atomic_add_int(__global int *input, __global int *val)
{
    numba_dppl_atomic_add_i32(input, *val, 0);
}

__kernel void atomic_sub_int(__global int *input, __global int *val)
{
    numba_dppl_atomic_sub_i32(input, *val, 0);
}


#ifdef cl_khr_int64_base_atomics
long numba_dppl_atomic_add_i64(volatile __generic long *p, long val, int type);
long numba_dppl_atomic_sub_i64(volatile __generic long *p, long val, int type);

__kernel void atomic_add_long(__global long *input, __global long *val)
{
    numba_dppl_atomic_add_i64(input, *val, 0);
}

__kernel void atomic_sub_long(__global long *input, __global long *val)
{
    numba_dppl_atomic_sub_i64(input, *val, 0);
}

#ifdef cl_khr_fp64
double numba_dppl_atomic_add_f64(volatile __generic double *p, double val, int type);
double numba_dppl_atomic_sub_f64(volatile __generic double *p, double val, int type);

__kernel void atomic_add_double(__global double *input, __global double *val)
{
    numba_dppl_atomic_add_f64(input, *val, 0);
}

__kernel void atomic_sub_double(__global double *input, __global double *val)
{
    numba_dppl_atomic_sub_f64(input, *val, 0);
}

#endif

#endif
