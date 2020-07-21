int atomic_add_i32(volatile __global int *p, int val);
int atomic_sub_i32(volatile __global int *p, int val);
float atomic_add_f32(volatile __global float *p, float val);
float atomic_sub_f32(volatile __global float *p, float val);


__kernel void atomic_add_float(__global float *input, __global float *val)
{
    atomic_add_f32(input, *val);
}

__kernel void atomic_sub_float(__global float *input, __global float *val)
{
    atomic_sub_f32(input, *val);
}

__kernel void atomic_add_int(__global int *input, __global int *val)
{
    atomic_add_i32(input, *val);
}

__kernel void atomic_sub_int(__global int *input, __global int *val)
{
    atomic_sub_i32(input, *val);
}


#ifdef cl_khr_int64_base_atomics
long atomic_add_i64(volatile __global long *p, long val);
long atomic_sub_i64(volatile __global long *p, long val);

__kernel void atomic_add_long(__global long *input, __global long *val)
{
    atomic_add_i64(input, *val);
}

__kernel void atomic_sub_long(__global long *input, __global long *val)
{
    atomic_sub_i64(input, *val);
}

#ifdef cl_khr_fp64
double atomic_add_f64(volatile __global double *p, double val);
double atomic_sub_f64(volatile __global double *p, double val);

__kernel void atomic_add_double(__global double *input, __global double *val)
{
    atomic_add_f64(input, *val);
}

__kernel void atomic_sub_double(__global double *input, __global double *val)
{
    atomic_sub_f64(input, *val);
}

#endif

#endif
