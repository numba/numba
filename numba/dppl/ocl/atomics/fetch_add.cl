float atomic_cmpxchg_f32(volatile __global float *p, float cmp, float val) {
    union {
        unsigned int u32;
        float        f32;
    } cmp_union, val_union, old_union;

    cmp_union.f32 = cmp;
    val_union.f32 = val;
    old_union.u32 = atomic_cmpxchg((volatile __global unsigned int *) p, cmp_union.u32, val_union.u32);
    return old_union.f32;
}

float atomic_add_f32(volatile __global float *p, float val) {
    float found = *p;
    float expected;
    do {
        expected = found;
        found = atomic_cmpxchg_f32(p, expected, expected + val);
    } while (found != expected);
    return found;
}

int atomic_add_i32(volatile __global int *p, int val) {
    int found = *p;
    int expected;
    do {
        expected = found;
        found = atomic_cmpxchg((volatile __global unsigned int *)p, expected, expected + val);
    } while (found != expected);
    return found;
}

float atomic_sub_f32(volatile __global float *p, float val) {
    float found = *p;
    float expected;
    do {
        expected = found;
        found = atomic_cmpxchg_f32(p, expected, expected - val);
    } while (found != expected);
    return found;
}

int atomic_sub_i32(volatile __global int *p, int val) {
    int found = *p;
    int expected;
    do {
        expected = found;
        found = atomic_cmpxchg((volatile __global unsigned int *)p, expected, expected - val);
    } while (found != expected);
    return found;
}

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
