#ifdef cl_khr_int64_base_atomics
  #pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

  long atomic_add_i64(volatile __global long *p, long val) {
      long found = *p;
      long expected;
      do {
          expected = found;
          found = atom_cmpxchg((volatile __global ulong *)p, expected, expected + val);
      } while (found != expected);
      return found;
  }

  long atomic_sub_i64(volatile __global long *p, long val) {
      long found = *p;
      long expected;
      do {
          expected = found;
          found = atom_cmpxchg((volatile __global ulong *)p, expected, expected - val);
      } while (found != expected);
      return found;
  }

  #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64: enable

    double atomic_cmpxchg_f64(volatile __global double *p, double cmp, double val) {
        union {
            ulong  u64;
            double f64;
        } cmp_union, val_union, old_union;

        cmp_union.f64 = cmp;
        val_union.f64 = val;
        old_union.u64 = atom_cmpxchg((volatile __global ulong *) p, cmp_union.u64, val_union.u64);
        return old_union.f64;
    }

    double atomic_add_f64(volatile __global double *p, double val) {
        double  found = *p;
        double  expected;
        do {
            expected = found;
            found = atomic_cmpxchg_f64(p, expected, expected + val);
        } while (found != expected);
        return found;
    }

    double atomic_sub_f64(volatile __global double *p, double val) {
        double  found = *p;
        double  expected;
        do {
            expected = found;
            found = atomic_cmpxchg_f64(p, expected, expected - val);
        } while (found != expected);
        return found;
    }
  #endif
#endif

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
