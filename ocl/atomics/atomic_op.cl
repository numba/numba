#ifdef cl_khr_int64_base_atomics
  #pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

  long numba_dppl_atomic_add_i64(volatile __generic long *p, long val, int type) {
      long found = *p;
      long expected;
      do {
          expected = found;
          if (type == 1) { /* The address qualifier should be __local */
              found = atom_cmpxchg((volatile __local ulong *)p, expected, expected + val);
          } else {
              found = atom_cmpxchg((volatile __global ulong *)p, expected, expected + val);
          }
      } while (found != expected);
      return found;
  }

  long numba_dppl_atomic_sub_i64(volatile __generic long *p, long val, int type) {
      long found = *p;
      long expected;
      do {
          expected = found;
          if (type == 1) { /* The address qualifier should be __local */
              found = atom_cmpxchg((volatile __local ulong *)p, expected, expected - val);
          } else {
              found = atom_cmpxchg((volatile __global ulong *)p, expected, expected - val);
          }
      } while (found != expected);
      return found;
  }

  #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64: enable

    double numba_dppl_atomic_cmpxchg_f64(volatile __generic double *p, double cmp, double val, int type) {
        union {
            ulong  u64;
            double f64;
        } cmp_union, val_union, old_union;

        cmp_union.f64 = cmp;
        val_union.f64 = val;
        if (type == 1) { /* The address qualifier should be __local */
            old_union.u64 = atom_cmpxchg((volatile __local ulong *) p, cmp_union.u64, val_union.u64);
        } else {
            old_union.u64 = atom_cmpxchg((volatile __global ulong *) p, cmp_union.u64, val_union.u64);
        }
        return old_union.f64;
    }

    double numba_dppl_atomic_add_f64(volatile __generic double *p, double val, int type) {
        double  found = *p;
        double  expected;
        do {
            expected = found;
            found = numba_dppl_atomic_cmpxchg_f64(p, expected, expected + val, type);
        } while (found != expected);
        return found;
    }

    double numba_dppl_atomic_sub_f64(volatile __generic double *p, double val, int type) {
        double  found = *p;
        double  expected;
        do {
            expected = found;
            found = numba_dppl_atomic_cmpxchg_f64(p, expected, expected - val, type);
        } while (found != expected);
        return found;
    }
  #endif
#endif

float numba_dppl_atomic_cmpxchg_f32(volatile __generic float *p, float cmp, float val, int type) {
    union {
        unsigned int u32;
        float        f32;
    } cmp_union, val_union, old_union;

    cmp_union.f32 = cmp;
    val_union.f32 = val;
    if (type == 1) { /* The address qualifier should be __local */
        old_union.u32 = atomic_cmpxchg((volatile __local unsigned int *) p, cmp_union.u32, val_union.u32);
    } else {
        old_union.u32 = atomic_cmpxchg((volatile __global unsigned int *) p, cmp_union.u32, val_union.u32);
    }
    return old_union.f32;
}

float numba_dppl_atomic_add_f32(volatile __generic float *p, float val, int type) {
    float found = *p;
    float expected;
    do {
        expected = found;
        found = numba_dppl_atomic_cmpxchg_f32(p, expected, expected + val, type);
    } while (found != expected);
    return found;
}

int numba_dppl_atomic_add_i32(volatile __generic int *p, int val, int type) {
    int found = *p;
    int expected;
    do {
        expected = found;
        if (type == 1) { /* The address qualifier should be __local */
            found = atomic_cmpxchg((volatile __local unsigned int *)p, expected, expected + val);
        } else {
            found = atomic_cmpxchg((volatile __global unsigned int *)p, expected, expected + val);
        }
    } while (found != expected);
    return found;
}

float numba_dppl_atomic_sub_f32(volatile __generic float *p, float val, int type) {
    float found = *p;
    float expected;
    do {
        expected = found;
        found = numba_dppl_atomic_cmpxchg_f32(p, expected, expected - val, type);
    } while (found != expected);
    return found;
}

int numba_dppl_atomic_sub_i32(volatile __generic int *p, int val, int type) {
    int found = *p;
    int expected;
    do {
        expected = found;
        if (type == 1) { /* The address qualifier should be __local */
            found = atomic_cmpxchg((volatile __local unsigned int *)p, expected, expected - val);
        } else {
            found = atomic_cmpxchg((volatile __global unsigned int *)p, expected, expected - val);
        }
    } while (found != expected);
    return found;
}
