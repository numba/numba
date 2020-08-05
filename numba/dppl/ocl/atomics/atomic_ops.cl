#ifdef cl_khr_int64_base_atomics
  #pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

  long numba_dppl_atomic_add_i64_local(volatile __generic long *p, long val) {
      long found = *p;
      long expected;
      do {
          expected = found;
          found = atom_cmpxchg((volatile __local ulong *)p, expected, expected + val);
      } while (found != expected);
      return found;
  }

  long numba_dppl_atomic_add_i64_global(volatile __generic long *p, long val) {
      long found = *p;
      long expected;
      do {
          expected = found;
          found = atom_cmpxchg((volatile __global ulong *)p, expected, expected + val);
      } while (found != expected);
      return found;
  }

  long numba_dppl_atomic_sub_i64_local(volatile __generic long *p, long val) {
      long found = *p;
      long expected;
      do {
          expected = found;
          found = atom_cmpxchg((volatile __local ulong *)p, expected, expected - val);
      } while (found != expected);
      return found;
  }

  long numba_dppl_atomic_sub_i64_global(volatile __generic long *p, long val) {
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

    double numba_dppl_atomic_cmpxchg_f64_local(volatile __generic double *p, double cmp, double val) {
        union {
            ulong  u64;
            double f64;
        } cmp_union, val_union, old_union;

        cmp_union.f64 = cmp;
        val_union.f64 = val;
        old_union.u64 = atom_cmpxchg((volatile __local ulong *) p, cmp_union.u64, val_union.u64);
        return old_union.f64;
    }

    double numba_dppl_atomic_cmpxchg_f64_global(volatile __generic double *p, double cmp, double val) {
        union {
            ulong  u64;
            double f64;
        } cmp_union, val_union, old_union;

        cmp_union.f64 = cmp;
        val_union.f64 = val;
        old_union.u64 = atom_cmpxchg((volatile __global ulong *) p, cmp_union.u64, val_union.u64);
        return old_union.f64;
    }

    double numba_dppl_atomic_add_f64_local(volatile __generic double *p, double val) {
        double  found = *p;
        double  expected;
        do {
            expected = found;
            found = numba_dppl_atomic_cmpxchg_f64_local(p, expected, expected + val);
        } while (found != expected);
        return found;
    }

    double numba_dppl_atomic_add_f64_global(volatile __generic double *p, double val) {
        double  found = *p;
        double  expected;
        do {
            expected = found;
            found = numba_dppl_atomic_cmpxchg_f64_global(p, expected, expected + val);
        } while (found != expected);
        return found;
    }


    double numba_dppl_atomic_sub_f64_local(volatile __generic double *p, double val) {
        double  found = *p;
        double  expected;
        do {
            expected = found;
            found = numba_dppl_atomic_cmpxchg_f64_local(p, expected, expected - val);
        } while (found != expected);
        return found;
    }

    double numba_dppl_atomic_sub_f64_global(volatile __generic double *p, double val) {
        double  found = *p;
        double  expected;
        do {
            expected = found;
            found = numba_dppl_atomic_cmpxchg_f64_global(p, expected, expected - val);
        } while (found != expected);
        return found;
    }
  #endif
#endif

float numba_dppl_atomic_cmpxchg_f32_local(volatile __generic float *p, float cmp, float val) {
    union {
        unsigned int u32;
        float        f32;
    } cmp_union, val_union, old_union;

    cmp_union.f32 = cmp;
    val_union.f32 = val;
    old_union.u32 = atomic_cmpxchg((volatile __local unsigned int *) p, cmp_union.u32, val_union.u32);
    return old_union.f32;
}

float numba_dppl_atomic_cmpxchg_f32_global(volatile __generic float *p, float cmp, float val) {
    union {
        unsigned int u32;
        float        f32;
    } cmp_union, val_union, old_union;

    cmp_union.f32 = cmp;
    val_union.f32 = val;
    old_union.u32 = atomic_cmpxchg((volatile __global unsigned int *) p, cmp_union.u32, val_union.u32);
    return old_union.f32;
}

float numba_dppl_atomic_add_f32_local(volatile __generic float *p, float val) {
    float found = *p;
    float expected;
    do {
        expected = found;
        found = numba_dppl_atomic_cmpxchg_f32_local(p, expected, expected + val);
    } while (found != expected);
    return found;
}

float numba_dppl_atomic_add_f32_global(volatile __generic float *p, float val) {
    float found = *p;
    float expected;
    do {
        expected = found;
        found = numba_dppl_atomic_cmpxchg_f32_global(p, expected, expected + val);
    } while (found != expected);
    return found;
}

float numba_dppl_atomic_sub_f32_local(volatile __generic float *p, float val) {
    float found = *p;
    float expected;
    do {
        expected = found;
        found = numba_dppl_atomic_cmpxchg_f32_local(p, expected, expected - val);
    } while (found != expected);
    return found;
}

float numba_dppl_atomic_sub_f32_global(volatile __generic float *p, float val) {
    float found = *p;
    float expected;
    do {
        expected = found;
        found = numba_dppl_atomic_cmpxchg_f32_global(p, expected, expected - val);
    } while (found != expected);
    return found;
}

int numba_dppl_atomic_add_i32_local(volatile __generic int *p, int val) {
    int found = *p;
    int expected;
    do {
        expected = found;
        found = atomic_cmpxchg((volatile __local unsigned int *)p, expected, expected + val);
    } while (found != expected);
    return found;
}

int numba_dppl_atomic_add_i32_global(volatile __generic int *p, int val) {
    int found = *p;
    int expected;
    do {
        expected = found;
        found = atomic_cmpxchg((volatile __global unsigned int *)p, expected, expected + val);
    } while (found != expected);
    return found;
}

int numba_dppl_atomic_sub_i32_local(volatile __generic int *p, int val) {
    int found = *p;
    int expected;
    do {
        expected = found;
        found = atomic_cmpxchg((volatile __local unsigned int *)p, expected, expected - val);
    } while (found != expected);
    return found;
}

int numba_dppl_atomic_sub_i32_global(volatile __generic int *p, int val) {
    int found = *p;
    int expected;
    do {
        expected = found;
        found = atomic_cmpxchg((volatile __global unsigned int *)p, expected, expected - val);
    } while (found != expected);
    return found;
}
