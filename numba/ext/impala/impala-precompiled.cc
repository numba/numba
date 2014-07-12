#include "impala-precompiled.h"
#include <cstring>

bool EqStringValImpl(const StringVal& s1, const StringVal& s2) {
    if (s1.is_null != s2.is_null)
        return false;
    if (s1.is_null)
        return true;
    if (s1.len != s2.len)
        return false;
    return (s1.ptr == s2.ptr) || memcmp(s1.ptr, s2.ptr, s1.len) == 0;
}

StringVal GetItemStringValImpl(const StringVal& s, int i) {
    if (s.is_null) return StringVal::null();
    if ((i < -s.len) || (i >= s.len)) return StringVal::null();
    int offset = (i >= 0) ? i : (s.len + i);
    StringVal retval(s.ptr + offset, 1);
    return retval;
}

StringVal AddStringValImpl(FunctionContext* context, const StringVal& s1, const StringVal& s2) {
    int len1 = 0;
    int len2 = 0;
    if (!s1.is_null) len1 = s1.len;
    if (!s2.is_null) len2 = s2.len;
    StringVal retval(context, len1 + len2);
    if (len1 > 0) memcpy(retval.ptr, s1.ptr, len1);
    if (len2 > 0) memcpy(retval.ptr + len1, s2.ptr, len2);
    return retval;
}

// regular (unoptimized)
// /usr/local/Cellar/llvm/3.3/bin/clang++ -emit-llvm -O0 -I numba/ext/impala -c numba/ext/impala/impala-precompiled.cc -o numba/ext/impala/impala-precompiled.bc

// optimized
// /usr/local/Cellar/llvm/3.3/bin/clang++ -emit-llvm -O3 -I numba/ext/impala -c numba/ext/impala/impala-precompiled.cc -o numba/ext/impala/impala-precompiled.bc

// text output
// /usr/local/Cellar/llvm/3.3/bin/clang++ -emit-llvm -S -O3 -I numba/ext/impala -c numba/ext/impala/impala-precompiled.cc -o numba/ext/impala/impala-precompiled.ll