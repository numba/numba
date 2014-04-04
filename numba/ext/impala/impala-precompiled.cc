#include "impala-udf.h"
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



// regular (unoptimized)
// /usr/local/Cellar/llvm/3.3/bin/clang++ -emit-llvm -O0 -I . -c impala-udf.cc -o impala-udf.bc

// optimized
// /usr/local/Cellar/llvm/3.3/bin/clang++ -emit-llvm -O3 -I . -c impala-udf.cc -o impala-udf.bc

// text output
// /usr/local/Cellar/llvm/3.3/bin/clang++ -emit-llvm -S -O0 -I . -c impala-udf.cc -o impala-udf.ll