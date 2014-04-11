#ifndef NUMBA_IMPALA_UDF_H
#define NUMBA_IMPALA_UDF_H

#include "udf.h"

using namespace impala_udf;

bool EqStringValImpl(const StringVal& s1, const StringVal& s2);
StringVal GetItemStringValImpl(const StringVal& s, int i);
StringVal AddStringValImpl(FunctionContext* context, const StringVal& s1, const StringVal& s2);

#endif
