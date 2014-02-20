"""
A simple demonstration of Impala UDF generation.
"""

from numba.ext.impala import (udf, IntVal, FunctionContext, BooleanVal,
                              DoubleVal, TinyIntVal)

@udf(IntVal(FunctionContext, IntVal, IntVal))
def add_udf(context, arg1, arg2):
    if arg1.is_null or arg2.is_null:
        return IntVal.null
    return IntVal(arg1.val + arg2.val)

# Simply print the module IR
print(add_udf.llvm_module)


@udf(BooleanVal(FunctionContext, DoubleVal, TinyIntVal))
def exercise_double_tinyint_bool(context, arg1, arg2):
    if arg1.is_null or arg2.is_null:
        return BooleanVal.null

    lt = arg1.val < arg2.val
    return BooleanVal(lt)


print(exercise_double_tinyint_bool.llvm_module)
