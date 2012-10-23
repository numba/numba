#! /usr/bin/env python
# ______________________________________________________________________

def doslice (in_string, lower, upper):
    l = strlen(in_string)
    if lower < lc_size_t(0):
        lower += l
    if upper < lc_size_t(0):
        upper += l
    temp_len = upper - lower
    if temp_len < lc_size_t(0):
        temp_len = lc_size_t(0)
    ret_val = alloca_array(li8, temp_len + lc_size_t(1))
    strncpy(ret_val, in_string + lower, temp_len)
    ret_val[temp_len] = li8(0)
    return ret_val

def ipow (val, exp):
    ret_val = 1
    temp = val
    w = exp
    while w > 0:
        if (w & 1) != 0:
            ret_val *= temp
            # TODO: Overflow check on ret_val
        w >>= 1
        if w == 0: break
        temp *= temp
        # TODO: Overflow check on temp
    return ret_val

def pymod (arg1, arg2):
    ret_val = arg1 % arg2
    if ret_val < 0:
        if arg2 > 0:
            ret_val += arg2
    elif arg2 < 0:
        ret_val += arg2
    return ret_val

# ______________________________________________________________________
# End of llfuncs.py
