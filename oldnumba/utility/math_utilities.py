import numba as nb

def py_modulo(restype, argtypes):
    if restype.is_float:
        def py_modulo(a, n):
            r = rem(a, n)

            if (r != 0) and (r < 0) ^ (n < 0):
                r += n

            return r

        instr = 'frem'
    else:
        assert restype.is_int

        def py_modulo(a, n):
            r = rem(a, n)

            if r != 0 and (r ^ n) < 0:
                r += n

            return r

        if restype.is_unsigned:
            instr = 'urem'
        else:
            instr = 'srem'

    rem = nb.declare_instruction(restype(restype, restype), instr)
    return nb.jit(restype(*argtypes))(py_modulo)
