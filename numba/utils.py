import opcode


def itercode(code):
    """Return a generator of byte-offset, opcode, and argument 
    from a byte-code-string
    """
    i = 0
    extended_arg = 0
    n = len(code)
    while i < n:
        print ' -> i', i
        c = code[i]
        num = i
        op = ord(c)
        i = i + 1
        oparg = None
        if op >= opcode.HAVE_ARGUMENT:
            oparg = ord(code[i]) + ord(code[i+1])*256 + extended_arg
            extended_arg = 0
            i = i + 2
            if op == opcode.EXTENDED_ARG:
                extended_arg = oparg*65536L

        delta = yield num, op, oparg
        if delta is not None:
            print '->delta', delta
            abs_rel, dst = delta
            if abs_rel == 'abs':
                i = dst
            elif abs_rel == 'rel':
                i += dst
            else:
                assert 0

