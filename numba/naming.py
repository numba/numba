import re

_ptx_invalid_char = re.compile('[^a-zA-Z0-9_]')

def _fix_naming(string):
    def repl(m):
        return '_%X_' % (ord(m.group(0)))
    return _ptx_invalid_char.sub(repl, string)

def type_mangle(*types):
    return "_".join(str(t).replace(" ", "_") for t in types)

function_counter = 0

def specialized_mangle(func_name, types):
    global function_counter
    # pre = "__numba_specialized_%d_%s" % (func_name, type_mangle(*types))
    pre = "__numba_specialized_%d_%s" % (function_counter, func_name)
    function_counter += 1
    return _fix_naming(pre)

