import re

_ptx_invalid_char = re.compile('[^a-zA-Z0-9_]')

def _fix_naming(string):
    def repl(m):
        return '_%X_' % (ord(m.group(0)))
    return _ptx_invalid_char.sub(repl, string)

def type_mangle(*types):
    return "_".join(str(t).replace(" ", "_") for t in types)

def specialized_mangle(func_name, types):
    pre = "__numba_specialized_%s_%s" % (func_name, type_mangle(*types))
    return _fix_naming(pre)
