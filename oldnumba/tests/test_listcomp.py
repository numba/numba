# Based on cython/tests/run/listcomp.pyx

from numba import *
from numba.testing.test_support import testmod, autojit_py3doc

@autojit_py3doc
def smoketest():
    """
    >>> smoketest()
    ([0, 4, 8], 4)
    """
    x = -10 # 'abc'
    result = [x*2 for x in range(5) if x % 2 == 0]
    # assert x != -10 # 'abc'
    return result, x

@autojit_py3doc(warnstyle="simple")
def list_genexp():
    """
    >>> list_genexp()
    Traceback (most recent call last):
        ...
    NumbaError: ...: Generator comprehensions are not yet supported
    """
    x = -10 # 'abc'
    result = list(x*2 for x in range(5) if x % 2 == 0)
    # assert x == 'abc'
    return result, x

@autojit_py3doc(locals={"x": int_})
def int_runvar():
    """
    >>> int_runvar()
    [0, 4, 8]
    """
    print([x*2 for x in range(5) if x % 2 == 0])

#@jit
#class A(object):
#    @object_()
#    def __repr__(self):
#        return u"A"

#@autojit
#def typed():
#    """
#    >>> typed()
#    [A, A, A]
#    """
#    cdef A obj
#    print [obj for obj in [A(), A(), A()]]

#@autojit
#def iterdict():
#    """
#    >>> iterdict()
#    [1, 2, 3]
#    """
#    d = dict(a=1,b=2,c=3)
#    l = [d[key] for key in d]
#    l.sort()
#    print l

@autojit_py3doc
def nested_result():
    """
    >>> nested_result()
    [[], [-1], [-1, 0], [-1, 0, 1]]
    """
    result = [[a-1 for a in range(b)] for b in range(4)]
    return result

# TODO: object and string iteration
#@autojit_py3doc
#def listcomp_as_condition(sequence):
#    """
#    >>> listcomp_as_condition(['a', 'b', '+'])
#    True
#    >>> listcomp_as_condition('ab+')
#    True
#    >>> listcomp_as_condition('abc')
#    False
#    """
#    if [1 for c in sequence if c in '+-*/<=>!%&|([^~,']:
#        return True
#    return False

#@autojit_py3doc
#def sorted_listcomp(sequence):
#    """
#    >>> sorted_listcomp([3,2,4])
#    [3, 4, 5]
#    """
#    return sorted([ n+1 for n in sequence ])

@autojit_py3doc
def listcomp_const_condition_false():
    """
    >>> listcomp_const_condition_false()
    []
    """
    return [x*2 for x in range(3) if 0]

@autojit_py3doc
def listcomp_const_condition_true():
    """
    >>> listcomp_const_condition_true()
    [0, 2, 4]
    """
    return [x*2 for x in range(3) if 1]


if __name__ == '__main__':
#    print test_pointer_arithmetic()
#    a = np.array([1, 2, 3, 4], dtype=np.float32)
#    print test_pointer_indexing(a.ctypes.data, float32.pointer())
    pass

#int_runvar()
#smoketest()
#list_genexp()
#test_listcomp()
# smoketest()
testmod()
