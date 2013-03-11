"""
>>> compile_class(False).__name__
'Base'
>>> compile_class(True).__name__
Warning 17:11: Unused argument 'self'
Warning 17:17: Unused argument 'argument'
'Base'
"""

from numba import *

def compile_class(warn):
    @jit(warn=warn) # TODO: only issue error once !
    class Base(object):

        @void(int_)
        def method(self, argument):
            pass

    return Base

if __name__ == '__main__':
    # compile_class(False)
    import doctest
    doctest.testmod()
