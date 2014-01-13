"""
>>> compile_class(False).__name__
'Base'
>>> compile_class(True).__name__
Warning ...: Unused argument '...'
Warning ...: Unused argument '...'
'Base'
"""

from numba import *

def compile_class(warn):
    @jit(warn=warn, warnstyle='simple') # TODO: only issue error once !
    class Base(object):

        @void(int_)
        def method(self, argument):
            pass

    return Base

if __name__ == '__main__':
    # compile_class(True)
    import numba
    numba.testing.testmod()
