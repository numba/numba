"""
>>> compile_class(False).__name__
'Base'
>>> compile_class(True).__name__
--------------------- Numba Encountered Errors or Warnings ---------------------
<BLANKLINE>
Warning 29:11: Unused argument 'self'
<BLANKLINE>
Warning 29:17: Unused argument 'argument'
<BLANKLINE>
--------------------------------------------------------------------------------
--------------------- Numba Encountered Errors or Warnings ---------------------
<BLANKLINE>
Warning 29:11: Unused argument 'self'
<BLANKLINE>
Warning 29:17: Unused argument 'argument'
<BLANKLINE>
--------------------------------------------------------------------------------
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

def test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    # compile_class()
    test()