# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from numba import autojit, jit

@autojit
def closure_modulo(a, b):
    @jit('int32()')
    def foo():
        return a % b
    return foo()


def test_closure_modulo():
    assert closure_modulo(100, 48) == 4

if __name__ == '__main__':
    test_closure_modulo()
