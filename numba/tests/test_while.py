#! /usr/bin/env python
# ______________________________________________________________________

from numba import double, long_

import numpy
from nose.tools import nottest

from numba.tests import test_support
from numba.utils import debugout

# ______________________________________________________________________

def while_loop_fn_0(max_index, indexable):
    i = 0
    acc = 0.
    while i < max_index:
        acc += indexable[i]
        i += 1
    return acc

# ______________________________________________________________________

def while_loop_fn_1(indexable):
    i = 0
    acc = 0.
    while i < len(indexable):
        acc += indexable[i]
        i += 1
    return acc

# ______________________________________________________________________

def while_loop_fn_2(ndarr):
    i = 0
    acc = 0.
    while i < ndarr.shape[0]:
        acc += ndarr[i]
        i += 1
    return acc

# ______________________________________________________________________

def while_loop_fn_3(count):
    i = 0
    acc = 1.
    while i < count:
        acc *= 2
        i += 1
    return acc

# ______________________________________________________________________

def while_loop_fn_4(start, stop, inc):
    '''Intended to parallel desired translation target for
    test_forloop.for_loop_fn_1.'''
    acc = 0
    i = start
    while i != stop:
        acc += i
        i += inc
    return acc

# ______________________________________________________________________

def while_loop_fn_5(i_max, j_max):
    j = 1.
    acc = 0.
    while j < j_max:
        i = 1.
        while i < i_max:
            acc += i * j
            i += 1.
        j += 1.
    return acc

# ______________________________________________________________________

def while_loop_fn_6(test_input):
    '''While-loop version of for-loop tests for issue #25.
    https://github.com/numba/numba/issues/25'''
    acc = 0.0
    i = 0.0
    while i < 5.0:
        if i == test_input:
            acc += 100.0
        else:
            acc += i
        i += 1.0
    return acc

# ______________________________________________________________________

def while_loop_fn_7(test_input):
    '''While-loop version of for-loop tests for issue #25.
    https://github.com/numba/numba/issues/25'''    
    acc = 0.0
    i = 0.0
    while i < 5.0:
        tmp = i
        acc += i
        i += 1.0
        if tmp == test_input:
            return acc
    return acc

# ______________________________________________________________________

def while_loop_fn_8(test_input):
    acc = 0.0
    i = 0.0
    while i < 5.0:
        acc += i
        if i == test_input:
            i += 0.5
            continue
        elif i > test_input:
            break
        i += 1.
    return acc

# ______________________________________________________________________

class TestWhile(test_support.ByteCodeTestCase):
    def _do_test(self, function, argtypes, *args, **kws):
        _jit = (self.jit(argtypes = argtypes)
                if argtypes is not None else self.jit())
        compiled_fn = _jit(function)
        self.assertEqual(compiled_fn(*args, **kws), function(*args, **kws))

    def test_while_loop_fn_0(self):
        test_data = numpy.array([1., 2., 3.])
        self._do_test(while_loop_fn_0, [long_, double[:]], len(test_data),
                      test_data)

    def test_while_loop_fn_1(self):
        self._do_test(while_loop_fn_1, [double[:]], numpy.array([1., 2., 3.]))

    def test_while_loop_fn_2(self):
        self._do_test(while_loop_fn_2, [double[:]], numpy.array([1., 2., 3.]))

    def test_while_loop_fn_3(self):
        compiled_fn = self.jit(argtypes = [long_])(while_loop_fn_3)
        compiled_result = compiled_fn(3)
        self.assertEqual(compiled_result, while_loop_fn_3(3))
        self.assertEqual(compiled_result, 8.)

    def test_while_loop_fn_4(self):
        compiled_fn = self.jit(argtypes = (long_, long_, long_),
                               restype = long_)(while_loop_fn_4)
        compiled_result = compiled_fn(1, 4, 1)
        self.assertEqual(compiled_result, while_loop_fn_4(1, 4, 1))
        self.assertEqual(compiled_result, 6)

    def test_while_loop_fn_5(self):
        compiled_fn = self.jit(argtypes = [double, double])(while_loop_fn_5)
        compiled_result = compiled_fn(3, 4)
        self.assertEqual(compiled_result, while_loop_fn_5(3, 4))
        self.assertEqual(compiled_result, 18.)

    def test_while_loop_fn_6(self):
        compiled_fn = self.jit()(while_loop_fn_6)
        self.assertEqual(while_loop_fn_6(4.), compiled_fn(4.))
        self.assertEqual(while_loop_fn_6(5.), compiled_fn(5.))

    def test_while_loop_fn_7(self):
        compiled_fn = self.jit()(while_loop_fn_7)
        self.assertEqual(while_loop_fn_7(4.), compiled_fn(4.))
        self.assertEqual(while_loop_fn_7(5.), compiled_fn(5.))

    def test_while_loop_fn_8(self):
        compiled_fn = self.jit()(while_loop_fn_8)
        self.assertEqual(while_loop_fn_8(3.), compiled_fn(3.))
        self.assertEqual(while_loop_fn_8(4.), compiled_fn(4.))
        self.assertEqual(while_loop_fn_8(5.), compiled_fn(5.))

# ______________________________________________________________________

class TestASTWhile(test_support.ASTTestCase, TestWhile):
    @test_support.checkSkipFlag("Test fails due to problem in Meta.")
    @nottest
    def test_while_loop_fn_7(self, *args, **kws):
        return super(TestASTWhile, self).test_while_loop_fn_7(*args, **kws)


# ______________________________________________________________________

if __name__ == "__main__":
    #TestASTWhile("test_while_loop_fn_1").debug()
    test_support.main()

# ______________________________________________________________________
# End of test_while.py
