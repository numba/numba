from llvm_cbuilder import *
from llvm_cbuilder import shortnames as C
from llvm_cbuilder.translator import translate
from ctypes import *
from llvm.core import *
from llvm.passes import *
import numpy as np
import unittest
import logging
floatv4 = C.vector(C.float, 4)

class VectorArith(CDefinition):
    _name_ = 'vector_arith'
    _argtys_ = [('a', floatv4),
                ('b', floatv4),
                ('c', floatv4),]
    _retty_ = floatv4

    def body(self, a, b, c):
        '''
        Arguments
        ---------
        a, b, c -- must be vectors
        '''
        @translate
        def _(): # write like python in here
            return a * b + c

class VectorArithDriver1(CDefinition):
    _name_ = 'vector_arith_driver_1'
    _argtys_ = [('A', C.pointer(C.float)),
                ('B', C.pointer(C.float)),
                ('C', C.pointer(C.float)),
                ('D', C.pointer(C.float)),
                ('n', C.int),]

    def body(self, Aary, Bary, Cary, Dary, n):
        '''
        This version uses vector load to fetch array elements as vectors.

        '''
        vecarith = self.depends(VectorArith())
        elem_per_vec = self.constant(C.int, floatv4.count)
        with self.for_range(0, n, elem_per_vec) as (loop, i):
            # Aary[i:] offset the array at i
            a = Aary[i:].vector_load(4, align=1)  # unaligned vector load
            b = Bary[i:].vector_load(4, align=1)
            c = Cary[i:].vector_load(4, align=1)
            r = vecarith(a, b, c)
            Dary[i:].vector_store(r, align=1)
            #    self.debug(r[0], r[1], r[2], r[3])
        self.ret()


class VectorArithDriver2(CDefinition):
    _name_ = 'vector_arith_driver_2'
    _argtys_ = [('A', C.pointer(C.float)),
                ('B', C.pointer(C.float)),
                ('C', C.pointer(C.float)),
                ('D', C.pointer(C.float)),
                ('n', C.int),]

    def body(self, Aary, Bary, Cary, Dary, n):
        '''
        This version loads element of vector individually.
        This style generates scalar ld/st instead of vector ld/st.
        '''
        vecarith = self.depends(VectorArith())
        a = self.var(floatv4)
        b = self.var(floatv4)
        c = self.var(floatv4)
        elem_per_vec = self.constant(C.int, floatv4.count)
        with self.for_range(0, n, elem_per_vec) as (outer, i):
            with self.for_range(elem_per_vec) as (inner, j):
                a[j] = Aary[i + j]
                b[j] = Bary[i + j]
                c[j] = Cary[i + j]
            r = vecarith(a, b, c)
            Dary[i:].vector_store(r, align=1)
            #    self.debug(r[0], r[1], r[2], r[3])
        self.ret()



def aligned_zeros(shape, boundary=16, dtype=float, order='C'):
    '''
    Is there a better way to allocate aligned memory?
    '''
    N = np.prod(shape)
    d = np.dtype(dtype)
    tmp = np.zeros(N * d.itemsize + boundary, dtype=np.uint8)
    address = tmp.__array_interface__['data'][0]
    offset = (boundary - address % boundary) % boundary
    viewed = tmp[offset:offset + N * d.itemsize].view(dtype=d)
    return viewed.reshape(shape, order=order)

class TestVectorArith(unittest.TestCase):
    def test_vector_arith_1(self):
        self.run_and_test_udt(VectorArithDriver1(), 16) # aligned for SSE
        self.run_and_test_udt(VectorArithDriver1(), 20) # misaligned for SSE

    def test_vector_arith_2(self):
        self.run_and_test_udt(VectorArithDriver2(), 16) # aligned for SSE
        self.run_and_test_udt(VectorArithDriver2(), 20) # misaligned for SSE

    def run_and_test_udt(self, udt, align):
        module = Module.new('mod.test.vectoriarith')

        ldriver = udt(module)

        pm = PassManager.new()
        pmb = PassManagerBuilder.new()
        pmb.opt = 3
        pmb.vectorize = True
        pmb.populate(pm)
        pm.run(module)

        print(module.to_native_assembly())

        exe = CExecutor(module)

        float_p = POINTER(c_float)

        driver = exe.get_ctype_function(ldriver,
                                        None,
                                        float_p, float_p, float_p,
                                        float_p,
                                        c_int)

        # prepare for execution

        n = 4*10

        Aary = aligned_zeros(n, boundary=align, dtype=np.float32)
        Bary = aligned_zeros(n, boundary=align, dtype=np.float32)
        Cary = aligned_zeros(n, boundary=align, dtype=np.float32)
        Dary = aligned_zeros(n, boundary=align, dtype=np.float32)

        Aary[:] = range(n)
        Bary[:] = range(n, 2 * n)
        Cary[:] = range(2 * n, 3 * n)

        golden = Aary * Bary + Cary

        getptr = lambda ary: ary.ctypes.data_as(float_p)

        driver(getptr(Aary), getptr(Bary), getptr(Cary), getptr(Dary), n)

        for x, y in zip(golden, Dary):
            self.assertEqual(x, y)


if __name__ == '__main__':
    unittest.main()


