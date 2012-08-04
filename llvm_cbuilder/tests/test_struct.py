from llvm.core import *
from llvm_cbuilder import *
import llvm_cbuilder.shortnames as C
import unittest, ctypes

class Vector2D(CStruct):
    _fields_ = [
        ('x', C.float),
        ('y', C.float),
    ]

class Vector2DCtype(ctypes.Structure):
    _fields_ = [
        ('x', ctypes.c_float),
        ('y', ctypes.c_float),
    ]

def gen_vector2d_dist(mod):
    functype = Type.function(C.float, [C.pointer(Vector2D.llvm_type())])
    func = mod.add_function(functype, 'vector2d_dist')

    cb = CBuilder(func)
    vec = cb.var(Vector2D, cb.args[0].load())
    dist = vec.x * vec.x + vec.y * vec.y

    cb.ret(dist)
    cb.close()
    return func


class TestStruct(unittest.TestCase):
    def test_vector2d_dist(self):
        # prepare module
        mod = Module.new('mod')
        lfunc = gen_vector2d_dist(mod)
        mod.verify()
        # run
        exe = CExecutor(mod)
        func = exe.get_ctype_function(lfunc, ctypes.c_float, ctypes.POINTER(Vector2DCtype))

        from random import random
        pydist = lambda x, y: x * x + y * y
        for _ in range(100):
            x, y = random(), random()
            vec = Vector2DCtype(x=x, y=y)
            ans = func(ctypes.pointer(vec))
            gold = pydist(x, y)

            self.assertLess(abs(ans-gold)/gold, 1e-6)

if __name__ == '__main__':
    unittest.main()
