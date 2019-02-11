from __future__ import print_function

import numpy as np

from numba.bytecode import FunctionIdentity, ByteCode
from numba.compiler import translate_stage
from numba.vdg import VarDependencyGraph
import numba.unittest_support as unittest

from .support import TestCase


def _name_only(varset):
    return {x.name for x in varset}


class TestVarDependencyGraph(TestCase):

    def get_vdg(self, func):
        func_id = FunctionIdentity.from_function(func)
        bc = ByteCode(func_id)
        func_ir = translate_stage(func_id, bc)
        return VarDependencyGraph.from_blocks(func_ir.blocks)

    def test_usecase1(self):
        def foo(x, y):
            for i in range(y):
                x += i
            return x

        vdg = self.get_vdg(foo)
        # One of descendants of "x" must be a leaf node
        leaves = vdg.get_leaves()
        self.assertTrue(vdg.get_descendants('x') & leaves)
        # One of predecessors of "x" must be a leaf node
        roots = vdg.get_roots()
        self.assertTrue(vdg.get_predecessors('x') & roots)
        # "x" must be self-depedent but not "y" or "i"
        selfdeps = _name_only(vdg.find_self_dependents())
        self.assertIn('x', selfdeps)
        self.assertNotIn('y', selfdeps)
        self.assertNotIn('i', selfdeps)
        # "x" must be a descendant of "i" and "y"
        self.assertIn('x', _name_only(vdg.get_descendants('i')))
        self.assertIn('x', _name_only(vdg.get_descendants('y')))
        self.assertIn('i', _name_only(vdg.get_descendants('y')))

    def test_usecase2(self):
        def foo(n):
            x = []
            y = [1, 2, n]
            z = x
            if n > 1:
                z.append(1)
            arr = np.array(y)
            return arr

        vdg = self.get_vdg(foo)

        # "x" is created from build_list
        [defx] = vdg.get_defs_no_alias('x')
        self.assertEqual(defx.value.op, 'build_list')

        # "x" is used for .append
        [usex] = vdg.get_uses_no_alias('x')
        self.assertEqual(usex.value.op, 'getattr')
        self.assertEqual(usex.value.attr, 'append')

        # "y" is created from build_list
        [defy] = vdg.get_defs_no_alias('y')
        self.assertEqual(defy.value.op, 'build_list')

        # "y" is used as np.array
        [usey] = vdg.get_uses_no_alias('y')
        self.assertEqual(usey.value.op, 'call')
        func = usey.value.func
        [funcdef] = vdg.get_defs_no_alias(func)
        self.assertEqual(funcdef.value.op, 'getattr')
        self.assertEqual(funcdef.value.attr, 'array')
        [moddef] = vdg.get_defs_no_alias(funcdef.value.value)
        self.assertEqual(moddef.value.name, 'np')
        self.assertEqual(moddef.value.value, np)


if __name__ == '__main__':
    unittest.main()
