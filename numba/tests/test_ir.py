from __future__ import print_function
import unittest
from numba import ir


class TestIR(unittest.TestCase):

    def test_IRScope(self):
        top = ir.Scope(parent=None, loc=ir.Loc(line=1))
        local = ir.Scope(parent=top, loc=ir.Loc(line=2))

        apple = local.define('apple', loc=ir.Loc(line=3))
        self.assertTrue(local.refer('apple') is apple)
        self.assertEqual(len(local.localvars), 1)

        orange = top.define('orange', loc=ir.Loc(line=4))
        self.assertEqual(len(local.localvars), 1)
        self.assertEqual(len(top.localvars), 1)
        self.assertTrue(top.refer('orange') is orange)
        self.assertTrue(local.refer('orange') is orange)

        more_orange = local.define('orange', loc=ir.Loc(line=5))
        self.assertTrue(top.refer('orange') is orange)
        self.assertTrue(local.refer('orange') is not orange)
        self.assertTrue(local.refer('orange') is more_orange)

        try:
            bad_orange = local.define('orange', loc=ir.Loc(line=5))
        except ir.RedefinedError:
            pass
        else:
            self.fail("Expecting an %s" % ir.RedefinedError)



if __name__ == '__main__':
    unittest.main()

