from __future__ import print_function
import numba.unittest_support as unittest
from numba import ir


class TestIR(unittest.TestCase):

    def test_IRScope(self):
        filename = "<?>"
        top = ir.Scope(parent=None, loc=ir.Loc(filename=filename, line=1))
        local = ir.Scope(parent=top, loc=ir.Loc(filename=filename, line=2))

        apple = local.define('apple', loc=ir.Loc(filename=filename, line=3))
        self.assertTrue(local.get('apple') is apple)
        self.assertEqual(len(local.localvars), 1)

        orange = top.define('orange', loc=ir.Loc(filename=filename, line=4))
        self.assertEqual(len(local.localvars), 1)
        self.assertEqual(len(top.localvars), 1)
        self.assertTrue(top.get('orange') is orange)
        self.assertTrue(local.get('orange') is orange)

        more_orange = local.define('orange', loc=ir.Loc(filename=filename,
                                                        line=5))
        self.assertTrue(top.get('orange') is orange)
        self.assertTrue(local.get('orange') is not orange)
        self.assertTrue(local.get('orange') is more_orange)

        try:
            bad_orange = local.define('orange', loc=ir.Loc(filename=filename,
                                                           line=5))
        except ir.RedefinedError:
            pass
        else:
            self.fail("Expecting an %s" % ir.RedefinedError)



if __name__ == '__main__':
    unittest.main()

