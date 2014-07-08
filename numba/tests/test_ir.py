from __future__ import print_function

import textwrap

import numba.unittest_support as unittest
from numba import bytecode, interpreter, ir
from numba.utils import PYVERSION, StringIO


def var_swapping(a, b, c, d, e):
    """
    label 0:
        a.1 = a
        b.1 = b
        c.1 = c
        d.1 = d
        e.1 = e
        a.2 = b.1
        b.2 = a.1
        c.2 = e.1
        d.2 = c.1
        e.2 = d.1
        $0.8 = a.2 + b.2
        $0.10 = $0.8 + c.2
        $0.12 = $0.10 + d.2
        $0.14 = $0.12 + e.2
        return $0.14
    """
    a, b = b, a
    c, d, e = e, c, d
    return a + b + c + d + e

def var_propagate1(a, b):
    """
    label 0:
        a.1 = a
        b.1 = b
        $0.3 = a.1 > b.1
        branch $0.3, 12, 18
    label 12:
        $phi21.2 = a.1
        jump 21
    label 18:
        $phi21.2 = b.1
        jump 21
    label 21:
        $const21.1 = const(int, 5)
        $21.3 = $phi21.2 + $const21.1
        c = $21.3
        return c
    """
    c = (a if a > b else b) + 5
    return c


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


class TestIRDump(unittest.TestCase):
    """
    Exercise the IR dump of some constructs.  These tests are fragile
    (may need to be updated when details of IR generation change, may
    need to be skipped for some Python versions) but help find out
    regressions.
    """

    def get_ir(self, pyfunc):
        bc = bytecode.ByteCode(func=pyfunc)
        interp = interpreter.Interpreter(bc)
        interp.interpret()
        return interp

    def check_ir_dump(self, pyfunc):
        interp = self.get_ir(pyfunc)
        out = StringIO()
        interp.dump(file=out)
        expected = textwrap.dedent(pyfunc.__doc__).strip().splitlines()
        got = out.getvalue().strip().splitlines()
        self.assertEqual(got, expected,
                         "dump might need to be refreshed; here is the "
                         "actual dump:\n%s\n" % (out.getvalue()))

    def test_var_swapping(self):
        # This exercises removal of unused temporaries.
        self.check_ir_dump(var_swapping)

    @unittest.skipIf(PYVERSION < (2, 7),
                     "2.6 bytecode gives slightly different IR dump")
    def test_var_propagate1(self):
        # This exercises generation of phi nodes.
        self.check_ir_dump(var_propagate1)


if __name__ == '__main__':
    unittest.main()
