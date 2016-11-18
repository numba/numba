from __future__ import absolute_import, division

import re

import numba
from numba import unittest_support as unittest
from numba.compiler import compile_isolated, Flags
from numba import types
from numba.io_support import StringIO

try:
    import jinja2
except ImportError:
    jinja2 = None


@unittest.skipIf(jinja2 is None, "please install the 'jinja2' package")
class TestAnnotation(unittest.TestCase):

    def test_exercise_code_path(self):
        """
        Ensures template.html is available
        """

        def foo(n, a):
            s = a
            for i in range(n):
                s += i
            return s

        cres = compile_isolated(foo, [types.int32, types.int32])
        ta = cres.type_annotation
        self.assertIs(ta.html_output, None)

        buf = StringIO()
        ta.html_annotate(buf)
        output = buf.getvalue()
        buf.close()
        self.assertIn("foo", output)

    def test_exercise_code_path_with_lifted_loop(self):
        """
        Ensures that lifted loops are handled correctly in obj mode
        """
        # the functions to jit
        def bar(x):
            return x

        def foo(x):
            h = 0.
            for k in range(x):
                h = h + k
            if x:
                h = h - bar(x)
            return h

        # compile into an isolated context
        flags = Flags()
        flags.set('enable_pyobject')
        flags.set('enable_looplift')
        cres = compile_isolated(foo, [types.intp], flags=flags)

        ta = cres.type_annotation
        self.assertIs(ta.html_output, None)

        buf = StringIO()
        ta.html_annotate(buf)
        output = buf.getvalue()
        buf.close()
        self.assertIn("bar", output)
        self.assertIn("foo", output)
        self.assertIn("LiftedLoop", output)

if __name__ == '__main__':
    unittest.main()
