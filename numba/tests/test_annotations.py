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

        buf = StringIO()
        ta.html_annotate(buf)
        output = buf.getvalue()
        buf.close()
        self.assertIn("bar", output)
        self.assertIn("foo", output)
        self.assertIn("LiftedLoop", output)

    def test_html_output_with_lifted_loop(self):
        """
        Test some format and behavior of the html annotation with lifted loop
        """
        @numba.jit
        def udt(x):
            object()  # to force object mode
            z = 0
            for i in range(x):  # this line is tagged
                z += i
            return z

        # Regex pattern to check for the "lifted_tag" in the line of the loop
        re_lifted_tag = re.compile(
            r'<td class="lifted_tag">\s*'
            r'[&nbsp;]+for i in range\(x\):  # this line is tagged\s*'
            r'</td>', re.MULTILINE)

        # Compile int64 version
        sig_i64 = (types.int64,)
        udt.compile(sig_i64)  # compile with lifted loop
        cres = udt.overloads[sig_i64]

        # Make html output
        buf = StringIO()
        cres.type_annotation.html_annotate(buf)
        output = buf.getvalue()
        buf.close()

        # There should be only one function output.
        self.assertEqual(output.count("Function name: udt"), 1)

        sigfmt = "with signature: {} -&gt; pyobject"
        self.assertEqual(output.count(sigfmt.format(sig_i64)), 1)
        # Ensure the loop is tagged
        self.assertEqual(len(re.findall(re_lifted_tag, output)), 1)

        # Compile float64 version
        sig_f64 = (types.float64,)
        udt.compile(sig_f64)
        cres = udt.overloads[sig_f64]

        # Make html output
        buf = StringIO()
        cres.type_annotation.html_annotate(buf)
        output = buf.getvalue()
        buf.close()

        # There should be two function output
        self.assertEqual(output.count("Function name: udt"), 2)
        self.assertEqual(output.count(sigfmt.format(sig_i64)), 1)
        self.assertEqual(output.count(sigfmt.format(sig_f64)), 1)
        # Ensure the loop is tagged in both output
        self.assertEqual(len(re.findall(re_lifted_tag, output)), 2)


if __name__ == '__main__':
    unittest.main()
