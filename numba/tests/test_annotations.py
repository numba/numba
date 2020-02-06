import re
from io import StringIO

import numba
from numba.core.compiler import compile_isolated, Flags
from numba.core import types
import unittest

try:
    import jinja2
except ImportError:
    jinja2 = None

try:
    import pygments
except ImportError:
    pygments = None


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
            for i in range(x): # py 38 needs two loops for one to lift?!
                h = h + i
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
            r'\s*<details>'
            r'\s*<summary>'
            r'\s*<code>'
            r'\s*[0-9]+:'
            r'\s*[&nbsp;]+for i in range\(x\):  # this line is tagged\s*',
            re.MULTILINE)

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
        self.assertEqual(len(re.findall(re_lifted_tag, output)), 1,
                         msg='%s not found in %s' % (re_lifted_tag, output))

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

    @unittest.skipIf(pygments is None, "please install the 'pygments' package")
    def test_pretty_print(self):

        @numba.njit
        def foo(x, y):
            return x, y

        foo(1, 2)
        # Exercise the method
        obj = foo.inspect_types(pretty=True)

        # Exercise but supply a not None file kwarg, this is invalid
        with self.assertRaises(ValueError) as raises:
            obj = foo.inspect_types(pretty=True, file='should be None')
        self.assertIn('`file` must be None if `pretty=True`', str(raises.exception))


class TestTypeAnnotation(unittest.TestCase):
    def test_delete(self):
        @numba.njit
        def foo(appleorange, berrycherry):
            return appleorange + berrycherry

        foo(1, 2)
        # Exercise the method
        strbuf = StringIO()
        foo.inspect_types(strbuf)
        # Ensure deletion show up after their use
        lines = strbuf.getvalue().splitlines()

        def findpatloc(pat):
            for i, ln in enumerate(lines):
                if pat in ln:
                    return i
            raise ValueError("can't find {!r}".format(pat))

        sa = findpatloc('appleorange = arg(0, name=appleorange)')
        sb = findpatloc('berrycherry = arg(1, name=berrycherry)')

        ea = findpatloc('del appleorange')
        eb = findpatloc('del berrycherry')

        self.assertLess(sa, ea)
        self.assertLess(sb, eb)


if __name__ == '__main__':
    unittest.main()
