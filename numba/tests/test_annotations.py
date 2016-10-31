from __future__ import absolute_import, division

from numba import unittest_support as unittest
from numba.compiler import compile_isolated, compile_extra, Flags
from numba import types, typing
from numba.io_support import StringIO
from numba.targets.registry import cpu_target
from numba.targets import cpu

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
        typingctx = typing.Context()
        targetctx = cpu.CPUContext(typingctx)
        with cpu_target.nested_context(typingctx, targetctx):
            FLAGS = Flags()
            FLAGS.set('force_pyobject')

            def compilethis(func, args):
                return compile_extra(typingctx, targetctx,
                                     func, args, None, FLAGS, {})

            args = [types.int32]
            # need bar to compile foo
            compilethis(bar, args)
            # store result of compiling foo
            cres = compilethis(foo, args)

        ta = cres.type_annotation
        self.assertIs(ta.html_output, None)

        buf = StringIO()
        ta.html_annotate(buf)
        output = buf.getvalue()
        buf.close()
        self.assertIn("bar", output)
        self.assertIn("foo", output)

if __name__ == '__main__':
    unittest.main()
