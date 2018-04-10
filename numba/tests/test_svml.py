from __future__ import division, print_function

import math
import numpy as np
import subprocess
import numbers
import importlib
import sys

from .support import TestCase, tag, override_env_config
import numba
from numba.compiler import compile_isolated, Flags
from numba import unittest_support as unittest

needs_svml = unittest.skipUnless(numba.config.USING_SVML,
                                 "SVML tests need SVML to be present")


def math_sin_scalar(x):
    return math.sin(x)


def math_sin_loop(n):
    ret = np.empty(n, dtype=np.float64)
    for x in range(n):
        ret[x] = math.sin(np.float64(x))
    return ret


@needs_svml
class TestSVML(TestCase):
    """ Tests SVML behaves as expected """

    # env mutating, must not run in parallel
    _numba_parallel_test_ = False

    def __init__(self, *args):
        self.flags = Flags()
        self.flags.set('nrt')

        # flags for njit(fastmath=True)
        self.fastflags = Flags()
        self.fastflags.set('nrt')
        self.fastflags.set('fastmath')
        super(TestSVML, self).__init__(*args)

    def compile(self, func, *args, **kwargs):
        assert not kwargs
        sig = tuple([numba.typeof(x) for x in args])

        std = compile_isolated(func, sig, flags=self.flags)
        fast = compile_isolated(func, sig, flags=self.fastflags)

        return std, fast

    def copy_args(self, *args):
        if not args:
            return tuple()
        new_args = []
        for x in args:
            if isinstance(x, np.ndarray):
                new_args.append(x.copy('k'))
            elif isinstance(x, np.number):
                new_args.append(x.copy())
            elif isinstance(x, numbers.Number):
                new_args.append(x)
            else:
                raise ValueError('Unsupported argument type encountered')
        return tuple(new_args)

    def check(self, pyfunc, *args, **kwargs):

        jitstd, jitfast = self.compile(pyfunc, *args)

        std_pattern = kwargs.pop('std_pattern', None)
        fast_pattern = kwargs.pop('fast_pattern', None)
        cpu_name = kwargs.pop('cpu_name', 'skylake-avx512')

        # python result
        py_expected = pyfunc(*self.copy_args(*args))

        # jit result
        jitstd_result = jitstd.entry_point(*self.copy_args(*args))

        # fastmath result
        jitfast_result = jitfast.entry_point(*self.copy_args(*args))

        # assert numerical equality
        np.testing.assert_almost_equal(jitstd_result, py_expected, **kwargs)
        np.testing.assert_almost_equal(jitfast_result, py_expected, **kwargs)

        # look for specific patters in the asm for a given target
        with override_env_config('NUMBA_CPU_NAME', cpu_name), \
             override_env_config('NUMBA_CPU_FEATURES', ''):
            # recompile for overridden CPU
            jitstd, jitfast = self.compile(pyfunc, *args)
            if std_pattern:
                self.check_svml_presence(jitstd, std_pattern)
            if fast_pattern:
                self.check_svml_presence(jitfast, fast_pattern)

    def check_svml_presence(self, func, pattern):
        asm = func.library.get_asm_str()
        self.assertIn(pattern, asm)

    def test_scalar_context(self):
        # SVML will not be used.
        pat = '$_sin' if numba.config.IS_OSX else '$sin'
        self.check(math_sin_scalar, 7., std_pattern=pat)
        self.check(math_sin_scalar, 7., fast_pattern=pat)

    def test_svml(self):
        # loops both with and without fastmath should use SVML.
        # The high accuracy routines are dropped if `fastmath` is set
        std = "__svml_sin8_ha,"
        fast = "__svml_sin8,"  # No `_ha`!
        self.check(math_sin_loop, 10, std_pattern=std, fast_pattern=fast)

    def test_svml_disabled(self):
        code = """if 1:
            import os
            import numpy as np
            import math

            def math_sin_loop(n):
                ret = np.empty(n, dtype=np.float64)
                for x in range(n):
                    ret[x] = math.sin(np.float64(x))
                return ret

            def check_no_svml():
                try:
                    # ban the use of SVML
                    os.environ['NUMBA_DISABLE_INTEL_SVML'] = '1'

                    # delay numba imports to account for env change as
                    # numba.__init__ picks up SVML and it is too late by
                    # then to override using `numba.config`
                    import numba
                    from numba import config
                    from numba.tests.support import override_env_config
                    from numba.compiler import compile_isolated, Flags

                    # compile for overridden CPU, with and without fastmath
                    with override_env_config('NUMBA_CPU_NAME', 'skylake-avx512'), \
                         override_env_config('NUMBA_CPU_FEATURES', ''):
                        sig = (numba.int32,)
                        f = Flags()
                        f.set('nrt')
                        std = compile_isolated(math_sin_loop, sig, flags=f)
                        f.set('fastmath')
                        fast = compile_isolated(math_sin_loop, sig, flags=f)
                        fns = std, fast

                        # assert no SVML call is present in the asm
                        for fn in fns:
                            asm = fn.library.get_asm_str()
                            assert '__svml_sin' not in asm
                finally:
                    # not really needed as process is separate
                    os.environ['NUMBA_DISABLE_INTEL_SVML'] = '0'
                    config.reload_config()
            check_no_svml()
            """
        popen = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = popen.communicate()
        if popen.returncode != 0:
            raise AssertionError(
                "process failed with code %s: stderr follows\n%s\n" %
                (popen.returncode, err.decode()))

if __name__ == '__main__':
    unittest.main()
