import math
import numpy as np
import subprocess
import numbers
import importlib
import sys
import re
from itertools import chain, combinations

import numba
from numba.core import config, cpu
from numba import prange, njit
from numba.core.compiler import compile_isolated, Flags
from numba.tests.support import TestCase, tag, override_env_config
import unittest

needs_svml = unittest.skipUnless(config.USING_SVML,
                                 "SVML tests need SVML to be present")

# a map of float64 vector lenghs with corresponding CPU architecture
vlen2cpu = {2: 'nehalem', 4: 'haswell', 8: 'skylake-avx512'}

# K: SVML functions, V: python functions which are expected to be SIMD-vectorized
# using SVML, explicit references to Python functions here are mostly for sake of
# instant import checks.
# TODO: [] and comments below mean unused/untested SVML function, it's to be
#       either enabled or to be replaced with the explanation why the function
#       cannot be used in Numba
# TODO: this test does not support functions with more than 1 arguments yet
# The test logic should be modified if there is an SVML function being used under
# different name or module from Python
svml_funcs = {
    "sin":     [np.sin, math.sin],
    "cos":     [np.cos, math.cos],
    "pow":        [],  # pow, math.pow],
    "exp":     [np.exp, math.exp],
    "log":     [np.log, math.log],
    "acos":    [math.acos],
    "acosh":   [math.acosh],
    "asin":    [math.asin],
    "asinh":   [math.asinh],
    "atan2":      [],  # math.atan2],
    "atan":    [math.atan],
    "atanh":   [math.atanh],
    "cbrt":       [],  # np.cbrt],
    "cdfnorm":    [],
    "cdfnorminv": [],
    "ceil":       [],  # np.ceil, math.ceil],
    "cosd":       [],
    "cosh":    [np.cosh, math.cosh],
    "erf":     [math.erf],  # np.erf is available in Intel Distribution
    "erfc":    [math.erfc],
    "erfcinv":    [],
    "erfinv":     [],
    "exp10":      [],
    "exp2":       [],  # np.exp2],
    "expm1":   [np.expm1, math.expm1],
    "floor":      [],  # np.floor, math.floor],
    "fmod":       [],  # np.fmod, math.fmod],
    "hypot":      [],  # np.hypot, math.hypot],
    "invsqrt":    [],  # available in Intel Distribution
    "log10":   [np.log10, math.log10],
    "log1p":   [np.log1p, math.log1p],
    "log2":       [],  # np.log2],
    "logb":       [],
    "nearbyint":  [],
    "rint":       [],  # np.rint],
    "round":      [],  # round],
    "sind":       [],
    "sinh":    [np.sinh, math.sinh],
    "sqrt":    [np.sqrt, math.sqrt],
    "tan":     [np.tan, math.tan],
    "tanh":    [np.tanh, math.tanh],
    "trunc":      [],  # np.trunc, math.trunc],
}
# TODO: these functions are not vectorizable with complex types
complex_funcs_exclude = ["sqrt", "tan", "log10", "expm1", "log1p", "tanh", "log"]

# remove untested entries
svml_funcs = {k: v for k, v in svml_funcs.items() if len(v) > 0}
# lists for functions which belong to numpy and math modules correpondently
numpy_funcs = [f for f, v in svml_funcs.items() if "<ufunc" in \
                  [str(p).split(' ')[0] for p in v]]
other_funcs = [f for f, v in svml_funcs.items() if "<built-in" in \
                  [str(p).split(' ')[0] for p in v]]


def func_patterns(func, args, res, dtype, mode, vlen, flags, pad=' '*8):
    """
    For a given function and its usage modes,
    returns python code and assembly patterns it should and should not generate
    """

    # generate a function call according to the usecase
    if mode == "scalar":
        arg_list = ','.join([a+'[0]' for a in args])
        body = '%s%s[0] += math.%s(%s)\n' % (pad, res, func, arg_list)
    elif mode == "numpy":
        body = '%s%s += np.%s(%s)' % (pad, res, func, ','.join(args))
        body += '.astype(np.%s)\n' % dtype if dtype.startswith('int') else '\n'
    else:
        assert mode == "range" or mode == "prange"
        arg_list = ','.join([a+'[i]' for a in args])
        body = '{pad}for i in {mode}({res}.size):\n' \
               '{pad}{pad}{res}[i] += math.{func}({arg_list})\n'. \
               format(**locals())
    # TODO: refactor so this for-loop goes into umbrella function,
    #       'mode' can be 'numpy', '0', 'i' instead
    # TODO: it will enable mixed usecases like prange + numpy

    # type specialization
    is_f32 = dtype == 'float32' or dtype == 'complex64'
    f = func+'f' if is_f32 else func
    v = vlen*2 if is_f32 else vlen
    # general expectations
    prec_suff = '' if getattr(flags, 'fastmath', False) else '_ha'
    scalar_func = '$_'+f if config.IS_OSX else '$'+f
    svml_func = '__svml_%s%d%s,' % (f, v, prec_suff)
    if mode == "scalar":
        contains = [scalar_func]
        avoids = ['__svml_', svml_func]
    else:            # will vectorize
        contains = [svml_func]
        avoids = []  # [scalar_func] - TODO: if possible, force LLVM to prevent
                     #                     generating the failsafe scalar paths
        if vlen != 8 and (is_f32 or dtype == 'int32'):  # Issue #3016
            avoids += ['%zmm', '__svml_%s%d%s,' % (f, v*2, prec_suff)]
    # special handling
    if func == 'sqrt':
        if mode == "scalar":
            contains = ['sqrts']
            avoids = [scalar_func, svml_func]  # LLVM uses CPU instruction
        elif vlen == 8:
            contains = ['vsqrtp']
            avoids = [scalar_func, svml_func]  # LLVM uses CPU instruction
        # else expect use of SVML for older architectures
    return body, contains, avoids


def usecase_name(dtype, mode, vlen, flags):
    """ Returns pretty name for given set of modes """

    return "{dtype}_{mode}{vlen}_{flags.__name__}".format(**locals())


def combo_svml_usecase(dtype, mode, vlen, flags):
    """ Combine multiple function calls under single umbrella usecase """

    name = usecase_name(dtype, mode, vlen, flags)
    body = """def {name}(n):
        x   = np.empty(n*8, dtype=np.{dtype})
        ret = np.empty_like(x)\n""".format(**locals())
    funcs = set(numpy_funcs if mode == "numpy" else other_funcs)
    if dtype.startswith('complex'):
        funcs = funcs.difference(complex_funcs_exclude)
    contains = set()
    avoids = set()
    # fill body and expectation patterns
    for f in funcs:
        b, c, a = func_patterns(f, ['x'], 'ret', dtype, mode, vlen, flags)
        avoids.update(a)
        body += b
        contains.update(c)
    body += " "*8 + "return ret"
    # now compile and return it along with its body in __doc__  and patterns
    ldict = {}
    exec(body, globals(), ldict)
    ldict[name].__doc__ = body
    return ldict[name], contains, avoids


@needs_svml
class TestSVMLGeneration(TestCase):
    """ Tests all SVML-generating functions produce desired calls """

    # env mutating, must not run in parallel
    _numba_parallel_test_ = False
    # RE for a generic symbol reference and for each particular SVML function
    asm_filter = re.compile('|'.join(['\$[a-z_]\w+,']+list(svml_funcs)))

    @classmethod
    def _inject_test(cls, dtype, mode, vlen, flags):
        # unsupported combinations
        if dtype.startswith('complex') and mode != 'numpy':
            return 
        # TODO: address skipped tests below
        skipped = dtype.startswith('int') and vlen == 2
        args = (dtype, mode, vlen, flags)
        # unit test body template
        @unittest.skipUnless(not skipped, "Not implemented")
        def test_template(self):
            fn, contains, avoids = combo_svml_usecase(*args)
            # look for specific patters in the asm for a given target
            with override_env_config('NUMBA_CPU_NAME', vlen2cpu[vlen]), \
                 override_env_config('NUMBA_CPU_FEATURES', ''):
                # recompile for overridden CPU
                try:
                    jit = compile_isolated(fn, (numba.int64, ), flags=flags)
                except:
                    raise Exception("raised while compiling "+fn.__doc__)
            asm = jit.library.get_asm_str()
            missed = [pattern for pattern in contains if not pattern in asm]
            found = [pattern for pattern in avoids if pattern in asm]
            self.assertTrue(not missed and not found,
                "While expecting %s and no %s,\n"
                "it contains:\n%s\n"
                "when compiling %s" % (str(missed), str(found), '\n'.join(
                    [line for line in asm.split('\n')
                     if cls.asm_filter.search(line) and not '"' in line]),
                     fn.__doc__))
        # inject it into the class
        setattr(cls, "test_"+usecase_name(*args), test_template)

    @classmethod
    def autogenerate(cls):
        test_flags = ['fastmath', ]  # TODO: add 'auto_parallel' ?
        # generate all the combinations of the flags
        test_flags = sum([list(combinations(test_flags, x)) for x in range( \
                                                    len(test_flags)+1)], [])
        flag_list = []  # create Flag class instances
        for ft in test_flags:
            flags = Flags()
            flags.set('nrt')
            flags.set('error_model', 'numpy')
            flags.__name__ = '_'.join(ft+('usecase',))
            for f in ft:
                flags.set(f, {
                    'fastmath': cpu.FastMathOptions(True)
                }.get(f, True))
            flag_list.append(flags)
        # main loop covering all the modes and use-cases
        for dtype in ('complex64', 'float64', 'float32', 'int32', ):
            for vlen in vlen2cpu:
                for flags in flag_list:
                    for mode in "scalar", "range", "prange", "numpy":
                        cls._inject_test(dtype, mode, vlen, flags)
        # mark important
        for n in ( "test_int32_range4_usecase",  # issue #3016
                    ):
            setattr(cls, n, tag("important")(getattr(cls, n)))


TestSVMLGeneration.autogenerate()


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
        self.fastflags.set('fastmath', cpu.FastMathOptions(True))
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
        pat = '$_sin' if config.IS_OSX else '$sin'
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
                    from numba.core import cpu
                    from numba.tests.support import override_env_config
                    from numba.core.compiler import compile_isolated, Flags

                    # compile for overridden CPU, with and without fastmath
                    with override_env_config('NUMBA_CPU_NAME', 'skylake-avx512'), \
                         override_env_config('NUMBA_CPU_FEATURES', ''):
                        sig = (numba.int32,)
                        f = Flags()
                        f.set('nrt')
                        std = compile_isolated(math_sin_loop, sig, flags=f)
                        f.set('fastmath', cpu.FastMathOptions(True))
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

    def test_svml_working_in_non_isolated_context(self):
        @njit(fastmath={'fast'}, error_model="numpy")
        def impl(n):
            x   = np.empty(n * 8, dtype=np.float64)
            ret = np.empty_like(x)
            for i in range(ret.size):
                    ret[i] += math.cosh(x[i])
            return ret
        impl(1)
        self.assertTrue('intel_svmlcc' in impl.inspect_llvm(impl.signatures[0]))


if __name__ == '__main__':
    unittest.main()
