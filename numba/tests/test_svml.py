import math
import numpy as np
import subprocess
import numbers
import importlib
import sys
import re
import traceback
import multiprocessing as mp
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
# force LLVM to use AVX512 registers for vectorization
# https://reviews.llvm.org/D67259
vlen2cpu_features = {2: '', 4: '', 8: '-prefer-256-bit'}

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


def func_patterns(func, args, res, dtype, mode, vlen, fastmath, pad=' '*8):
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
    prec_suff = '' if fastmath else '_ha'
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


def usecase_name(dtype, mode, vlen, name):
    """ Returns pretty name for given set of modes """

    return f"{dtype}_{mode}{vlen}_{name}"


def combo_svml_usecase(dtype, mode, vlen, fastmath, name):
    """ Combine multiple function calls under single umbrella usecase """

    name = usecase_name(dtype, mode, vlen, name)
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
        b, c, a = func_patterns(f, ['x'], 'ret', dtype, mode, vlen, fastmath)
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
    def mp_runner(cls, testname, outqueue):
        method = getattr(cls, testname)
        try:
            ok, msg = method()
        except Exception:
            msg = traceback.format_exc()
            ok = False
        outqueue.put({'status': ok, 'msg': msg})

    @classmethod
    def _inject_test(cls, dtype, mode, vlen, flags):
        # unsupported combinations
        if dtype.startswith('complex') and mode != 'numpy':
            return
        # TODO: address skipped tests below
        skipped = dtype.startswith('int') and vlen == 2
        sig = (numba.int64,)
        # unit test body template
        @staticmethod
        def run_template():
            fn, contains, avoids = combo_svml_usecase(dtype, mode, vlen,
                                                      flags['fastmath'],
                                                      flags['name'])
            # look for specific patters in the asm for a given target
            with override_env_config('NUMBA_CPU_NAME', vlen2cpu[vlen]), \
                 override_env_config('NUMBA_CPU_FEATURES', vlen2cpu_features[vlen]):
                # recompile for overridden CPU
                try:
                    jitted_fn = njit(sig, fastmath=flags['fastmath'],
                                     error_model=flags['error_model'],)(fn)
                except:
                    raise Exception("raised while compiling "+fn.__doc__)
            asm = jitted_fn.inspect_asm(sig)
            missed = [pattern for pattern in contains if not pattern in asm]
            found = [pattern for pattern in avoids if pattern in asm]
            ok = not missed and not found
            detail = '\n'.join(
                [line for line in asm.split('\n')
                 if cls.asm_filter.search(line) and not '"' in line])
            msg = (
                f"While expecting {missed} and not {found},\n"
                f"it contains:\n{detail}\n"
                f"when compiling {fn.__doc__}"
            )
            return ok, msg
        # inject it into the class
        postfix = usecase_name(dtype, mode, vlen, flags['name'])
        testname = f"run_{postfix}"
        setattr(cls, testname, run_template)

        @unittest.skipUnless(not skipped, "Not implemented")
        def test_runner(self):
            ctx = mp.get_context("spawn")
            q = ctx.Queue()
            p = ctx.Process(target=type(self).mp_runner, args=[testname, q])
            p.start()
            # timeout to avoid hanging and long enough to avoid bailing too
            # early. Note: this was timeout=10 but that seemed to caused
            # intermittent failures on heavily loaded machines.
            term_or_timeout = p.join(timeout=30)
            exitcode = p.exitcode
            if term_or_timeout is None:
                if exitcode is None:
                    self.fail("Process timed out.")
                elif exitcode < 0:
                    self.fail(f"Process terminated with signal {-exitcode}.")
            self.assertEqual(exitcode, 0, msg="process ended unexpectedly")
            out = q.get()
            status = out['status']
            msg = out['msg']
            self.assertTrue(status, msg=msg)

        setattr(cls, f"test_{postfix}", test_runner)

    @classmethod
    def autogenerate(cls):
        flag_list = [{'fastmath':False, 'error_model':'numpy',
                     'name':'usecase'},
                     {'fastmath':True, 'error_model':'numpy',
                     'name':'fastmath_usecase'},]
        # main loop covering all the modes and use-cases
        for dtype in ('complex64', 'float64', 'float32', 'int32', ):
            for vlen in vlen2cpu:
                for flags in flag_list:
                    for mode in "scalar", "range", "prange", "numpy":
                        cls._inject_test(dtype, mode, vlen, dict(flags))
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
        self.flags.nrt = True

        # flags for njit(fastmath=True)
        self.fastflags = Flags()
        self.fastflags.nrt = True
        self.fastflags.fastmath = cpu.FastMathOptions(True)
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
        # force LLVM to use AVX512 registers for vectorization
        # https://reviews.llvm.org/D67259
        cpu_features = kwargs.pop('cpu_features', '-prefer-256-bit')

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
             override_env_config('NUMBA_CPU_FEATURES', cpu_features):
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
                        f.nrt = True
                        std = compile_isolated(math_sin_loop, sig, flags=f)
                        f.fastmath = cpu.FastMathOptions(True)
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
