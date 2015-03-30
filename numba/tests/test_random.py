from __future__ import print_function

import functools
import math
import os
import random
import subprocess
import sys

import numpy as np

import numba.unittest_support as unittest
from numba import jit, _helperlib, types
from numba.compiler import compile_isolated
from .support import TestCase, compile_function


# State size of the Mersenne Twister
N = 624

py_state_ptr = _helperlib.c_helpers['py_random_state']
np_state_ptr = _helperlib.c_helpers['np_random_state']


def numpy_randint1(a):
    return np.random.randint(a)

def numpy_randint2(a, b):
    return np.random.randint(a, b)

def random_randint(a, b):
    return random.randint(a, b)

def random_randrange1(a):
    return random.randrange(a)

def random_randrange2(a, b):
    return random.randrange(a, b)

def random_randrange3(a, b, c):
    return random.randrange(a, b, c)


def jit_with_args(name, argstring):
    code = """def func(%(argstring)s):
        return %(name)s(%(argstring)s)
""" % locals()
    pyfunc = compile_function("func", code, globals())
    return jit(nopython=True)(pyfunc)

def jit_nullary(name):
    return jit_with_args(name, "")

def jit_unary(name):
    return jit_with_args(name, "a")

def jit_binary(name):
    return jit_with_args(name, "a, b")

def jit_ternary(name):
    return jit_with_args(name, "a, b, c")


random_gauss = jit_binary("random.gauss")
random_random = jit_nullary("random.random")
random_seed = jit_unary("random.seed")

numpy_normal = jit_binary("np.random.normal")
numpy_random = jit_nullary("np.random.random")
numpy_seed = jit_unary("np.random.seed")


def _copy_py_state(r, ptr):
    """
    Copy state of Python random *r* to Numba state *ptr*.
    """
    mt = r.getstate()[1]
    ints, index = mt[:-1], mt[-1]
    _helperlib.rnd_set_state(ptr, (index, list(ints)))
    return ints, index

def _copy_np_state(r, ptr):
    """
    Copy state of Numpy random *r* to Numba state *ptr*.
    """
    ints, index = r.get_state()[1:3]
    _helperlib.rnd_set_state(ptr, (index, [int(x) for x in ints]))
    return ints, index

def sync_to_numpy(r):
    _ver, mt_st, _gauss_next = r.getstate()
    mt_pos = mt_st[-1]
    mt_ints = mt_st[:-1]
    assert len(mt_ints) == 624

    np_st = ('MT19937', np.array(mt_ints, dtype='uint32'), mt_pos)
    if _gauss_next is None:
        np_st += (0, 0.0)
    else:
        np_st += (1, _gauss_next)

    np.random.set_state(np_st)

# Pure Python equivalents of some of the Numpy distributions, using
# Python's basic generators.

def py_chisquare(r, df):
    return 2.0 * r.gammavariate(df / 2.0, 1.0)

def py_f(r, num, denom):
    return ((py_chisquare(r, num) * denom) /
            (py_chisquare(r, denom) * num))


class TestInternals(TestCase):
    """
    Test low-level internals of the implementation.
    """

    def _check_get_set_state(self, ptr):
        state = _helperlib.rnd_get_state(ptr)
        i, ints = state
        self.assertIsInstance(i, int)
        self.assertIsInstance(ints, list)
        self.assertEqual(len(ints), N)
        j = (i * 100007) % N
        ints = [i * 3 for i in range(N)]
        # Roundtrip
        _helperlib.rnd_set_state(ptr, (j, ints))
        self.assertEqual(_helperlib.rnd_get_state(ptr), (j, ints))

    def _check_shuffle(self, ptr):
        # We test shuffling against CPython
        r = random.Random()
        ints, index = _copy_py_state(r, ptr)
        # Force shuffling in CPython generator
        for i in range(index, N + 1, 2):
            r.random()
        _helperlib.rnd_shuffle(ptr)
        # Check new integer keys
        mt = r.getstate()[1]
        ints, index = mt[:-1], mt[-1]
        self.assertEqual(_helperlib.rnd_get_state(ptr)[1], list(ints))

    def _check_init(self, ptr):
        # We use the same integer seeding as Numpy
        # (CPython is different: it treats the integer as a byte array)
        r = np.random.RandomState()
        for i in [0, 1, 125, 2**32 - 5]:
            # Need to cast to a C-sized int (for Numpy <= 1.7)
            r.seed(np.uint32(i))
            st = r.get_state()
            ints = list(st[1])
            index = st[2]
            assert index == N  # sanity check
            _helperlib.rnd_seed(ptr, i)
            self.assertEqual(_helperlib.rnd_get_state(ptr), (index, ints))

    def _check_perturb(self, ptr):
        states = []
        for i in range(10):
            # Initialize with known state
            _helperlib.rnd_seed(ptr, 0)
            # Perturb with entropy
            _helperlib.rnd_seed(ptr, os.urandom(512))
            states.append(tuple(_helperlib.rnd_get_state(ptr)[1]))
        # No two identical states
        self.assertEqual(len(set(states)), len(states))

    def test_get_set_state(self):
        self._check_get_set_state(py_state_ptr)

    def test_shuffle(self):
        self._check_shuffle(py_state_ptr)

    def test_init(self):
        self._check_init(py_state_ptr)

    def test_perturb(self):
        self._check_perturb(py_state_ptr)


class TestRandom(TestCase):

    # NOTE: there may be cascading imprecision issues (e.g. between x87-using
    # C code and SSE-using LLVM code), which is especially brutal for some
    # iterative algorithms with sensitive exit conditions.
    # Therefore we stick to hardcoded integers for seed values below.

    def _follow_cpython(self, ptr, seed=2):
        r = random.Random(seed)
        _copy_py_state(r, ptr)
        return r

    def _follow_numpy(self, ptr, seed=2):
        r = np.random.RandomState(seed)
        _copy_np_state(r, ptr)
        return r

    def _check_random_seed(self, seedfunc, randomfunc):
        """
        Check seed()- and random()-like functions.
        """
        # Our seed() mimicks Numpy's.
        r = np.random.RandomState()
        for i in [0, 1, 125, 2**32 - 1]:
            # Need to cast to a C-sized int (for Numpy <= 1.7)
            r.seed(np.uint32(i))
            seedfunc(i)
            # Be sure to trigger a reshuffle
            for j in range(N + 10):
                self.assertPreciseEqual(randomfunc(), r.uniform(0.0, 1.0))

    def test_random_random(self):
        self._check_random_seed(random_seed, random_random)

    def test_numpy_random(self):
        self._check_random_seed(numpy_seed, numpy_random)
        # Test aliases
        self._check_random_seed(numpy_seed, jit_nullary("np.random.random_sample"))
        self._check_random_seed(numpy_seed, jit_nullary("np.random.ranf"))
        self._check_random_seed(numpy_seed, jit_nullary("np.random.sample"))
        self._check_random_seed(numpy_seed, jit_nullary("np.random.rand"))

    def test_independent_generators(self):
        # PRNGs for Numpy and Python are independent.
        N = 10
        random_seed(1)
        py_numbers = [random_random() for i in range(N)]
        numpy_seed(2)
        np_numbers = [numpy_random() for i in range(N)]
        random_seed(1)
        numpy_seed(2)
        pairs = [(random_random(), numpy_random()) for i in range(N)]
        self.assertPreciseEqual([p[0] for p in pairs], py_numbers)
        self.assertPreciseEqual([p[1] for p in pairs], np_numbers)

    def _check_getrandbits(self, func, ptr):
        """
        Check a getrandbits()-like function.
        """
        # Our implementation follows CPython's for bits <= 64.
        r = self._follow_cpython(ptr)
        for nbits in range(1, 65):
            expected = r.getrandbits(nbits)
            got = func(nbits)
            self.assertPreciseEqual(expected, got)
        self.assertRaises(OverflowError, func, 65)
        self.assertRaises(OverflowError, func, 9999999)
        self.assertRaises(OverflowError, func, -1)

    def test_random_getrandbits(self):
        self._check_getrandbits(jit_unary("random.getrandbits"), py_state_ptr)

    # Explanation for the large ulps value: on 32-bit platforms, our
    # LLVM-compiled functions use SSE but they are compared against
    # C functions which use x87.
    # On some distributions, the errors seem to accumulate dramatically.

    def _check_dist(self, func, pyfunc, argslist, niters=3,
                    prec='double', ulps=12):
        assert len(argslist)
        for args in argslist:
            results = [func(*args) for i in range(niters)]
            pyresults = [pyfunc(*args) for i in range(niters)]
            self.assertPreciseEqual(results, pyresults, prec=prec, ulps=ulps,
                                    msg="for arguments %s" % (args,))

    def _check_gauss(self, func2, func1, func0, ptr):
        """
        Check a gauss()-like function.
        """
        # Our implementation follows Numpy's.
        r = self._follow_numpy(ptr)
        if func2 is not None:
            self._check_dist(func2, r.normal,
                             [(1.0, 1.0), (2.0, 0.5), (-2.0, 0.5)],
                             niters=N // 2 + 10)
        if func1 is not None:
            self._check_dist(func1, r.normal, [(0.5,)])
        if func0 is not None:
            self._check_dist(func0, r.normal, [()])

    def test_random_gauss(self):
        self._check_gauss(jit_binary("random.gauss"), None, None, py_state_ptr)

    def test_random_normalvariate(self):
        # normalvariate() is really an alias to gauss() in Numba
        # (not in Python, though - they use different algorithms)
        self._check_gauss(jit_binary("random.normalvariate"), None, None,
                          py_state_ptr)

    def test_numpy_normal(self):
        self._check_gauss(jit_binary("np.random.normal"),
                          jit_unary("np.random.normal"),
                          jit_nullary("np.random.normal"),
                          np_state_ptr)

    def test_numpy_standard_normal(self):
        self._check_gauss(None, None, jit_nullary("np.random.standard_normal"),
                          np_state_ptr)

    def test_numpy_randn(self):
        self._check_gauss(None, None, jit_nullary("np.random.randn"),
                          np_state_ptr)

    def _check_lognormvariate(self, func2, func1, func0, ptr):
        """
        Check a lognormvariate()-like function.
        """
        # Our implementation follows Numpy's.
        r = self._follow_numpy(ptr)
        if func2 is not None:
            self._check_dist(func2, r.lognormal,
                             [(1.0, 1.0), (2.0, 0.5), (-2.0, 0.5)],
                             niters=N // 2 + 10)
        if func1 is not None:
            self._check_dist(func1, r.lognormal, [(0.5,)])
        if func0 is not None:
            self._check_dist(func0, r.lognormal, [()])

    def test_random_lognormvariate(self):
        self._check_lognormvariate(jit_binary("random.lognormvariate"),
                                   None, None, py_state_ptr)

    def test_numpy_lognormal(self):
        self._check_lognormvariate(jit_binary("np.random.lognormal"),
                                   jit_unary("np.random.lognormal"),
                                   jit_nullary("np.random.lognormal"),
                                   np_state_ptr)

    def _check_randrange(self, func1, func2, func3, ptr, max_width):
        """
        Check a randrange()-like function.
        """
        # Sanity check
        ints = []
        for i in range(10):
            ints.append(func1(500000000))
            ints.append(func2(5, 500000000))
            if func3 is not None:
                ints.append(func3(5, 500000000, 3))
        self.assertEqual(len(ints), len(set(ints)), ints)
        # Our implementation follows Python 3's.
        if sys.version_info >= (3,):
            r = self._follow_cpython(ptr)
            widths = [w for w in [1, 5, 5000, 2**62 + 2**61] if w < max_width]
            for width in widths:
                self._check_dist(func1, r.randrange, [(width,)], niters=10)
                self._check_dist(func2, r.randrange, [(-2, 2 +width)], niters=10)
                if func3 is not None:
                    self.assertPreciseEqual(func3(-2, 2 + width, 6),
                                            r.randrange(-2, 2 + width, 6))
                    self.assertPreciseEqual(func3(2 + width, 2, -3),
                                            r.randrange(2 + width, 2, -3))
        # Empty ranges
        self.assertRaises(ValueError, func1, 0)
        self.assertRaises(ValueError, func1, -5)
        self.assertRaises(ValueError, func2, 5, 5)
        self.assertRaises(ValueError, func2, 5, 2)
        if func3 is not None:
            self.assertRaises(ValueError, func3, 5, 7, -1)
            self.assertRaises(ValueError, func3, 7, 5, 1)

    def test_random_randrange(self):
        for tp, max_width in [(types.int64, 2**63), (types.int32, 2**31)]:
            cr1 = compile_isolated(random_randrange1, (tp,))
            cr2 = compile_isolated(random_randrange2, (tp, tp))
            cr3 = compile_isolated(random_randrange3, (tp, tp, tp))
            self._check_randrange(cr1.entry_point, cr2.entry_point,
                                  cr3.entry_point, py_state_ptr, max_width)

    def test_numpy_randint(self):
        for tp, max_width in [(types.int64, 2**63), (types.int32, 2**31)]:
            cr1 = compile_isolated(numpy_randint1, (tp,))
            cr2 = compile_isolated(numpy_randint2, (tp, tp))
            self._check_randrange(cr1.entry_point, cr2.entry_point,
                                  None, np_state_ptr, max_width)

    def _check_randint(self, func, ptr, max_width):
        """
        Check a randint()-like function.
        """
        # Sanity check
        ints = []
        for i in range(10):
            ints.append(func(5, 500000000))
        self.assertEqual(len(ints), len(set(ints)), ints)
        # Our implementation follows Python 3's.
        if sys.version_info >= (3,):
            r = self._follow_cpython(ptr)
            for args in [(1, 5), (13, 5000), (20, 2**62 + 2**61)]:
                if args[1] > max_width:
                    continue
                self._check_dist(func, r.randint, [args], niters=10)
        # Empty ranges
        self.assertRaises(ValueError, func, 5, 4)
        self.assertRaises(ValueError, func, 5, 2)

    def test_random_randint(self):
        for tp, max_width in [(types.int64, 2**63), (types.int32, 2**31)]:
            cr = compile_isolated(random_randint, (tp, tp))
            self._check_randint(cr.entry_point, py_state_ptr, max_width)

    def _check_uniform(self, func, ptr):
        """
        Check a uniform()-like function.
        """
        # Our implementation follows Python's.
        r = self._follow_cpython(ptr)
        self._check_dist(func, r.uniform,
                         [(1.5, 1e6), (-2.5, 1e3), (1.5, -2.5)])

    def test_random_uniform(self):
        self._check_uniform(jit_binary("random.uniform"), py_state_ptr)

    def test_numpy_uniform(self):
        self._check_uniform(jit_binary("np.random.uniform"), np_state_ptr)

    def _check_triangular(self, func2, func3, ptr):
        """
        Check a triangular()-like function.
        """
        # Our implementation follows Python's.
        r = self._follow_cpython(ptr)
        if func2 is not None:
            self._check_dist(func2, r.triangular,
                             [(1.5, 3.5), (-2.5, 1.5), (1.5, 1.5)])
        self._check_dist(func3, r.triangular, [(1.5, 3.5, 2.2)])

    def test_random_triangular(self):
        self._check_triangular(jit_binary("random.triangular"),
                               jit_ternary("random.triangular"),
                               py_state_ptr)

    def test_numpy_triangular(self):
        triangular = jit_ternary("np.random.triangular")
        fixed_triangular = lambda l, r, m: triangular(l, m, r)
        self._check_triangular(None, fixed_triangular, np_state_ptr)

    def _check_gammavariate(self, func2, func1, ptr):
        """
        Check a gammavariate()-like function.
        """
        # Our implementation follows Python's.
        r = self._follow_cpython(ptr)
        if func2 is not None:
            self._check_dist(func2, r.gammavariate,
                             [(0.5, 2.5), (1.0, 1.5), (1.5, 3.5)])
        if func1 is not None:
            self.assertPreciseEqual(func1(1.5), r.gammavariate(1.5, 1.0))
        # Invalid inputs
        if func2 is not None:
            self.assertRaises(ValueError, func2, 0.0, 1.0)
            self.assertRaises(ValueError, func2, 1.0, 0.0)
            self.assertRaises(ValueError, func2, -0.5, 1.0)
            self.assertRaises(ValueError, func2, 1.0, -0.5)
        if func1 is not None:
            self.assertRaises(ValueError, func1, 0.0)
            self.assertRaises(ValueError, func1, -0.5)

    def test_random_gammavariate(self):
        self._check_gammavariate(jit_binary("random.gammavariate"), None,
                                 py_state_ptr)

    def test_numpy_gamma(self):
        self._check_gammavariate(jit_binary("np.random.gamma"),
                                 jit_unary("np.random.gamma"),
                                 np_state_ptr)
        self._check_gammavariate(None,
                                 jit_unary("np.random.standard_gamma"),
                                 np_state_ptr)

    def _check_betavariate(self, func, ptr):
        """
        Check a betavariate()-like function.
        """
        # Our implementation follows Python's.
        r = self._follow_cpython(ptr)
        self._check_dist(func, r.betavariate, [(0.5, 2.5)])
        # Invalid inputs
        self.assertRaises(ValueError, func, 0.0, 1.0)
        self.assertRaises(ValueError, func, 1.0, 0.0)
        self.assertRaises(ValueError, func, -0.5, 1.0)
        self.assertRaises(ValueError, func, 1.0, -0.5)

    def test_random_betavariate(self):
        self._check_betavariate(jit_binary("random.betavariate"), py_state_ptr)

    def test_numpy_beta(self):
        self._check_betavariate(jit_binary("np.random.beta"), np_state_ptr)

    def _check_vonmisesvariate(self, func, ptr):
        """
        Check a vonmisesvariate()-like function.
        """
        # Our implementation follows Python 2.7+'s.
        r = self._follow_cpython(ptr)
        if sys.version_info >= (2, 7):
            self._check_dist(func, r.vonmisesvariate, [(0.5, 2.5)])
        else:
            # Sanity check
            for i in range(10):
                val = func(0.5, 2.5)
                self.assertGreaterEqual(val, 0.0)
                self.assertLess(val, 2 * math.pi)

    def test_random_vonmisesvariate(self):
        self._check_vonmisesvariate(jit_binary("random.vonmisesvariate"),
                                    py_state_ptr)

    def test_numpy_vonmises(self):
        self._check_vonmisesvariate(jit_binary("np.random.vonmises"),
                                    np_state_ptr)

    def _check_expovariate(self, func, ptr):
        """
        Check a expovariate()-like function.  Note the second argument
        is inversed compared to np.random.exponential().
        """
        # Our implementation follows Numpy's (and Python 2.7+'s).
        r = self._follow_numpy(ptr)
        for lambd in (0.2, 0.5, 1.5):
            for i in range(3):
                self.assertPreciseEqual(func(lambd), r.exponential(1 / lambd),
                                        prec='double')

    def test_random_expovariate(self):
        self._check_expovariate(jit_unary("random.expovariate"), py_state_ptr)

    def _check_exponential(self, func1, func0, ptr):
        """
        Check a exponential()-like function.
        """
        # Our implementation follows Numpy's (and Python 2.7+'s).
        r = self._follow_numpy(ptr)
        if func1 is not None:
            self._check_dist(func1, r.exponential, [(0.5,), (1.0,), (1.5,)])
        if func0 is not None:
            self._check_dist(func0, r.exponential, [()])

    def test_numpy_exponential(self):
        self._check_exponential(jit_unary("np.random.exponential"),
                                jit_nullary("np.random.exponential"),
                                np_state_ptr)

    def test_numpy_standard_exponential(self):
        self._check_exponential(None,
                                jit_nullary("np.random.standard_exponential"),
                                np_state_ptr)

    def _check_paretovariate(self, func, ptr):
        """
        Check a paretovariate()-like function.
        """
        # Our implementation follows Python's.
        r = self._follow_cpython(ptr)
        self._check_dist(func, r.paretovariate, [(0.5,), (3.5,)])

    def test_random_paretovariate(self):
        self._check_paretovariate(jit_unary("random.paretovariate"), py_state_ptr)

    def test_numpy_pareto(self):
        pareto = jit_unary("np.random.pareto")
        fixed_pareto = lambda a: pareto(a) + 1.0
        self._check_paretovariate(fixed_pareto, np_state_ptr)

    def _check_weibullvariate(self, func2, func1, ptr):
        """
        Check a weibullvariate()-like function.
        """
        # Our implementation follows Python's.
        r = self._follow_cpython(ptr)
        if func2 is not None:
            self._check_dist(func2, r.weibullvariate, [(0.5, 2.5)])
        if func1 is not None:
            for i in range(3):
                self.assertPreciseEqual(func1(2.5),
                                        r.weibullvariate(1.0, 2.5))

    def test_random_weibullvariate(self):
        self._check_weibullvariate(jit_binary("random.weibullvariate"),
                                   None, py_state_ptr)

    def test_numpy_weibull(self):
        self._check_weibullvariate(None, jit_unary("np.random.weibull"),
                                   np_state_ptr)

    def test_numpy_binomial(self):
        # We follow Numpy's algorithm up to n*p == 30
        self._follow_numpy(np_state_ptr, 0)
        binomial = jit_binary("np.random.binomial")
        self.assertRaises(ValueError, binomial, -1, 0.5)
        self.assertRaises(ValueError, binomial, 10, -0.1)
        self.assertRaises(ValueError, binomial, 10, 1.1)

    def test_numpy_chisquare(self):
        chisquare = jit_unary("np.random.chisquare")
        r = self._follow_cpython(np_state_ptr)
        self._check_dist(chisquare,
                         functools.partial(py_chisquare, r),
                         [(1.5,), (2.5,)])

    def test_numpy_f(self):
        f = jit_binary("np.random.f")
        r = self._follow_cpython(np_state_ptr)
        self._check_dist(f, functools.partial(py_f, r),
                         [(0.5, 1.5), (1.5, 0.8)])

    def test_numpy_geometric(self):
        geom = jit_unary("np.random.geometric")
        # p out of domain
        self.assertRaises(ValueError, geom, -1.0)
        self.assertRaises(ValueError, geom, 0.0)
        self.assertRaises(ValueError, geom, 1.001)
        # Some basic checks
        N = 200
        r = [geom(1.0) for i in range(N)]
        self.assertPreciseEqual(r, [1] * N)
        r = [geom(0.9) for i in range(N)]
        n = r.count(1)
        self.assertGreaterEqual(n, N // 2)
        self.assertLess(n, N)
        self.assertFalse([i for i in r if i > 1000])  # unlikely
        r = [geom(0.4) for i in range(N)]
        self.assertTrue([i for i in r if i > 4])  # likely
        r = [geom(0.01) for i in range(N)]
        self.assertTrue([i for i in r if i > 50])  # likely
        r = [geom(1e-15) for i in range(N)]
        self.assertTrue([i for i in r if i > 2**32])  # likely

    def test_numpy_gumbel(self):
        gumbel = jit_binary("np.random.gumbel")
        r = self._follow_numpy(np_state_ptr)
        self._check_dist(gumbel, r.gumbel, [(0.0, 1.0), (-1.5, 3.5)])

    def test_numpy_hypergeometric(self):
        # Our implementation follows Numpy's up to nsamples = 10.
        hg = jit_ternary("np.random.hypergeometric")
        r = self._follow_numpy(np_state_ptr)
        self._check_dist(hg, r.hypergeometric,
                         [(1000, 5000, 10), (5000, 1000, 10)],
                         niters=30)
        # Sanity checks
        r = [hg(1000, 1000, 100) for i in range(100)]
        self.assertTrue(all(x >= 0 and x <= 100 for x in r), r)
        self.assertGreaterEqual(np.mean(r), 40.0)
        self.assertLessEqual(np.mean(r), 60.0)
        r = [hg(1000, 100000, 100) for i in range(100)]
        self.assertTrue(all(x >= 0 and x <= 100 for x in r), r)
        self.assertLessEqual(np.mean(r), 10.0)
        r = [hg(100000, 1000, 100) for i in range(100)]
        self.assertTrue(all(x >= 0 and x <= 100 for x in r), r)
        self.assertGreaterEqual(np.mean(r), 90.0)

    def test_numpy_laplace(self):
        r = self._follow_numpy(np_state_ptr)
        self._check_dist(jit_binary("np.random.laplace"), r.laplace,
                         [(0.0, 1.0), (-1.5, 3.5)])
        self._check_dist(jit_unary("np.random.laplace"), r.laplace,
                         [(0.0,), (-1.5,)])
        self._check_dist(jit_nullary("np.random.laplace"), r.laplace, [()])

    def test_numpy_logistic(self):
        r = self._follow_numpy(np_state_ptr)
        self._check_dist(jit_binary("np.random.logistic"), r.logistic,
                         [(0.0, 1.0), (-1.5, 3.5)])
        self._check_dist(jit_unary("np.random.logistic"), r.logistic,
                         [(0.0,), (-1.5,)])
        self._check_dist(jit_nullary("np.random.logistic"), r.logistic, [()])

    def test_numpy_logseries(self):
        r = self._follow_numpy(np_state_ptr)
        logseries = jit_unary("np.random.logseries")
        self._check_dist(logseries, r.logseries,
                         [(0.1,), (0.99,), (0.9999,)],
                         niters=50)
        # Numpy's logseries overflows on 32-bit builds, so instead
        # hardcode Numpy's (correct) output on 64-bit builds.
        r = self._follow_numpy(np_state_ptr, seed=1)
        self.assertEqual([logseries(0.9999999999999) for i in range(10)],
                         [2022733531, 77296, 30, 52204, 9341294, 703057324,
                          413147702918, 1870715907, 16009330, 738])
        self.assertRaises(ValueError, logseries, 0.0)
        self.assertRaises(ValueError, logseries, -0.1)
        self.assertRaises(ValueError, logseries, 1.1)

    def test_numpy_poisson(self):
        r = self._follow_numpy(np_state_ptr)
        poisson = jit_unary("np.random.poisson")
        # Our implementation follows Numpy's.
        self._check_dist(poisson, r.poisson,
                         [(0.0,), (0.5,), (2.0,), (10.0,), (900.5,)],
                         niters=50)
        self.assertRaises(ValueError, poisson, -0.1)

    def test_numpy_negative_binomial(self):
        self._follow_numpy(np_state_ptr, 0)
        negbin = jit_binary("np.random.negative_binomial")
        self.assertEqual([negbin(10, 0.9) for i in range(10)],
                         [2, 3, 1, 5, 2, 1, 0, 1, 0, 0])
        self.assertEqual([negbin(10, 0.1) for i in range(10)],
                         [55, 71, 56, 57, 56, 56, 34, 55, 101, 67])
        self.assertEqual([negbin(1000, 0.1) for i in range(10)],
                         [9203, 8640, 9081, 9292, 8938,
                          9165, 9149, 8774, 8886, 9117])
        m = np.mean([negbin(1000000000, 0.1)
                     for i in range(50)])
        self.assertGreater(m, 9e9 * 0.99)
        self.assertLess(m, 9e9 * 1.01)
        self.assertRaises(ValueError, negbin, 0, 0.5)
        self.assertRaises(ValueError, negbin, -1, 0.5)
        self.assertRaises(ValueError, negbin, 10, -0.1)
        self.assertRaises(ValueError, negbin, 10, 1.1)

    def test_numpy_power(self):
        r = self._follow_numpy(np_state_ptr)
        power = jit_unary("np.random.power")
        self._check_dist(power, r.power,
                         [(0.1,), (0.5,), (0.9,), (6.0,)])
        self.assertRaises(ValueError, power, 0.0)
        self.assertRaises(ValueError, power, -0.1)

    def test_numpy_rayleigh(self):
        r = self._follow_numpy(np_state_ptr)
        rayleigh1 = jit_unary("np.random.rayleigh")
        rayleigh0 = jit_nullary("np.random.rayleigh")
        self._check_dist(rayleigh1, r.rayleigh,
                         [(0.1,), (0.8,), (25.,), (1e3,)])
        self._check_dist(rayleigh0, r.rayleigh, [()])
        self.assertRaises(ValueError, rayleigh1, 0.0)
        self.assertRaises(ValueError, rayleigh1, -0.1)

    def test_numpy_standard_cauchy(self):
        r = self._follow_numpy(np_state_ptr)
        cauchy = jit_nullary("np.random.standard_cauchy")
        self._check_dist(cauchy, r.standard_cauchy, [()])

    def test_numpy_standard_t(self):
        # We use CPython's algorithm for the gamma dist and numpy's
        # for the normal dist.  Standard T calls both so we can't check
        # against either generator's output.
        r = self._follow_cpython(np_state_ptr)
        standard_t = jit_unary("np.random.standard_t")
        avg = np.mean([standard_t(5) for i in range(5000)])
        # Sanity check
        self.assertLess(abs(avg), 0.5)

    def test_numpy_wald(self):
        r = self._follow_numpy(np_state_ptr)
        wald = jit_binary("np.random.wald")
        self._check_dist(wald, r.wald, [(1.0, 1.0), (2.0, 5.0)])
        self.assertRaises(ValueError, wald, 0.0, 1.0)
        self.assertRaises(ValueError, wald, -0.1, 1.0)
        self.assertRaises(ValueError, wald, 1.0, 0.0)
        self.assertRaises(ValueError, wald, 1.0, -0.1)

    def test_numpy_zipf(self):
        r = self._follow_numpy(np_state_ptr)
        zipf = jit_unary("np.random.zipf")
        self._check_dist(zipf, r.zipf, [(1.5,), (2.5,)], niters=100)
        for val in (1.0, 0.5, 0.0, -0.1):
            self.assertRaises(ValueError, zipf, val)

    def _check_shuffle(self, func, ptr):
        """
        Check a shuffle()-like function for 1D arrays.
        """
        # Our implementation follows Python 3's.
        a = np.arange(20)
        if sys.version_info >= (3,):
            r = self._follow_cpython(ptr)
            for i in range(3):
                got = a.copy()
                expected = a.copy()
                func(got)
                r.shuffle(expected)
                self.assertTrue(np.all(got == expected), (got, expected))
        else:
            # Sanity check
            for i in range(3):
                b = a.copy()
                func(b)
                self.assertFalse(np.all(a == b))
                self.assertEqual(sorted(a), sorted(b))
                a = b
        if sys.version_info >= (2, 7):
            # Test with an arbitrary buffer-providing object
            b = a.copy()
            func(memoryview(b))
            self.assertFalse(np.all(a == b))
            self.assertEqual(sorted(a), sorted(b))
            # Read-only object
            with self.assertTypingError():
                func(memoryview(b"xyz"))

    def test_random_shuffle(self):
        self._check_shuffle(jit_unary("random.shuffle"), py_state_ptr)

    def test_numpy_shuffle(self):
        self._check_shuffle(jit_unary("np.random.shuffle"), np_state_ptr)

    def _check_startup_randomness(self, func_name, func_args):
        """
        Check that the state is properly randomized at startup.
        """
        code = """if 1:
            from numba.tests import test_random
            func = getattr(test_random, %(func_name)r)
            print(func(*%(func_args)r))
            """ % (locals())
        numbers = set()
        for i in range(3):
            popen = subprocess.Popen([sys.executable, "-c", code],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = popen.communicate()
            if popen.returncode != 0:
                raise AssertionError("process failed with code %s: stderr follows\n%s\n"
                                     % (popen.returncode, err.decode()))
            numbers.add(float(out.strip()))
        self.assertEqual(len(numbers), 3, numbers)

    def test_random_random_startup(self):
        self._check_startup_randomness("random_random", ())

    def test_random_gauss_startup(self):
        self._check_startup_randomness("random_gauss", (1.0, 1.0))

    def test_numpy_random_startup(self):
        self._check_startup_randomness("numpy_random", ())

    def test_numpy_gauss_startup(self):
        self._check_startup_randomness("numpy_normal", (1.0, 1.0))


if __name__ == "__main__":
    unittest.main()

