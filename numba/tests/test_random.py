from __future__ import print_function

import random
import subprocess
import sys

import numpy as np

import numba.unittest_support as unittest
from numba import jit, _helperlib, types
from numba.compiler import compile_isolated
from numba.pythonapi import NativeError
from .support import TestCase


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
    s = """def func(%(argstring)s):
        return %(name)s(%(argstring)s)
    """ % locals()
    co = compile(s, "<string>", "exec")
    ns = {}
    eval(co, globals(), ns)
    return jit(nopython=True)(ns['func'])

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
            r.seed(i)
            st = r.get_state()
            ints = list(st[1])
            index = st[2]
            assert index == N  # sanity check
            _helperlib.rnd_init(ptr, i)
            self.assertEqual(_helperlib.rnd_get_state(ptr), (index, ints))

    def test_get_set_state(self):
        self._check_get_set_state(py_state_ptr)

    def test_shuffle(self):
        self._check_shuffle(py_state_ptr)

    def test_init(self):
        self._check_init(py_state_ptr)


class TestRandom(TestCase):

    def _follow_cpython(self, ptr):
        r = random.Random()
        _copy_py_state(r, ptr)
        return r

    def _follow_numpy(self, ptr):
        r = np.random.RandomState()
        _copy_np_state(r, ptr)
        return r

    def _check_random_seed(self, seedfunc, randomfunc):
        """
        Check seed()- and random()-like functions.
        """
        # Our seed() mimicks Numpy's.
        r = np.random.RandomState()
        for i in [0, 1, 125, 2**32 - 1]:
            r.seed(i)
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

    def test_random_getrandbits(self):
        self._check_getrandbits(jit_unary("random.getrandbits"), py_state_ptr)

    def _check_gauss(self, func2, func1, func0, ptr):
        """
        Check a gauss()-like function.
        """
        # Our implementation follows Numpy's.
        r = self._follow_numpy(ptr)
        if func2 is not None:
            for mu, sigma in [(1.0, 1.0), (2.0, 0.5), (-2.0, 0.5)]:
                for i in range(N // 2 + 10):
                    self.assertPreciseEqual(func2(mu, sigma), r.normal(mu, sigma))
        if func1 is not None:
            for i in range(3):
                self.assertPreciseEqual(func1(0.5), r.normal(0.5))
        if func0 is not None:
            for i in range(3):
                self.assertPreciseEqual(func0(), r.normal())

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

    def _check_lognormvariate(self, func2, func1, func0, ptr):
        """
        Check a lognormvariate()-like function.
        """
        # Our implementation follows Numpy's.
        r = self._follow_numpy(ptr)
        if func2 is not None:
            for mu, sigma in [(1.0, 1.0), (2.0, 0.5), (-2.0, 0.5)]:
                for i in range(N // 2 + 10):
                    self.assertPreciseEqual(func2(mu, sigma),
                                            r.lognormal(mu, sigma))
        if func1 is not None:
            for i in range(3):
                self.assertPreciseEqual(func1(0.5), r.lognormal(0.5))
        if func0 is not None:
            for i in range(3):
                self.assertPreciseEqual(func0(), r.lognormal())

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
            for width in [1, 5, 5000, 2**62 + 2**61]:
                if width > max_width:
                    continue
                for i in range(10):
                    self.assertPreciseEqual(func1(width), r.randrange(width))
                self.assertPreciseEqual(func2(-2, 2 + width),
                                        r.randrange(-2, 2 + width))
                if func3 is not None:
                    self.assertPreciseEqual(func3(-2, 2 + width, 6),
                                            r.randrange(-2, 2 + width, 6))
                    self.assertPreciseEqual(func3(2 + width, 2, -3),
                                            r.randrange(2 + width, 2, -3))
        # Empty ranges
        self.assertRaises(NativeError, func1, 0)
        self.assertRaises(NativeError, func1, -5)
        self.assertRaises(NativeError, func2, 5, 5)
        self.assertRaises(NativeError, func2, 5, 2)
        if func3 is not None:
            self.assertRaises(NativeError, func3, 5, 7, -1)
            self.assertRaises(NativeError, func3, 7, 5, 1)

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
                for i in range(10):
                    self.assertPreciseEqual(func(*args), r.randint(*args))
        # Empty ranges
        self.assertRaises(NativeError, func, 5, 4)
        self.assertRaises(NativeError, func, 5, 2)

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
        for args in [(1.5, 1e6), (-2.5, 1e3), (1.5, -2.5)]:
            self.assertPreciseEqual(func(*args), r.uniform(*args))

    def test_random_uniform(self):
        self._check_uniform(jit_binary("random.uniform"), py_state_ptr)

    def _check_triangular(self, func2, func3, ptr):
        """
        Check a triangular()-like function.
        """
        # Our implementation follows Python's.
        r = self._follow_cpython(ptr)
        for args in [(1.5, 3.5), (-2.5, 1.5), (1.5, 1.5)]:
            self.assertPreciseEqual(func2(*args), r.triangular(*args))

    def test_random_triangular(self):
        self._check_triangular(jit_binary("random.triangular"),
                               jit_ternary("random.triangular"),
                               py_state_ptr)

    def _check_gammavariate(self, func2, func1, ptr):
        """
        Check a gammavariate()-like function.
        """
        # Our implementation follows Python's.
        r = self._follow_cpython(ptr)
        for args in [(0.5, 2.5), (1.0, 1.5), (1.5, 3.5)]:
            if func2 is not None:
                for i in range(3):
                    self.assertPreciseEqual(func2(*args), r.gammavariate(*args))
            if func1 is not None:
                self.assertPreciseEqual(func1(1.5), r.gammavariate(1.5, 1.0))
        # Invalid inputs
        if func2 is not None:
            self.assertRaises(NativeError, func2, 0.0, 1.0)
            self.assertRaises(NativeError, func2, 1.0, 0.0)
            self.assertRaises(NativeError, func2, -0.5, 1.0)
            self.assertRaises(NativeError, func2, 1.0, -0.5)
        if func1 is not None:
            self.assertRaises(NativeError, func1, 0.0)
            self.assertRaises(NativeError, func1, -0.5)

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
        args = (0.5, 2.5)
        for i in range(3):
            self.assertPreciseEqual(func(*args), r.betavariate(*args))
        # Invalid inputs
        self.assertRaises(NativeError, func, 0.0, 1.0)
        self.assertRaises(NativeError, func, 1.0, 0.0)
        self.assertRaises(NativeError, func, -0.5, 1.0)
        self.assertRaises(NativeError, func, 1.0, -0.5)

    def test_random_betavariate(self):
        self._check_betavariate(jit_binary("random.betavariate"), py_state_ptr)

    def test_numpy_beta(self):
        self._check_betavariate(jit_binary("np.random.beta"), np_state_ptr)

    def _check_vonmisesvariate(self, func, ptr):
        """
        Check a vonmisesvariate()-like function.
        """
        # Our implementation follows Python's.
        r = self._follow_cpython(ptr)
        args = (0.5, 2.5)
        for i in range(3):
            self.assertPreciseEqual(func(*args), r.vonmisesvariate(*args))

    def test_random_vonmisesvariate(self):
        self._check_vonmisesvariate(jit_binary("random.vonmisesvariate"), py_state_ptr)

    def _check_unary(self, func, pyfunc, argslist):
        for args in argslist:
            for i in range(3):
                self.assertPreciseEqual(func(*args), pyfunc(*args))

    def _check_expovariate(self, func, ptr):
        """
        Check a expovariate()-like function.
        """
        # Our implementation follows Python's.
        r = self._follow_cpython(ptr)
        self._check_unary(func, r.expovariate, [(-0.5,), (0.5,)])

    def test_random_expovariate(self):
        self._check_expovariate(jit_unary("random.expovariate"), py_state_ptr)

    def _check_exponential(self, func1, func0, ptr):
        """
        Check a exponential()-like function. Note the second argument
        is inversed compared to expovariate().
        """
        r = self._follow_cpython(ptr)
        if func1 is not None:
            for scale in (0.5, 1.0, 1.5):
                for i in range(3):
                    self.assertPreciseEqual(func1(scale),
                                            r.expovariate(1 / scale),
                                            prec="double")
        if func0 is not None:
            self.assertPreciseEqual(func0(), r.expovariate(1.0))

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
        self._check_unary(func, r.paretovariate, [(0.5,), (3.5,)])

    def test_random_paretovariate(self):
        self._check_paretovariate(jit_unary("random.paretovariate"), py_state_ptr)

    def test_numpy_pareto(self):
        pareto = jit_unary("np.random.pareto")
        fixed_pareto = lambda a: pareto(a) + 1.0
        self._check_paretovariate(fixed_pareto, np_state_ptr)

    def _check_weibullvariate(self, func, ptr):
        """
        Check a weibullvariate()-like function.
        """
        # Our implementation follows Python's.
        r = self._follow_cpython(ptr)
        args = (0.5, 2.5)
        for i in range(3):
            self.assertPreciseEqual(func(*args), r.weibullvariate(*args))

    def test_random_weibullvariate(self):
        self._check_weibullvariate(jit_binary("random.weibullvariate"), py_state_ptr)

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

    def test_random_shuffle(self):
        self._check_shuffle(jit_unary("random.shuffle"), py_state_ptr)

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

