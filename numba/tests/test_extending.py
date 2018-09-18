from __future__ import print_function, division, absolute_import

import math
import sys
import pickle
import multiprocessing
import ctypes
from distutils.version import LooseVersion
import re

import numpy as np

from numba import unittest_support as unittest
from numba import jit, types, errors, typing
from numba.compiler import compile_isolated
from .support import (TestCase, captured_stdout, tag, temp_directory,
                      override_config)

from numba.extending import (typeof_impl, type_callable,
                             lower_builtin, lower_cast,
                             overload, overload_attribute,
                             overload_method,
                             models, register_model,
                             box, unbox, NativeValue,
                             make_attribute_wrapper,
                             intrinsic, _Intrinsic,
                             register_jitable,
                             get_cython_function_address
                             )
from numba.typing.templates import (
    ConcreteTemplate, signature, infer)

# Pandas-like API implementation
from .pdlike_usecase import Index, Series

try:
    import scipy
    if LooseVersion(scipy.__version__) < '0.19':
        sc = None
    else:
        import scipy.special.cython_special as sc
except ImportError:
    sc = None


# -----------------------------------------------------------------------
# Define a custom type and an implicit cast on it

class MyDummy(object):
    pass

class MyDummyType(types.Opaque):

    def can_convert_to(self, context, toty):
        if isinstance(toty, types.Number):
            from numba.typeconv import Conversion
            return Conversion.safe

mydummy_type = MyDummyType('mydummy')
mydummy = MyDummy()

@typeof_impl.register(MyDummy)
def typeof_mydummy(val, c):
    return mydummy_type

@lower_cast(MyDummyType, types.Number)
def mydummy_to_number(context, builder, fromty, toty, val):
    """
    Implicit conversion from MyDummy to int.
    """
    return context.get_constant(toty, 42)

def get_dummy():
    return mydummy

register_model(MyDummyType)(models.OpaqueModel)

@unbox(MyDummyType)
def unbox_index(typ, obj, c):
    return NativeValue(c.context.get_dummy_value())


# -----------------------------------------------------------------------
# Define a function's typing and implementation using the classical
# two-step API

def func1(x=None):
    raise NotImplementedError

@type_callable(func1)
def type_func1(context):
    def typer(x=None):
        if x in (None, types.none):
            # 0-arg or 1-arg with None
            return types.int32
        elif isinstance(x, types.Float):
            # 1-arg with float
            return x

    return typer

@lower_builtin(func1)
@lower_builtin(func1, types.none)
def func1_nullary(context, builder, sig, args):
    return context.get_constant(sig.return_type, 42)

@lower_builtin(func1, types.Float)
def func1_unary(context, builder, sig, args):
    def func1_impl(x):
        return math.sqrt(2 * x)
    return context.compile_internal(builder, func1_impl, sig, args)

# We can do the same for a known internal operation, here "print_item"
# which we extend to support MyDummyType.

@infer
class PrintDummy(ConcreteTemplate):
    key = "print_item"
    cases = [signature(types.none, mydummy_type)]

@lower_builtin("print_item", MyDummyType)
def print_dummy(context, builder, sig, args):
    [x] = args
    pyapi = context.get_python_api(builder)
    strobj = pyapi.unserialize(pyapi.serialize_object("hello!"))
    pyapi.print_object(strobj)
    pyapi.decref(strobj)
    return context.get_dummy_value()


# -----------------------------------------------------------------------
# Define an overloaded function (combined API)

def where(cond, x, y):
    raise NotImplementedError

def np_where(cond, x, y):
    """
    Wrap np.where() to allow for keyword arguments
    """
    return np.where(cond, x, y)

def call_where(cond, x, y):
    return where(cond, y=y, x=x)


@overload(where)
def overload_where_arrays(cond, x, y):
    """
    Implement where() for arrays.
    """
    # Choose implementation based on argument types.
    if isinstance(cond, types.Array):
        if x.dtype != y.dtype:
            raise errors.TypingError("x and y should have the same dtype")

        # Array where() => return an array of the same shape
        if all(ty.layout == 'C' for ty in (cond, x, y)):
            def where_impl(cond, x, y):
                """
                Fast implementation for C-contiguous arrays
                """
                shape = cond.shape
                if x.shape != shape or y.shape != shape:
                    raise ValueError("all inputs should have the same shape")
                res = np.empty_like(x)
                cf = cond.flat
                xf = x.flat
                yf = y.flat
                rf = res.flat
                for i in range(cond.size):
                    rf[i] = xf[i] if cf[i] else yf[i]
                return res
        else:
            def where_impl(cond, x, y):
                """
                Generic implementation for other arrays
                """
                shape = cond.shape
                if x.shape != shape or y.shape != shape:
                    raise ValueError("all inputs should have the same shape")
                res = np.empty_like(x)
                for idx, c in np.ndenumerate(cond):
                    res[idx] = x[idx] if c else y[idx]
                return res

        return where_impl

# We can define another overload function for the same function, they
# will be tried in turn until one succeeds.

@overload(where)
def overload_where_scalars(cond, x, y):
    """
    Implement where() for scalars.
    """
    if not isinstance(cond, types.Array):
        if x != y:
            raise errors.TypingError("x and y should have the same type")

        def where_impl(cond, x, y):
            """
            Scalar where() => return a 0-dim array
            """
            scal = x if cond else y
            # Can't use full_like() on Numpy < 1.8
            arr = np.empty_like(scal)
            arr[()] = scal
            return arr

        return where_impl

# -----------------------------------------------------------------------
# Overload an already defined built-in function, extending it for new types.

@overload(len)
def overload_len_dummy(arg):
    if isinstance(arg, MyDummyType):
        def len_impl(arg):
            return 13

        return len_impl


@overload_method(MyDummyType, 'length')
def overload_method_length(arg):
    def imp(arg):
        return len(arg)
    return imp


def cache_overload_method_usecase(x):
    return x.length()


def call_func1_nullary():
    return func1()

def call_func1_unary(x):
    return func1(x)

def len_usecase(x):
    return len(x)

def print_usecase(x):
    print(x)

def getitem_usecase(x, key):
    return x[key]

def npyufunc_usecase(x):
    return np.cos(np.sin(x))

def get_data_usecase(x):
    return x._data

def get_index_usecase(x):
    return x._index

def is_monotonic_usecase(x):
    return x.is_monotonic_increasing

def make_series_usecase(data, index):
    return Series(data, index)

def clip_usecase(x, lo, hi):
    return x.clip(lo, hi)


# -----------------------------------------------------------------------

def return_non_boxable():
    return np


@overload(return_non_boxable)
def overload_return_non_boxable():
    def imp():
        return np
    return imp


def non_boxable_ok_usecase(sz):
    mod = return_non_boxable()
    return mod.arange(sz)


def non_boxable_bad_usecase():
    return return_non_boxable()


class TestLowLevelExtending(TestCase):
    """
    Test the low-level two-tier extension API.
    """

    # We check with both @jit and compile_isolated(), to exercise the
    # registration logic.

    def test_func1(self):
        pyfunc = call_func1_nullary
        cfunc = jit(nopython=True)(pyfunc)
        self.assertPreciseEqual(cfunc(), 42)
        pyfunc = call_func1_unary
        cfunc = jit(nopython=True)(pyfunc)
        self.assertPreciseEqual(cfunc(None), 42)
        self.assertPreciseEqual(cfunc(18.0), 6.0)

    def test_func1_isolated(self):
        pyfunc = call_func1_nullary
        cr = compile_isolated(pyfunc, ())
        self.assertPreciseEqual(cr.entry_point(), 42)
        pyfunc = call_func1_unary
        cr = compile_isolated(pyfunc, (types.float64,))
        self.assertPreciseEqual(cr.entry_point(18.0), 6.0)

    def test_cast_mydummy(self):
        pyfunc = get_dummy
        cr = compile_isolated(pyfunc, (), types.float64)
        self.assertPreciseEqual(cr.entry_point(), 42.0)


class TestPandasLike(TestCase):
    """
    Test implementing a pandas-like Index object.
    Also stresses most of the high-level API.
    """

    def test_index_len(self):
        i = Index(np.arange(3))
        cfunc = jit(nopython=True)(len_usecase)
        self.assertPreciseEqual(cfunc(i), 3)

    def test_index_getitem(self):
        i = Index(np.int32([42, 8, -5]))
        cfunc = jit(nopython=True)(getitem_usecase)
        self.assertPreciseEqual(cfunc(i, 1), 8)
        ii = cfunc(i, slice(1, None))
        self.assertIsInstance(ii, Index)
        self.assertEqual(list(ii), [8, -5])

    def test_index_ufunc(self):
        """
        Check Numpy ufunc on an Index object.
        """
        i = Index(np.int32([42, 8, -5]))
        cfunc = jit(nopython=True)(npyufunc_usecase)
        ii = cfunc(i)
        self.assertIsInstance(ii, Index)
        self.assertPreciseEqual(ii._data, np.cos(np.sin(i._data)))

    def test_index_get_data(self):
        # The _data attribute is exposed with make_attribute_wrapper()
        i = Index(np.int32([42, 8, -5]))
        cfunc = jit(nopython=True)(get_data_usecase)
        data = cfunc(i)
        self.assertIs(data, i._data)

    def test_index_is_monotonic(self):
        # The is_monotonic_increasing attribute is exposed with
        # overload_attribute()
        cfunc = jit(nopython=True)(is_monotonic_usecase)
        for values, expected in [([8, 42, 5], False),
                                 ([5, 8, 42], True),
                                 ([], True)]:
            i = Index(np.int32(values))
            got = cfunc(i)
            self.assertEqual(got, expected)

    @tag('important')
    def test_series_len(self):
        i = Index(np.int32([2, 4, 3]))
        s = Series(np.float64([1.5, 4.0, 2.5]), i)
        cfunc = jit(nopython=True)(len_usecase)
        self.assertPreciseEqual(cfunc(s), 3)

    @tag('important')
    def test_series_get_index(self):
        i = Index(np.int32([2, 4, 3]))
        s = Series(np.float64([1.5, 4.0, 2.5]), i)
        cfunc = jit(nopython=True)(get_index_usecase)
        got = cfunc(s)
        self.assertIsInstance(got, Index)
        self.assertIs(got._data, i._data)

    def test_series_ufunc(self):
        """
        Check Numpy ufunc on an Series object.
        """
        i = Index(np.int32([42, 8, -5]))
        s = Series(np.int64([1, 2, 3]), i)
        cfunc = jit(nopython=True)(npyufunc_usecase)
        ss = cfunc(s)
        self.assertIsInstance(ss, Series)
        self.assertIsInstance(ss._index, Index)
        self.assertIs(ss._index._data, i._data)
        self.assertPreciseEqual(ss._values, np.cos(np.sin(s._values)))

    @tag('important')
    def test_series_constructor(self):
        i = Index(np.int32([42, 8, -5]))
        d = np.float64([1.5, 4.0, 2.5])
        cfunc = jit(nopython=True)(make_series_usecase)
        got = cfunc(d, i)
        self.assertIsInstance(got, Series)
        self.assertIsInstance(got._index, Index)
        self.assertIs(got._index._data, i._data)
        self.assertIs(got._values, d)

    @tag('important')
    def test_series_clip(self):
        i = Index(np.int32([42, 8, -5]))
        s = Series(np.float64([1.5, 4.0, 2.5]), i)
        cfunc = jit(nopython=True)(clip_usecase)
        ss = cfunc(s, 1.6, 3.0)
        self.assertIsInstance(ss, Series)
        self.assertIsInstance(ss._index, Index)
        self.assertIs(ss._index._data, i._data)
        self.assertPreciseEqual(ss._values, np.float64([1.6, 3.0, 2.5]))


class TestHighLevelExtending(TestCase):
    """
    Test the high-level combined API.
    """

    @tag('important')
    def test_where(self):
        """
        Test implementing a function with @overload.
        """
        pyfunc = call_where
        cfunc = jit(nopython=True)(pyfunc)

        def check(*args, **kwargs):
            expected = np_where(*args, **kwargs)
            got = cfunc(*args, **kwargs)
            self.assertPreciseEqual

        check(x=3, cond=True, y=8)
        check(True, 3, 8)
        check(np.bool_([True, False, True]), np.int32([1, 2, 3]),
              np.int32([4, 5, 5]))

        # The typing error is propagated
        with self.assertRaises(errors.TypingError) as raises:
            cfunc(np.bool_([]), np.int32([]), np.int64([]))
        self.assertIn("x and y should have the same dtype",
                      str(raises.exception))

    @tag('important')
    def test_len(self):
        """
        Test re-implementing len() for a custom type with @overload.
        """
        cfunc = jit(nopython=True)(len_usecase)
        self.assertPreciseEqual(cfunc(MyDummy()), 13)
        self.assertPreciseEqual(cfunc([4, 5]), 2)

    def test_print(self):
        """
        Test re-implementing print() for a custom type with @overload.
        """
        cfunc = jit(nopython=True)(print_usecase)
        with captured_stdout():
            cfunc(MyDummy())
            self.assertEqual(sys.stdout.getvalue(), "hello!\n")

    def test_no_cpython_wrapper(self):
        """
        Test overloading whose return value cannot be represented in CPython.
        """
        # Test passing Module type from a @overload implementation to ensure
        # that the *no_cpython_wrapper* flag works
        ok_cfunc = jit(nopython=True)(non_boxable_ok_usecase)
        n = 10
        got = ok_cfunc(n)
        expect = non_boxable_ok_usecase(n)
        np.testing.assert_equal(expect, got)
        # Verify that the Module type cannot be returned to CPython
        bad_cfunc = jit(nopython=True)(non_boxable_bad_usecase)
        with self.assertRaises(TypeError) as raises:
            bad_cfunc()
        errmsg = str(raises.exception)
        expectmsg = "cannot convert native Module"
        self.assertIn(expectmsg, errmsg)


def _assert_cache_stats(cfunc, expect_hit, expect_misses):
    hit = cfunc._cache_hits[cfunc.signatures[0]]
    if hit != expect_hit:
        raise AssertionError('cache not used')
    miss = cfunc._cache_misses[cfunc.signatures[0]]
    if miss != expect_misses:
        raise AssertionError('cache not used')


class TestOverloadMethodCaching(TestCase):
    # Nested multiprocessing.Pool raises AssertionError:
    # "daemonic processes are not allowed to have children"
    _numba_parallel_test_ = False

    def test_caching_overload_method(self):
        self._cache_dir = temp_directory(self.__class__.__name__)
        with override_config('CACHE_DIR', self._cache_dir):
            self.run_caching_overload_method()

    def run_caching_overload_method(self):
        cfunc = jit(nopython=True, cache=True)(cache_overload_method_usecase)
        self.assertPreciseEqual(cfunc(MyDummy()), 13)
        _assert_cache_stats(cfunc, 0, 1)
        llvmir = cfunc.inspect_llvm((mydummy_type,))
        # Ensure the inner method is not a declaration
        decls = [ln for ln in llvmir.splitlines()
                 if ln.startswith('declare') and 'overload_method_length' in ln]
        self.assertEqual(len(decls), 0)
        # Test in a separate process
        try:
            ctx = multiprocessing.get_context('spawn')
        except AttributeError:
            ctx = multiprocessing
        q = ctx.Queue()
        p = ctx.Process(target=run_caching_overload_method,
                        args=(q, self._cache_dir))
        p.start()
        q.put(MyDummy())
        p.join()
        # Ensure subprocess exited normally
        self.assertEqual(p.exitcode, 0)
        res = q.get(timeout=1)
        self.assertEqual(res, 13)


def run_caching_overload_method(q, cache_dir):
    """
    Used by TestOverloadMethodCaching.test_caching_overload_method
    """
    with override_config('CACHE_DIR', cache_dir):
        arg = q.get()
        cfunc = jit(nopython=True, cache=True)(cache_overload_method_usecase)
        res = cfunc(arg)
        q.put(res)
        # Check cache stat
        _assert_cache_stats(cfunc, 1, 0)

class TestIntrinsic(TestCase):
    def test_ll_pointer_cast(self):
        """
        Usecase test: custom reinterpret cast to turn int values to pointers
        """
        from ctypes import CFUNCTYPE, POINTER, c_float, c_int

        # Use intrinsic to make a reinterpret_cast operation
        def unsafe_caster(result_type):
            assert isinstance(result_type, types.CPointer)

            @intrinsic
            def unsafe_cast(typingctx, src):
                self.assertIsInstance(typingctx, typing.Context)
                if isinstance(src, types.Integer):
                    sig = result_type(types.uintp)

                    # defines the custom code generation
                    def codegen(context, builder, signature, args):
                        [src] = args
                        rtype = signature.return_type
                        llrtype = context.get_value_type(rtype)
                        return builder.inttoptr(src, llrtype)

                    return sig, codegen

            return unsafe_cast

        # make a nopython function to use our cast op.
        # this is not usable from cpython due to the returning of a pointer.
        def unsafe_get_ctypes_pointer(src):
            raise NotImplementedError("not callable from python")

        @overload(unsafe_get_ctypes_pointer)
        def array_impl_unsafe_get_ctypes_pointer(arrtype):
            if isinstance(arrtype, types.Array):
                unsafe_cast = unsafe_caster(types.CPointer(arrtype.dtype))

                def array_impl(arr):
                    return unsafe_cast(src=arr.ctypes.data)
                return array_impl

        # the ctype wrapped function for use in nopython mode
        def my_c_fun_raw(ptr, n):
            for i in range(n):
                print(ptr[i])

        prototype = CFUNCTYPE(None, POINTER(c_float), c_int)
        my_c_fun = prototype(my_c_fun_raw)

        # Call our pointer-cast in a @jit compiled function and use
        # the pointer in a ctypes function
        @jit(nopython=True)
        def foo(arr):
            ptr = unsafe_get_ctypes_pointer(arr)
            my_c_fun(ptr, arr.size)

        # Test
        arr = np.arange(10, dtype=np.float32)
        with captured_stdout() as buf:
            foo(arr)
            got = buf.getvalue().splitlines()
        buf.close()
        expect = list(map(str, arr))
        self.assertEqual(expect, got)

    def test_serialization(self):
        """
        Test serialization of intrinsic objects
        """
        # define a intrinsic
        @intrinsic
        def identity(context, x):
            def codegen(context, builder, signature, args):
                return args[0]

            sig = x(x)
            return sig, codegen

        # use in a jit function
        @jit(nopython=True)
        def foo(x):
            return identity(x)

        self.assertEqual(foo(1), 1)

        # get serialization memo
        memo = _Intrinsic._memo
        memo_size = len(memo)

        # pickle foo and check memo size
        serialized_foo = pickle.dumps(foo)
        # increases the memo size
        memo_size += 1
        self.assertEqual(memo_size, len(memo))
        # unpickle
        foo_rebuilt = pickle.loads(serialized_foo)
        self.assertEqual(memo_size, len(memo))
        # check rebuilt foo
        self.assertEqual(foo(1), foo_rebuilt(1))

        # pickle identity directly
        serialized_identity = pickle.dumps(identity)
        # memo size unchanged
        self.assertEqual(memo_size, len(memo))
        # unpickle
        identity_rebuilt = pickle.loads(serialized_identity)
        # must be the same object
        self.assertIs(identity, identity_rebuilt)
        # memo size unchanged
        self.assertEqual(memo_size, len(memo))

    def test_deserialization(self):
        """
        Test deserialization of intrinsic
        """
        def defn(context, x):
            def codegen(context, builder, signature, args):
                return args[0]

            return x(x), codegen

        memo = _Intrinsic._memo
        memo_size = len(memo)
        # invoke _Intrinsic indirectly to avoid registration which keeps an
        # internal reference inside the compiler
        original = _Intrinsic('foo', defn)
        self.assertIs(original._defn, defn)
        pickled = pickle.dumps(original)
        # by pickling, a new memo entry is created
        memo_size += 1
        self.assertEqual(memo_size, len(memo))
        del original  # remove original before unpickling

        # by deleting, the memo entry is NOT removed due to recent
        # function queue
        self.assertEqual(memo_size, len(memo))

        # Manually force clear of _recent queue
        _Intrinsic._recent.clear()
        memo_size -= 1
        self.assertEqual(memo_size, len(memo))

        rebuilt = pickle.loads(pickled)
        # verify that the rebuilt object is different
        self.assertIsNot(rebuilt._defn, defn)

        # the second rebuilt object is the same as the first
        second = pickle.loads(pickled)
        self.assertIs(rebuilt._defn, second._defn)


class TestRegisterJitable(unittest.TestCase):
    def test_no_flags(self):
        @register_jitable
        def foo(x, y):
            return x + y

        def bar(x, y):
            return foo(x, y)

        cbar = jit(nopython=True)(bar)

        expect = bar(1, 2)
        got = cbar(1, 2)
        self.assertEqual(expect, got)

    def test_flags_no_nrt(self):
        @register_jitable(_nrt=False)
        def foo(n):
            return np.arange(n)

        def bar(n):
            return foo(n)

        self.assertEqual(bar(3).tolist(), [0, 1, 2])

        cbar = jit(nopython=True)(bar)
        with self.assertRaises(errors.TypingError) as raises:
            cbar(2)
        msg = "Only accept returning of array passed into the function as argument"
        self.assertIn(msg, str(raises.exception))


class TestImportCythonFunction(unittest.TestCase):
    @unittest.skipIf(sc is None, "Only run if SciPy >= 0.19 is installed")
    def test_getting_function(self):
        addr = get_cython_function_address("scipy.special.cython_special", "j0")
        functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
        _j0 = functype(addr)
        j0 = jit(nopython=True)(lambda x: _j0(x))
        self.assertEqual(j0(0), 1)

    def test_missing_module(self):
        with self.assertRaises(ImportError) as raises:
            addr = get_cython_function_address("fakemodule", "fakefunction")
        # The quotes are not there in Python 2
        msg = "No module named '?fakemodule'?"
        match = re.match(msg, str(raises.exception))
        self.assertIsNotNone(match)

    @unittest.skipIf(sc is None, "Only run if SciPy >= 0.19 is installed")
    def test_missing_function(self):
        with self.assertRaises(ValueError) as raises:
            addr = get_cython_function_address("scipy.special.cython_special", "foo")
        msg = "No function 'foo' found in __pyx_capi__ of 'scipy.special.cython_special'"
        self.assertEqual(msg, str(raises.exception))


if __name__ == '__main__':
    unittest.main()
