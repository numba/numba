"""
Test Numpy Subclassing features
"""

import builtins
import unittest
from numbers import Number

import numpy as np
from llvmlite import ir

import numba
from numba import njit, typeof, objmode
from numba.core import cgutils, types, typing
from numba.core.pythonapi import box
from numba.core.errors import TypingError
from numba.core.registry import cpu_target
from numba.extending import (intrinsic, lower_builtin, overload_classmethod,
                             register_model, type_callable, typeof_impl,
                             register_jitable)
from numba.np import numpy_support

from numba.tests.support import TestCase

# A quick util to allow logging within jit code

_logger = None


def _do_log(*args):
    if _logger is not None:
        _logger.append(args)


@register_jitable
def log(*args):
    with objmode():
        _do_log(*args)


class myarray(np.ndarray):
    # Tell Numba to not seamlessly treat this type as a regular ndarray.
    __numba_array_subtype_dispatch__ = True

    # __array__ is not needed given that this is a ndarray subclass
    #
    # # Interoperate with Numpy outside of Numba.
    # def __array__(self, dtype=None):
    #     return self

    # Interoperate with Numpy outside of Numba.
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            N = None
            scalars = []
            for inp in inputs:
                if isinstance(inp, Number):
                    scalars.append(inp)
                elif isinstance(inp, (self.__class__, np.ndarray)):
                    if isinstance(inp, self.__class__):
                        scalars.append(np.ndarray(inp.shape, inp.dtype, inp))
                    else:
                        scalars.append(inp)
                    if N is not None:
                        if N != inp.shape:
                            raise TypeError("inconsistent sizes")
                    else:
                        N = inp.shape
                else:
                    return NotImplemented
            ret = ufunc(*scalars, **kwargs)
            return self.__class__(ret.shape, ret.dtype, ret)
        else:
            return NotImplemented


class MyArrayType(types.Array):
    def __init__(self, dtype, ndim, layout, readonly=False, aligned=True):
        name = f"MyArray({ndim}, {dtype}, {layout})"
        super().__init__(dtype, ndim, layout, readonly=False, aligned=aligned,
                         name=name)

    # Tell Numba typing how to combine MyArrayType with other ndarray types.
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            for inp in inputs:
                if not isinstance(inp, (types.Array, types.Number)):
                    return NotImplemented
            # Ban if all arguments are MyArrayType
            if all(isinstance(inp, MyArrayType) for inp in inputs):
                return NotImplemented
            return MyArrayType
        else:
            return NotImplemented

    @property
    def box_type(self):
        return myarray


@typeof_impl.register(myarray)
def typeof_ta_ndarray(val, c):
    try:
        dtype = numpy_support.from_dtype(val.dtype)
    except NotImplementedError:
        raise ValueError("Unsupported array dtype: %s" % (val.dtype,))
    layout = numpy_support.map_layout(val)
    readonly = not val.flags.writeable
    return MyArrayType(dtype, val.ndim, layout, readonly=readonly)


register_model(MyArrayType)(numba.core.datamodel.models.ArrayModel)


@type_callable(myarray)
def type_myarray(context):
    def typer(shape, dtype, buf):
        out = MyArrayType(
            dtype=buf.dtype, ndim=len(shape), layout=buf.layout
        )
        return out

    return typer


@lower_builtin(myarray, types.UniTuple, types.DType, types.Array)
def impl_myarray(context, builder, sig, args):
    from numba.np.arrayobj import make_array, populate_array

    srcaryty = sig.args[-1]
    shape, dtype, buf = args

    srcary = make_array(srcaryty)(context, builder, value=buf)
    # Copy source array and remove the parent field to avoid boxer re-using
    # the original ndarray instance.
    retary = make_array(sig.return_type)(context, builder)
    populate_array(retary,
                   data=srcary.data,
                   shape=srcary.shape,
                   strides=srcary.strides,
                   itemsize=srcary.itemsize,
                   meminfo=srcary.meminfo)

    ret = retary._getvalue()
    context.nrt.incref(builder, sig.return_type, ret)
    return ret


@box(MyArrayType)
def box_array(typ, val, c):
    assert c.context.enable_nrt
    np_dtype = numpy_support.as_dtype(typ.dtype)
    dtypeptr = c.env_manager.read_const(c.env_manager.add_const(np_dtype))
    # Steals NRT ref
    newary = c.pyapi.nrt_adapt_ndarray_to_python(typ, val, dtypeptr)
    return newary


@overload_classmethod(MyArrayType, "_allocate")
def _ol_array_allocate(cls, allocsize, align):
    """Implements a Numba-only classmethod on the array type.
    """
    def impl(cls, allocsize, align):
        log("LOG _ol_array_allocate", allocsize, align)
        return allocator_MyArray(allocsize, align)

    return impl


@intrinsic
def allocator_MyArray(typingctx, allocsize, align):
    def impl(context, builder, sig, args):
        # Implementation is just Numba's allocator at the moment for
        # illustration.
        # Haven't done the integration to test our real new allocator yet.
        context.nrt._require_nrt()
        size, align = args

        mod = builder.module
        u32 = ir.IntType(32)
        fnty = ir.FunctionType(cgutils.voidptr_t, [cgutils.intp_t, u32])
        fn = cgutils.get_or_insert_function(
            mod, fnty, name="NRT_MemInfo_alloc_safe_aligned"
        )
        fn.return_value.add_attribute("noalias")
        if isinstance(align, builtins.int):
            align = context.get_constant(types.uint32, align)
        else:
            assert align.type == u32, "align must be a uint32"
        call = builder.call(fn, [size, align])
        call.name = "allocate_MyArray"
        return call

    mip = types.MemInfoPointer(types.voidptr)  # return untyped pointer
    sig = typing.signature(mip, allocsize, align)
    return sig, impl


class TestNdarraySubclasses(TestCase):

    def setUp(self):
        global _logger
        _logger = []

    def test_myarray_return(self):
        """This test `types.Array.box_type`
        """
        @njit
        def foo(a):
            return a + 1

        buf = np.arange(4)
        a = myarray(buf.shape, buf.dtype, buf)
        expected = foo.py_func(a)
        got = foo(a)
        self.assertIsInstance(got, myarray)
        self.assertIs(type(expected), type(got))
        self.assertPreciseEqual(expected, got)

    def test_myarray_passthru(self):
        @njit
        def foo(a):
            return a

        buf = np.arange(4)
        a = myarray(buf.shape, buf.dtype, buf)
        expected = foo.py_func(a)
        got = foo(a)
        self.assertIsInstance(got, myarray)
        self.assertIs(type(expected), type(got))
        self.assertPreciseEqual(expected, got)

    def test_myarray_convert(self):
        @njit
        def foo(buf):
            return myarray(buf.shape, buf.dtype, buf)

        buf = np.arange(4)
        expected = foo.py_func(buf)
        got = foo(buf)
        self.assertIsInstance(got, myarray)
        self.assertIs(type(expected), type(got))
        self.assertPreciseEqual(expected, got)

    def test_myarray_asarray_non_jit(self):
        def foo(buf):
            coverted = myarray(buf.shape, buf.dtype, buf)
            return np.asarray(coverted) + buf

        buf = np.arange(4)
        got = foo(buf)
        self.assertIs(type(got), np.ndarray)
        self.assertPreciseEqual(got, buf + buf)

    @unittest.expectedFailure
    def test_myarray_asarray(self):
        @njit
        def foo(buf):
            coverted = myarray(buf.shape, buf.dtype, buf)
            return np.asarray(coverted)

        buf = np.arange(4)
        got = foo(buf)
        # the following fails because our np.asarray is returning the source
        # array type
        self.assertIs(type(got), np.ndarray)

    def test_myarray_ufunc_unsupported(self):
        @njit
        def foo(buf):
            coverted = myarray(buf.shape, buf.dtype, buf)
            return coverted + coverted

        buf = np.arange(4, dtype=np.float32)
        with self.assertRaises(TypingError) as raises:
            foo(buf)
        self.assertIn(
            "unsupported use of ufunc <ufunc 'add'> on MyArray(1, float32, C)",
            str(raises.exception),
        )

    def test_myarray_allocator_override(self):
        """
        Checks that our custom allocator is used
        """
        @njit
        def foo(a):
            b = a + np.arange(a.size, dtype=np.float64)
            c = a + 1j
            return b, c

        buf = np.arange(4, dtype=np.float64)
        a = myarray(buf.shape, buf.dtype, buf)

        expected = foo.py_func(a)
        got = foo(a)

        self.assertPreciseEqual(got, expected)

        logged_lines = _logger

        targetctx = cpu_target.target_context
        nb_dtype = typeof(buf.dtype)
        align = targetctx.get_preferred_array_alignment(nb_dtype)
        self.assertEqual(logged_lines, [
            ("LOG _ol_array_allocate", expected[0].nbytes, align),
            ("LOG _ol_array_allocate", expected[1].nbytes, align),
        ])


if __name__ == "__main__":
    unittest.main()
