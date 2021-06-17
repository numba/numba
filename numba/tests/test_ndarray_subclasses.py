import builtins
import unittest
from numbers import Number

import numpy as np
from llvmlite import ir

import numba
from numba import njit, typeof
from numba.core import cgutils, types, typing
from numba.core.pythonapi import box
from numba.core.registry import cpu_target
from numba.extending import (intrinsic, lower_builtin, overload_classmethod,
                             register_model, type_callable, typeof_impl)
from numba.np import numpy_support

from numba.tests.support import captured_stdout, TestCase


class myarray(np.ndarray):
    # Tell Numba to not seamlessly treat this type as a regular ndarray.
    __numba_array_subtype_dispatch__ = True

    # Interoperate with Numpy outside of Numba.
    def __array__(self):
        return self

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
    def __init__(
        self,
        dtype,
        ndim,
        layout,
        readonly=False,
        name=None,
        aligned=True,
        addrspace=None,
    ):
        name = "MyArray:ndarray(%s, %sd, %s)" % (dtype, ndim, layout)
        super(MyArrayType, self).__init__(
            dtype,
            ndim,
            layout,
            readonly=readonly,
            name=name,
        )

    # Tell Numba typing how to combine MyArrayType with other ndarray types.
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            for inp in inputs:
                if not isinstance(
                    inp, (MyArrayType, types.Array, types.Number)
                ):
                    return None

            return MyArrayType
        else:
            return None


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
        return MyArrayType(
            dtype=buf.dtype, ndim=len(shape), layout=buf.layout
        )

    return typer


@lower_builtin(myarray, types.UniTuple, types.DType, types.Array)
def impl_myarray(context, builder, sig, args):
    shape, dtype, buf = args
    context.nrt.incref(builder, sig.args[-1], buf)
    return buf


@box(MyArrayType)
def box_array(typ, val, c):
    nativearycls = c.context.make_array(typ)
    nativeary = nativearycls(c.context, c.builder, value=val)
    if c.context.enable_nrt:
        np_dtype = numpy_support.as_dtype(typ.dtype)
        dtypeptr = c.env_manager.read_const(c.env_manager.add_const(np_dtype))
        # Steals NRT ref
        newary = c.pyapi.nrt_adapt_ndarray_to_python(typ, val, dtypeptr)
        return newary
    else:
        parent = nativeary.parent
        c.pyapi.incref(parent)
        return parent


@overload_classmethod(MyArrayType, "_allocate")
def _ol_array_allocate(cls, allocsize, align):
    """Implements a Numba-only classmethod on the array type.
    """
    def impl(cls, allocsize, align):
        print("LOG _ol_array_allocate", allocsize, align)
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
    def test_allocator_override(self):
        """
        Checks that our custom allocator is used
        """
        @njit
        def foo(a):
            b = a + a
            c = a + 1j
            return b, c

        buf = np.arange(4)
        a = myarray(buf.shape, buf.dtype, buf)

        expected = foo.py_func(a)
        with captured_stdout() as stdout:
            got = foo(a)

        self.assertPreciseEqual(got, expected)

        logged_lines = stdout.getvalue().splitlines()

        targetctx = cpu_target.target_context
        nb_dtype = typeof(buf.dtype)
        align = targetctx.get_preferred_array_alignment(nb_dtype)
        self.assertEqual(logged_lines, [
            f"LOG _ol_array_allocate {expected[0].nbytes} {align}",
            f"LOG _ol_array_allocate {expected[1].nbytes} {align}",
        ])


if __name__ == "__main__":
    unittest.main()
