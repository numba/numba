from __future__ import print_function, division, absolute_import

import numpy
import math
import sys
import itertools
from collections import namedtuple

from llvm.core import Constant, Type, ICMP_UGT


from .imputils import implement, Registry
from .. import typing, types, cgutils, numpy_support
from ..config import PYVERSION
from ..numpy_support import (ufunc_find_matching_loop,
                             numpy_letter_types_to_numba_types,
                             numba_types_to_numpy_letter_types)
registry = Registry()
register = registry.register


class npy:
    """This will be used as an index of the npy_* functions"""
    pass


def _default_promotion_for_type(ty):
    """returns the default type to be used when generating code
    associated to the type ty."""
    if ty in types.real_domain:
        promote_type = types.float64
    elif ty in types.signed_domain:
        promote_type = types.int64
    elif ty in types.unsigned_domain:
        promote_type = types.uint64
    else:
        assert False, "type {0} not supported.".format(ty)

    return promote_type

########################################################################

# In the way we generate code, ufuncs work with scalar as well as
# with array arguments. The following helper classes help dealing
# with scalar and array arguments in a regular way.
#
# In short, the classes provide a uniform interface. The interface
# handles the indexing of as many dimensions as the array may have.
# For scalars, all indexing is ignored and when the value is read,
# the scalar is returned. For arrays code for actual indexing is
# generated and reading performs the appropriate indirection.

class _ScalarIndexingHelper(object):
    def update_indices(self, loop_indices, name):
        pass

    def as_values(self):
        pass


class _ScalarHelper(object):
    """Helper class to handle scalar arguments (and result).
    Note that store_data is only used when generating code for
    a scalar ufunc and to write the output value.

    For loading, the value is directly used without having any
    kind of indexing nor memory backing it up. This is the use
    for input arguments.

    For storing, a variable is created in the stack where the
    value will be written.

    Note that it is not supported (as it is unneeded for our
    current use-cases) reading back a stored value. This class
    will always "load" the original value it got at its creation.
    """
    def __init__(self, ctxt, bld, val, ty):
        self.context = ctxt
        self.builder = bld
        self.val = val
        self.base_type = ty
        intpty = ctxt.get_value_type(types.intp)
        self.shape = [Constant.int(intpty, 1)]
        self._ptr = cgutils.alloca_once(bld, ctxt.get_data_type(ty))

    def create_iter_indices(self):
        return _ScalarIndexingHelper()

    def load_data(self, indices):
        return self.val

    def store_data(self, indices, val):
        self.builder.store(val, self._ptr)

    @property
    def return_val(self):
        return self.builder.load(self._ptr)


class _ArrayIndexingHelper(namedtuple('_ArrayIndexingHelper',
                                      ('array', 'indices'))):
    def update_indices(self, loop_indices, name):
        bld = self.array.builder
        intpty = self.array.context.get_value_type(types.intp)
        ONE = Constant.int(Type.int(intpty.width), 1)

        # we are only interested in as many inner dimensions as dimensions
        # the indexed array has (the outer dimensions are broadcast, so
        # ignoring the outer indices produces the desired result.
        indices = loop_indices[len(loop_indices) - len(self.indices):]
        for src, dst, dim in zip(indices, self.indices, self.array.shape):
            cond = bld.icmp(ICMP_UGT, dim, ONE)
            with cgutils.ifthen(bld, cond):
                bld.store(src, dst)

    def as_values(self):
        """
        The indexing helper is built using alloca for each value, so it
        actually contains pointers to the actual indices to load. Note
        that update_indices assumes the same. This method returns the
        indices as values
        """
        bld=self.array.builder
        return [bld.load(index) for index in self.indices]


class _ArrayHelper(namedtuple('_ArrayHelper', ('context', 'builder', 'ary',
                                               'shape', 'strides', 'data',
                                               'layout', 'base_type', 'ndim',
                                               'return_val'))):
    """Helper class to handle array arguments/result.
    It provides methods to generate code loading/storing specific
    items as well as support code for handling indices.
    """
    def create_iter_indices(self):
        intpty = self.context.get_value_type(types.intp)
        ZERO = Constant.int(Type.int(intpty.width), 0)

        indices = []
        for i in range(self.ndim):
            x = cgutils.alloca_once(self.builder, Type.int(intpty.width))
            self.builder.store(ZERO, x)
            indices.append(x)
        return _ArrayIndexingHelper(self, indices)

    def _load_effective_address(self, indices):
        return cgutils.get_item_pointer2(self.builder,
                                         data=self.data,
                                         shape=self.shape,
                                         strides=self.strides,
                                         layout=self.layout,
                                         inds=indices)

    def load_data(self, indices):
        return self.builder.load(self._load_effective_address(indices))

    def store_data(self, indices, value):
        assert self.context.get_data_type(self.base_type) == value.type
        self.builder.store(value, self._load_effective_address(indices))


def _prepare_argument(ctxt, bld, inp, tyinp, where='input operand'):
    """returns an instance of the appropriate Helper (either
    _ScalarHelper or _ArrayHelper) class to handle the argument.
    using the polymorphic interface of the Helper classes, scalar
    and array cases can be handled with the same code"""
    if isinstance(tyinp, types.Array):
        ary     = ctxt.make_array(tyinp)(ctxt, bld, inp)
        shape   = cgutils.unpack_tuple(bld, ary.shape, tyinp.ndim)
        strides = cgutils.unpack_tuple(bld, ary.strides, tyinp.ndim)
        return _ArrayHelper(ctxt, bld, ary, shape, strides, ary.data,
                            tyinp.layout, tyinp.dtype, tyinp.ndim, inp)
    elif tyinp in types.number_domain:
        return _ScalarHelper(ctxt, bld, inp, tyinp)
    else:
        raise TypeError('unknown type for {0}'.format(where))


def npy_math_extern(fn, fnty):
    setattr(npy, fn, fn)
    fn_sym = eval("npy."+fn)
    fn_arity = len(fnty.args)

    n = "numba.npymath." + fn
    def ref_impl(context, builder, sig, args):
        mod = cgutils.get_module(builder)
        inner_fn = mod.get_or_insert_function(fnty, name=n)
        return builder.call(inner_fn, args)

    # This registers the function using different combinations of
    # input types that can be cast to the actual function type.
    #
    # Current limitation is that it only does so for homogeneous
    # source types. Note that it may be a better idea not providing
    # these specialization and let the ufunc generator functions
    # insert the appropriate castings before calling.
    #
    # TODO:
    # Either let the function only register the native version without
    # cast or provide the full range of specializations for functions
    # with arity > 1.
    ty_dst = types.float64
    for ty_src in [types.int64, types.uint64, types.float64]:
        @register
        @implement(fn_sym, *[ty_src]*fn_arity)
        def _impl(context, builder, sig, args):
            cast_vals = args
            if ty_dst != ty_src:
                cast = context.cast
                cast_vals = [cast(builder, val, ty_src, ty_dst) for val in args]
            sig = typing.signature(*[ty_dst]*(len(cast_vals)+1))
            return ref_impl(context, builder, sig, cast_vals)


def numpy_ufunc_kernel(context, builder, sig, args, kernel_class,
                       explicit_output=True):
    if not explicit_output:
        args.append(Constant.null(context.get_value_type(sig.return_type)))
        tyargs = sig.args + (sig.return_type,)
    else:
        tyargs = sig.args
    arguments = [_prepare_argument(context, builder, arg, tyarg)
                 for arg, tyarg in zip(args, tyargs)]

    inputs = arguments[0:-1]
    output = arguments[-1]

    outer_sig = [a.base_type for a in arguments]
    #signature expects return type first, while we have it last:
    outer_sig = outer_sig[-1:] + outer_sig[:-1]
    outer_sig = typing.signature(*outer_sig)
    kernel = kernel_class(context, builder, outer_sig)
    intpty = context.get_value_type(types.intp)

    indices = [inp.create_iter_indices() for inp in inputs]

    loopshape = output.shape
    with cgutils.loop_nest(builder, loopshape, intp=intpty) as loop_indices:
        vals_in = []
        for i, (index, arg) in enumerate(zip(indices, inputs)):
            index.update_indices(loop_indices, i)
            vals_in.append(arg.load_data(index.as_values()))

        val_out = kernel.generate(*vals_in)
        output.store_data(loop_indices, val_out)
    return arguments[-1].return_val


# Kernels are the code to be executed inside the multidimensional loop.
class _Kernel(object):
    pass


def _function_with_cast(op, inner_sig):
    """a kernel implemented by a function that only exists in one signature
    op is the operation (function
    inner_sig is the signature of op. Operands will be cast to that signature
    """
    class _KernelImpl(_Kernel):
        def __init__(self, context, builder, outer_sig):
            """
            op is the operation
            outer_sig is the outer type signature (the signature of the ufunc)
            inner_sig is the inner type signature (the signature of the
                      operation itself)
            """
            self.context = context
            self.builder = builder
            self.fnwork = context.get_function(op, inner_sig)
            self.inner_sig = inner_sig
            self.outer_sig = outer_sig

        def generate(self, *args):
            # convert args from the ufunc types to the one of the
            # kernel operation
            cast_args = [self.context.cast(self.builder, val, inty, outy)
                         for val, inty, outy in zip(args, self.outer_sig.args,
                                                    self.inner_sig.args)]
            # perform the operation
            res = self.fnwork(self.builder, cast_args)
            # return the result converted to the type of the ufunc
            # operation
            return self.context.cast(self.builder, res,
                                     self.inner_sig.return_type,
                                     self.outer_sig.return_type)

    return _KernelImpl


def _homogeneous_function(op, alias=None):
    """A function that uses the underlying ufunc loop information to chose
    an implementation for the operation op. It uses the loop
    information provided by the ufunc in alias. Alias defaults to the
    op if not provided.

    Using the loop information, code is generated that simulates the
    process of: converting input arguments (outer_sig) to the input
    arguments specifies by the ufunc selected loop. The operation is
    performed in the converted arguments, resulting in a value as
    specified by the selected loop information. Finally, the resulting
    value is converted to the requested output type (in outer_sig).
    """
    class _KernelImpl(_Kernel):
        def __init__(self, context, builder, outer_sig):
            ufunc = alias if alias is not None else op
            self.context = context
            self.builder = builder
            letter_arg_types = numba_types_to_numpy_letter_types(
                outer_sig.args[0:ufunc.nin])
            self.loop = ufunc_find_matching_loop(ufunc, letter_arg_types)
            self.loop_in_types = numpy_letter_types_to_numba_types(
                self.loop[:ufunc.nin])
            self.loop_out_types = numpy_letter_types_to_numba_types(
                self.loop[-ufunc.nout:])
            # only one outpt supported for now.
            assert(len(self.loop_out_types) == 1)
            self.outer_sig = outer_sig

        def generate(self, *args):
            inner_sig = typing.signature(self.loop_out_types[0],
                                         *self.loop_in_types)
            fn = self.context.get_function(op, inner_sig)
            cast_args = [self.context.cast(self.builder, val, inty, outty)
                         for val, inty, outty in zip(args, self.outer_sig.args,
                                                     self.loop_in_types)]
            res = fn(self.builder, cast_args)
            return self.context.cast(self.builder, res, self.loop_out_types[0],
                                     self.outer_sig.return_type)

    return _KernelImpl


def _division(ufunc, operator):
    """A kernel for division. It supports three kinds of division:
    operator is either '/' for true_division, '//' for floor_division
    or '/?' for python2 legacy divide
    """
    class _KernelImpl(_Kernel):
        def __init__(self, context, builder, outer_sig):
            self.context = context
            self.builder = builder
            letter_arg_types = numba_types_to_numpy_letter_types(
                outer_sig.args[0:ufunc.nin])
            self.loop = ufunc_find_matching_loop(ufunc, letter_arg_types)
            self.loop_in_types = numpy_letter_types_to_numba_types(
                self.loop[:ufunc.nin])
            self.loop_out_types = numpy_letter_types_to_numba_types(
                self.loop[-ufunc.nout:])
            self.outer_sig = outer_sig

        def generate(self,*args):
            assert len(args) == 2 # numerator and denominator
            builder=self.builder
            context=self.context
            tyinputs = self.outer_sig.args
            tyout = self.outer_sig.return_type
            tyout_llvm = context.get_data_type(tyout)
            inner_sig = typing.signature(self.loop_out_types[0],
                                         *self.loop_in_types)
            fn = context.get_function(operator, inner_sig)
            num, den = args

            iszero = cgutils.is_scalar_zero(builder, den)
            with cgutils.ifelse(builder, iszero, expect=False) as (then, orelse):
                with then:
                    # Divide by zero
                    if ((tyinputs[0] in types.real_domain or
                         tyinputs[1] in types.real_domain) or
                        not numpy_support.int_divbyzero_returns_zero) or \
                        operator=='/':
                        # If num is float and is 0 also, return Nan; else
                        # return Inf
                        outltype = context.get_data_type(types.float64)
                        shouldretnan = cgutils.is_scalar_zero(builder, num)
                        nan = Constant.real(outltype, float("nan"))
                        inf = Constant.real(outltype, float("inf"))
                        tempres = builder.select(shouldretnan, nan, inf)
                        res_then = context.cast(builder, tempres, types.float64,
                                                tyout)
                    elif tyout in types.signed_domain and \
                            not numpy_support.int_divbyzero_returns_zero:
                        res_then = Constant.int(tyout_llvm,
                                                0x1 << (den.type.width-1))
                    else:
                        res_then = Constant.null(tyout_llvm)
                    bb_then = builder.basic_block
                with orelse:
                    # Normal
                    cast_args = [self.context.cast(self.builder, val, inty,
                                                   outty)
                                 for val, inty, outty
                                 in zip(args, self.outer_sig.args,
                                        self.loop_in_types)]
                    tempres = fn(builder, cast_args)
                    res_else = context.cast(builder, tempres,
                                            self.loop_out_types[0], tyout)
                    bb_else = builder.basic_block
            out = builder.phi(tyout_llvm)
            out.add_incoming(res_then, bb_then)
            out.add_incoming(res_else, bb_else)
            return out
    return _KernelImpl

################################################################################
# Helper functions that register the ufuncs

def register_unary_ufunc_kernel(ufunc, kernel):
    def unary_ufunc(context, builder, sig, args):
        return numpy_ufunc_kernel(context, builder, sig, args, kernel)

    def unary_scalar_ufunc(context, builder, sig, args):
        return numpy_ufunc_kernel(context, builder, sig, args, kernel,
                                  explicit_output=False)

    register(implement(ufunc, types.Kind(types.Array),
        types.Kind(types.Array))(unary_ufunc))
    for ty in types.number_domain:
        register(implement(ufunc, ty,
            types.Kind(types.Array))(unary_ufunc))
    for ty in types.number_domain:
        register(implement(ufunc, ty)(unary_scalar_ufunc)) # scalar


def register_binary_ufunc_kernel(ufunc, kernel):
    def binary_ufunc(context, builder, sig, args):
        return numpy_ufunc_kernel(context, builder, sig, args, kernel)

    def binary_scalar_ufunc(context, builder, sig, args):
        return numpy_ufunc_kernel(context, builder, sig, args, kernel,
                                  explicit_output=False)

    register(implement(ufunc, types.Kind(types.Array), types.Kind(types.Array),
        types.Kind(types.Array))(binary_ufunc))
    for ty in types.number_domain:
        register(implement(ufunc, ty, types.Kind(types.Array),
            types.Kind(types.Array))(binary_ufunc))
        register(implement(ufunc, types.Kind(types.Array), ty,
            types.Kind(types.Array))(binary_ufunc))
    for ty1, ty2 in itertools.product(types.number_domain, types.number_domain):
        register(implement(ufunc, ty1, ty2,
            types.Kind(types.Array))(binary_ufunc))
        register(implement(ufunc, ty1, ty2)(binary_scalar_ufunc)) # scalar


################################################################################
# Actual registering of supported ufuncs

_float_unary_function_type = Type.function(Type.double(), [Type.double()])
_float_binary_function_type = Type.function(Type.double(),
                                            [Type.double(), Type.double()])
_float_unary_sig = typing.signature(types.float64, types.float64)
_float_binary_sig = typing.signature(types.float64, types.float64,
                                     types.float64)


# _externs will be used to register ufuncs.
# each tuple contains the ufunc to be translated. That ufunc will be converted to
# an equivalent loop that calls the function in the npymath support module (registered
# as external function as "numba.npymath."+func
_externs = [
    (numpy.exp, "exp"),
    (numpy.exp2, "exp2"),
    (numpy.expm1, "expm1"),
    (numpy.log, "log"),
    (numpy.log2, "log2"),
    (numpy.log10, "log10"),
    (numpy.log1p, "log1p"),
    (numpy.deg2rad, "deg2rad"),
    (numpy.rad2deg, "rad2deg"),
    (numpy.sin, "sin"),
    (numpy.cos, "cos"),
    (numpy.tan, "tan"),
    (numpy.sinh, "sinh"),
    (numpy.cosh, "cosh"),
    (numpy.tanh, "tanh"),
    (numpy.arcsin, "asin"),
    (numpy.arccos, "acos"),
    (numpy.arctan, "atan"),
    (numpy.arcsinh, "asinh"),
    (numpy.arccosh, "acosh"),
    (numpy.arctanh, "atanh"),
    (numpy.sqrt, "sqrt"),
    (numpy.floor, "floor"),
    (numpy.ceil, "ceil"),
    (numpy.trunc, "trunc"),
    (numpy.rint, "rint"),
    (numpy.fabs, "fabs"),
]

for sym, name in _externs:
    npy_math_extern(name, _float_unary_function_type)
    register_unary_ufunc_kernel(sym, _function_with_cast(getattr(npy, name), _float_unary_sig))

# radians and degrees ufuncs are equivalent to deg2rad and rad2deg resp.
# register them.
register_unary_ufunc_kernel(numpy.degrees, _function_with_cast(npy.rad2deg, _float_unary_sig))
register_unary_ufunc_kernel(numpy.radians, _function_with_cast(npy.deg2rad, _float_unary_sig))

# the following ufuncs rely on functions that are not based on a function
# from npymath
register_unary_ufunc_kernel(numpy.absolute, _homogeneous_function(types.abs_type, numpy.absolute))
register_unary_ufunc_kernel(numpy.sign, _homogeneous_function(types.sign_type, numpy.sign))
register_unary_ufunc_kernel(numpy.negative, _homogeneous_function(types.neg_type, numpy.negative))

# for these we mostly rely on code generation for python operators.
register_binary_ufunc_kernel(numpy.add, _homogeneous_function('+', numpy.add))
register_binary_ufunc_kernel(numpy.subtract, _homogeneous_function('-', numpy.subtract))
register_binary_ufunc_kernel(numpy.multiply, _homogeneous_function('*', numpy.multiply))
if not PYVERSION >= (3, 0):
    register_binary_ufunc_kernel(numpy.divide, _division(numpy.divide, '/?'))
register_binary_ufunc_kernel(numpy.floor_divide, _division(numpy.floor_divide, '//'))
register_binary_ufunc_kernel(numpy.true_divide, _division(numpy.true_divide, '/'))
register_binary_ufunc_kernel(numpy.power, _function_with_cast('**', _float_binary_sig))


_externs_2 = [
    (numpy.arctan2, "atan2"),
    (numpy.logaddexp, "logaddexp"),
    (numpy.logaddexp2, "logaddexp2"),
    (numpy.hypot, "hypot"),
]

for sym, name in _externs_2:
    npy_math_extern(name, _float_binary_function_type)
    register_binary_ufunc_kernel(sym, _function_with_cast(getattr(npy, name), _float_binary_sig))

del _float_binary_function_type, _float_binary_sig
del _float_unary_function_type, _float_unary_sig
