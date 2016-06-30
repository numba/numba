"""
Implementation of functions in the Numpy package.
"""

from __future__ import print_function, division, absolute_import

import math
import sys
import itertools
from collections import namedtuple

from llvmlite.llvmpy import core as lc

import numpy as np

from . import builtins, callconv, ufunc_db, arrayobj
from .imputils import Registry, impl_ret_new_ref, force_error_model
from .. import typing, types, cgutils, numpy_support, utils
from ..config import PYVERSION
from ..numpy_support import ufunc_find_matching_loop, select_array_wrapper
from ..typing import npydecl

registry = Registry()
lower = registry.lower


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
        self.shape = [lc.Constant.int(intpty, 1)]

        lty = ctxt.get_data_type(ty) if ty != types.boolean else lc.Type.int(1)
        self._ptr = cgutils.alloca_once(bld, lty)

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
        ONE = lc.Constant.int(lc.Type.int(intpty.width), 1)

        # we are only interested in as many inner dimensions as dimensions
        # the indexed array has (the outer dimensions are broadcast, so
        # ignoring the outer indices produces the desired result.
        indices = loop_indices[len(loop_indices) - len(self.indices):]
        for src, dst, dim in zip(indices, self.indices, self.array.shape):
            cond = bld.icmp(lc.ICMP_UGT, dim, ONE)
            with bld.if_then(cond):
                bld.store(src, dst)

    def as_values(self):
        """
        The indexing helper is built using alloca for each value, so it
        actually contains pointers to the actual indices to load. Note
        that update_indices assumes the same. This method returns the
        indices as values
        """
        bld = self.array.builder
        return [bld.load(index) for index in self.indices]


class _ArrayHelper(namedtuple('_ArrayHelper', ('context', 'builder',
                                               'shape', 'strides', 'data',
                                               'layout', 'base_type', 'ndim',
                                               'return_val'))):
    """Helper class to handle array arguments/result.
    It provides methods to generate code loading/storing specific
    items as well as support code for handling indices.
    """
    def create_iter_indices(self):
        intpty = self.context.get_value_type(types.intp)
        ZERO = lc.Constant.int(lc.Type.int(intpty.width), 0)

        indices = []
        for i in range(self.ndim):
            x = cgutils.alloca_once(self.builder, lc.Type.int(intpty.width))
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
        model = self.context.data_model_manager[self.base_type]
        ptr = self._load_effective_address(indices)
        return model.load_from_data_pointer(self.builder, ptr)

    def store_data(self, indices, value):
        ctx = self.context
        bld = self.builder
        store_value = ctx.get_value_as_data(bld, self.base_type, value)
        assert ctx.get_data_type(self.base_type) == store_value.type
        bld.store(store_value, self._load_effective_address(indices))


def _prepare_argument(ctxt, bld, inp, tyinp, where='input operand'):
    """returns an instance of the appropriate Helper (either
    _ScalarHelper or _ArrayHelper) class to handle the argument.
    using the polymorphic interface of the Helper classes, scalar
    and array cases can be handled with the same code"""
    if isinstance(tyinp, types.ArrayCompatible):
        ary     = ctxt.make_array(tyinp)(ctxt, bld, inp)
        shape   = cgutils.unpack_tuple(bld, ary.shape, tyinp.ndim)
        strides = cgutils.unpack_tuple(bld, ary.strides, tyinp.ndim)
        return _ArrayHelper(ctxt, bld, shape, strides, ary.data,
                            tyinp.layout, tyinp.dtype, tyinp.ndim, inp)
    elif tyinp in types.number_domain | set([types.boolean]):
        return _ScalarHelper(ctxt, bld, inp, tyinp)
    else:
        raise TypeError('unknown type for {0}: {1}'.format(where, str(tyinp)))


_broadcast_onto_sig = types.intp(types.intp, types.CPointer(types.intp),
                                 types.intp, types.CPointer(types.intp))
def _broadcast_onto(src_ndim, src_shape, dest_ndim, dest_shape):
    '''Low-level utility function used in calculating a shape for
    an implicit output array.  This function assumes that the
    destination shape is an LLVM pointer to a C-style array that was
    already initialized to a size of one along all axes.

    Returns an integer value:
    >= 1  :  Succeeded.  Return value should equal the number of dimensions in
             the destination shape.
    0     :  Failed to broadcast because source shape is larger than the
             destination shape (this case should be weeded out at type
             checking).
    < 0   :  Failed to broadcast onto destination axis, at axis number ==
             -(return_value + 1).
    '''
    if src_ndim > dest_ndim:
        # This check should have been done during type checking, but
        # let's be defensive anyway...
        return 0
    else:
        src_index = 0
        dest_index = dest_ndim - src_ndim
        while src_index < src_ndim:
            src_dim_size = src_shape[src_index]
            dest_dim_size = dest_shape[dest_index]
            # Check to see if we've already mutated the destination
            # shape along this axis.
            if dest_dim_size != 1:
                # If we have mutated the destination shape already,
                # then the source axis size must either be one,
                # or the destination axis size.
                if src_dim_size != dest_dim_size and src_dim_size != 1:
                    return -(dest_index + 1)
            elif src_dim_size != 1:
                # If the destination size is still its initial
                dest_shape[dest_index] = src_dim_size
            src_index += 1
            dest_index += 1
    return dest_index

def _build_array(context, builder, array_ty, input_types, inputs):
    """Utility function to handle allocation of an implicit output array
    given the target context, builder, output array type, and a list of
    _ArrayHelper instances.
    """
    intp_ty = context.get_value_type(types.intp)
    def make_intp_const(val):
        return context.get_constant(types.intp, val)

    ZERO = make_intp_const(0)
    ONE = make_intp_const(1)

    src_shape = cgutils.alloca_once(builder, intp_ty, array_ty.ndim,
                                    "src_shape")
    dest_ndim = make_intp_const(array_ty.ndim)
    dest_shape = cgutils.alloca_once(builder, intp_ty, array_ty.ndim,
                                     "dest_shape")
    dest_shape_addrs = tuple(cgutils.gep_inbounds(builder, dest_shape, index)
                             for index in range(array_ty.ndim))

    # Initialize the destination shape with all ones.
    for dest_shape_addr in dest_shape_addrs:
        builder.store(ONE, dest_shape_addr)

    # For each argument, try to broadcast onto the destination shape,
    # mutating along any axis where the argument shape is not one and
    # the destination shape is one.
    for arg_number, arg in enumerate(inputs):
        if not hasattr(arg, "ndim"): # Skip scalar arguments
            continue
        arg_ndim = make_intp_const(arg.ndim)
        for index in range(arg.ndim):
            builder.store(arg.shape[index],
                          cgutils.gep_inbounds(builder, src_shape, index))
        arg_result = context.compile_internal(
            builder, _broadcast_onto, _broadcast_onto_sig,
            [arg_ndim, src_shape, dest_ndim, dest_shape])
        with cgutils.if_unlikely(builder,
                                 builder.icmp(lc.ICMP_SLT, arg_result, ONE)):
            msg = "unable to broadcast argument %d to output array" % (
                arg_number,)
            context.call_conv.return_user_exc(builder, ValueError, (msg,))

    real_array_ty = array_ty.as_array

    dest_shape_tup = tuple(builder.load(dest_shape_addr)
                           for dest_shape_addr in dest_shape_addrs)
    array_val = arrayobj._empty_nd_impl(context, builder, real_array_ty,
                                        dest_shape_tup)

    # Get the best argument to call __array_wrap__ on
    array_wrapper_index = select_array_wrapper(input_types)
    array_wrapper_ty = input_types[array_wrapper_index]
    try:
        # __array_wrap__(source wrapped array, out array) -> out wrapped array
        array_wrap = context.get_function('__array_wrap__',
                                          array_ty(array_wrapper_ty, real_array_ty))
    except NotImplementedError:
        # If it's the same priority as a regular array, assume we
        # should use the allocated array unchanged.
        if array_wrapper_ty.array_priority != types.Array.array_priority:
            raise
        out_val = array_val._getvalue()
    else:
        wrap_args = (inputs[array_wrapper_index].return_val, array_val._getvalue())
        out_val = array_wrap(builder, wrap_args)

    ndim = array_ty.ndim
    shape   = cgutils.unpack_tuple(builder, array_val.shape, ndim)
    strides = cgutils.unpack_tuple(builder, array_val.strides, ndim)
    return _ArrayHelper(context, builder, shape, strides, array_val.data,
                        array_ty.layout, array_ty.dtype, ndim,
                        out_val)


def numpy_ufunc_kernel(context, builder, sig, args, kernel_class,
                       explicit_output=True):
    # This is the code generator that builds all the looping needed
    # to execute a numpy functions over several dimensions (including
    # scalar cases).
    #
    # context - the code generation context
    # builder - the code emitter
    # sig - signature of the ufunc
    # args - the args to the ufunc
    # kernel_class -  a code generating subclass of _Kernel that provides
    # explicit_output - if the output was explicit in the call
    #                   (ie: np.add(x,y,r))

    arguments = [_prepare_argument(context, builder, arg, tyarg)
                 for arg, tyarg in zip(args, sig.args)]
    if not explicit_output:
        ret_ty = sig.return_type
        if isinstance(ret_ty, types.ArrayCompatible):
            output = _build_array(context, builder, ret_ty, sig.args, arguments)
        else:
            output = _prepare_argument(
                context, builder,
                lc.Constant.null(context.get_value_type(ret_ty)), ret_ty)
        arguments.append(output)
    elif context.enable_nrt:
        # Incref the output
        context.nrt_incref(builder, sig.return_type, args[-1])

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
    out = arguments[-1].return_val
    return impl_ret_new_ref(context, builder, sig.return_type, out)


# Kernels are the code to be executed inside the multidimensional loop.
class _Kernel(object):
    def __init__(self, context, builder, outer_sig):
        self.context = context
        self.builder = builder
        self.outer_sig = outer_sig

    def cast(self, val, fromty, toty):
        """Numpy uses cast semantics that are different from standard Python
        (for example, it does allow casting from complex to float).

        This method acts as a patch to context.cast so that it allows
        complex to real/int casts.

        """
        if (isinstance(fromty, types.Complex) and
            not isinstance(toty, types.Complex)):
            # attempt conversion of the real part to the specified type.
            # note that NumPy issues a warning in this kind of conversions
            newty = fromty.underlying_float
            attr = self.context.get_getattr(fromty, 'real')
            val = attr(self.context, self.builder, fromty, val, 'real')
            fromty = newty
            # let the regular cast do the rest...

        return self.context.cast(self.builder, val, fromty, toty)


def _ufunc_db_function(ufunc):
    """Use the ufunc loop type information to select the code generation
    function from the table provided by the dict_of_kernels. The dict
    of kernels maps the loop identifier to a function with the
    following signature: (context, builder, signature, args).

    The loop type information has the form 'AB->C'. The letters to the
    left of '->' are the input types (specified as NumPy letter
    types).  The letters to the right of '->' are the output
    types. There must be 'ufunc.nin' letters to the left of '->', and
    'ufunc.nout' letters to the right.

    For example, a binary float loop resulting in a float, will have
    the following signature: 'ff->f'.

    A given ufunc implements many loops. The list of loops implemented
    for a given ufunc can be accessed using the 'types' attribute in
    the ufunc object. The NumPy machinery selects the first loop that
    fits a given calling signature (in our case, what we call the
    outer_sig). This logic is mimicked by 'ufunc_find_matching_loop'.
    """

    class _KernelImpl(_Kernel):
        def __init__(self, context, builder, outer_sig):
            super(_KernelImpl, self).__init__(context, builder, outer_sig)
            loop = ufunc_find_matching_loop(
                ufunc, outer_sig.args + (outer_sig.return_type,))
            self.fn = ufunc_db.get_ufunc_info(ufunc).get(loop.ufunc_sig)
            self.inner_sig = typing.signature(
                *(loop.outputs + loop.inputs))

            if self.fn is None:
                msg = "Don't know how to lower ufunc '{0}' for loop '{1}'"
                raise NotImplementedError(msg.format(ufunc.__name__, loop))

        def generate(self, *args):
            isig = self.inner_sig
            osig = self.outer_sig

            cast_args = [self.cast(val, inty, outty)
                         for val, inty, outty in zip(args, osig.args,
                                                     isig.args)]
            with force_error_model(self.context, 'numpy'):
                res = self.fn(self.context, self.builder, isig, cast_args)
            dmm = self.context.data_model_manager
            res = dmm[isig.return_type].from_return(self.builder, res)
            return self.cast(res, isig.return_type, osig.return_type)

    return _KernelImpl


################################################################################
# Helper functions that register the ufuncs

_kernels = {} # Temporary map from ufunc's to their kernel implementation class

def register_unary_ufunc_kernel(ufunc, kernel):
    def unary_ufunc(context, builder, sig, args):
        return numpy_ufunc_kernel(context, builder, sig, args, kernel)

    def unary_ufunc_no_explicit_output(context, builder, sig, args):
        return numpy_ufunc_kernel(context, builder, sig, args, kernel,
                                  explicit_output=False)

    _any = types.Any

    # (array or scalar, out=array)
    lower(ufunc, _any, types.Array)(unary_ufunc)
    # (array or scalar)
    lower(ufunc, _any)(unary_ufunc_no_explicit_output)

    _kernels[ufunc] = kernel


def register_binary_ufunc_kernel(ufunc, kernel):
    def binary_ufunc(context, builder, sig, args):
        return numpy_ufunc_kernel(context, builder, sig, args, kernel)

    def binary_ufunc_no_explicit_output(context, builder, sig, args):
        return numpy_ufunc_kernel(context, builder, sig, args, kernel,
                                  explicit_output=False)

    _any = types.Any

    # (array or scalar, array o scalar, out=array)
    lower(ufunc, _any, _any, types.Array)(binary_ufunc)
    # (scalar, scalar)
    lower(ufunc, _any, _any)(binary_ufunc_no_explicit_output)

    _kernels[ufunc] = kernel


def register_unary_operator_kernel(operator, kernel):
    def lower_unary_operator(context, builder, sig, args):
        return numpy_ufunc_kernel(context, builder, sig, args, kernel,
                                  explicit_output=False)
    _arr_kind = types.Array
    lower(operator, _arr_kind)(lower_unary_operator)


def register_binary_operator_kernel(operator, kernel):
    def lower_binary_operator(context, builder, sig, args):
        return numpy_ufunc_kernel(context, builder, sig, args, kernel,
                                  explicit_output=False)

    def lower_inplace_operator(context, builder, sig, args):
        # The visible signature is (A, B) -> A
        # The implementation's signature (with explicit output)
        # is (A, B, A) -> A
        args = args + (args[0],)
        sig = typing.signature(sig.return_type, *sig.args + (sig.args[0],))
        return numpy_ufunc_kernel(context, builder, sig, args, kernel,
                                  explicit_output=True)

    _any = types.Any
    _arr_kind = types.Array
    formal_sigs = [(_arr_kind, _arr_kind), (_any, _arr_kind), (_arr_kind, _any)]
    for sig in formal_sigs:
        lower(operator, *sig)(lower_binary_operator)
        inplace = operator + '='
        if inplace in utils.inplace_map:
            lower(inplace, *sig)(lower_inplace_operator)


################################################################################
# Use the contents of ufunc_db to initialize the supported ufuncs

for ufunc in ufunc_db.get_ufuncs():
    if ufunc.nin == 1:
        register_unary_ufunc_kernel(ufunc, _ufunc_db_function(ufunc))
    elif ufunc.nin == 2:
        register_binary_ufunc_kernel(ufunc, _ufunc_db_function(ufunc))
    else:
        raise RuntimeError("Don't know how to register ufuncs from ufunc_db with arity > 2")


@lower('+', types.Array)
def array_positive_impl(context, builder, sig, args):
    '''Lowering function for +(array) expressions.  Defined here
    (numba.targets.npyimpl) since the remaining array-operator
    lowering functions are also registered in this module.
    '''
    class _UnaryPositiveKernel(_Kernel):
        def generate(self, *args):
            [val] = args
            return val

    return numpy_ufunc_kernel(context, builder, sig, args,
                              _UnaryPositiveKernel, explicit_output=False)


for _op_map in (npydecl.NumpyRulesUnaryArrayOperator._op_map,
                npydecl.NumpyRulesArrayOperator._op_map):
    for operator, ufunc_name in _op_map.items():
        ufunc = getattr(np, ufunc_name)
        kernel = _kernels[ufunc]
        if ufunc.nin == 1:
            register_unary_operator_kernel(operator, kernel)
        elif ufunc.nin == 2:
            register_binary_operator_kernel(operator, kernel)
        else:
            raise RuntimeError("There shouldn't be any non-unary or binary operators")


del _kernels
