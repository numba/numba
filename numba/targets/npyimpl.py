"""
Implementation of functions in the Numpy package.
"""

from __future__ import print_function, division, absolute_import

import numpy
import math
import sys
import itertools
from collections import namedtuple

from llvmlite.llvmpy import core as lc

from . import builtins, ufunc_db
from .imputils import implement, Registry
from .. import typing, types, cgutils, numpy_support
from ..config import PYVERSION
from ..numpy_support import ufunc_find_matching_loop

registry = Registry()
register = registry.register


def caster(restype, constructor):
    """
    Implement explicit calls to Numpy type *constructor* (e.g. np.int16).
    """
    @implement(constructor, types.Any)
    def _cast(context, builder, sig, args):
        [val] = args
        [valty] = sig.args
        return context.cast(builder, val, valty, restype)

    return _cast

for tp in types.number_domain:
    register(caster(tp, getattr(numpy, str(tp))))

register(caster(types.intc, numpy.intc))
register(caster(types.uintc, numpy.uintc))
register(caster(types.intp, numpy.intp))
register(caster(types.uintp, numpy.uintp))

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
        return self.builder.load(self._load_effective_address(indices))

    def store_data(self, indices, value):
        ctx = self.context
        bld = self.builder

        # maybe there should be a "get_memory_value" or "get_storage_value"
        store_value = ctx.get_struct_member_value(bld, self.base_type, value)
        assert ctx.get_data_type(self.base_type) == store_value.type

        bld.store(store_value, self._load_effective_address(indices))


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
    elif tyinp in types.number_domain | set([types.boolean]):
        return _ScalarHelper(ctxt, bld, inp, tyinp)
    else:
        raise TypeError('unknown type for {0}: {1}'.format(where, str(tyinp)))


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
    if not explicit_output:
        args.append(lc.Constant.null(context.get_value_type(sig.return_type)))
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
        if fromty in types.complex_domain and toty not in types.complex_domain:
            # attempt conversion of the real part to the specified type.
            # note that NumPy issues a warning in this kind of conversions
            newty = fromty.underlying_float
            attr = self.context.get_attribute(val, fromty, 'real')
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
            res = self.fn(self.context, self.builder, isig, cast_args)
            return self.cast(res, isig.return_type, osig.return_type)

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
# Use the contents of ufunc_db to initialize the supported ufuncs

for ufunc in ufunc_db.get_ufuncs():
    if ufunc.nin == 1:
        register_unary_ufunc_kernel(ufunc, _ufunc_db_function(ufunc))
    elif ufunc.nin == 2:
        register_binary_ufunc_kernel(ufunc, _ufunc_db_function(ufunc))
    else:
        raise RuntimeError("Don't know how to register ufuncs from ufunc_db with arity > 2")

