"""Provides numba type FunctionType that makes functions as instances
of a first-class function types.
"""
# Author: Pearu Peterson
# Created: December 2019

import types
import numba
from numba import types as nbtypes
from numba.extending import typeof_impl
from numba.extending import models, register_model
from numba.extending import unbox, NativeValue, box
from numba.targets.imputils import lower_constant
from numba.ccallback import CFunc
from numba import cgutils
from llvmlite import ir
# from numba.dispatcher import Dispatcher
from numba.types import (
    FunctionType, FunctionProtoType, numbatype, WrapperAddressProtocol)


@typeof_impl.register(WrapperAddressProtocol)
def typeof_WrapperAddressProtocol(val, c):
    return numbatype(val.signature())


@typeof_impl.register(CFunc)
def typeof_CFunc(val, c):
    return numbatype(val._sig)


# TODO: when enabled, some numba tests fail. What is the reason?
# @typeof_impl.register(types.FunctionType)
def typeof_function(val, c):
    return FunctionType(val)


# @typeof_impl.register(Dispatcher)
def typeof_Dispatcher(val, c):
    return numbatype(val.py_func)


@register_model(FunctionProtoType)
class FunctionProtoModel(models.PrimitiveModel):
    """FunctionProtoModel describes the signatures of first-class functions
    """
    def __init__(self, dmm, fe_type):
        if isinstance(fe_type, FunctionType):
            ftype = fe_type.ftype
        elif isinstance(fe_type, FunctionProtoType):
            ftype = fe_type
        else:
            raise NotImplementedError((type(fe_type)))
        retty = dmm.lookup(ftype.rtype).get_value_type()
        args = [dmm.lookup(t).get_value_type() for t in ftype.atypes]
        be_type = ir.PointerType(ir.FunctionType(retty, args))
        super(FunctionProtoModel, self).__init__(dmm, fe_type, be_type)


@register_model(FunctionType)
class FunctionModel(models.StructModel):
    """FunctionModel holds addresses of function implementations
    """
    def __init__(self, dmm, fe_type):
        members = [
            # address of cfunc-ded function:
            ('addr', nbtypes.voidptr),
            # address of PyObject* referencing the Python function
            # object, currently it is CFunc:
            ('pyaddr', nbtypes.voidptr),
        ]
        super(FunctionModel, self).__init__(dmm, fe_type, members)


@lower_constant(FunctionType)
def lower_constant_function_type(context, builder, typ, pyval):
    # TODO: implement wrapper address protocol
    if isinstance(pyval, CFunc):
        addr = pyval._wrapper_address
        sfunc = cgutils.create_struct_proxy(typ)(context, builder)
        llty = context.get_value_type(nbtypes.voidptr)
        sfunc.addr = builder.inttoptr(ir.Constant(ir.IntType(64), addr), llty)
        # TODO: is incref(pyval) needed? See also related comments in
        # unboxing below.
        sfunc.pyaddr = builder.inttoptr(
            ir.Constant(ir.IntType(64), id(pyval)), llty)
        return sfunc._getvalue()

    if isinstance(pyval, types.FunctionType):
        # TODO: is this used??
        # TODO: make sure pyval matches with typ signature
        cfunc = numba.cfunc(typ.signature())(pyval)
        return lower_constant_function_type(context, builder, typ, cfunc)

    raise NotImplementedError(
        'lower_constant_struct_function_type({}, {}, {}, {})'
        .format(context, builder, typ, pyval))


def _get_wrapper_address(func, sig):
    """Return the address of a compiled function that implements `func` algoritm.

    Warning: The compiled function must be compatible with the given
    signature `sig`. If it is not, then calling the compiled function
    will likely crash the program with segfault.

    Parameters
    ----------
    func : object
      Specify a function object that can be numba.cfunc decorated or
      an object that implements wrapper address protocol (see note below).
    sig : str
      Specify function signature.

    Returns
    -------
    addr : int
      Address (pointer value) of a compiled function.


    Note: wrapper address protocol
    ------------------------------

    A object implements the wrapper address protocol iff the object
    provides a callable attribute named __wrapper_address__ that takes
    one string argument representing the signature, and returns an
    integer representing the address or pointer value of a compiled
    function with given signature.

    """
    if hasattr(func, '__wrapper_address__'):
        # func can be any object that implements the
        # __wrapper_address__ protocol.
        addr = func.__wrapper_address__(sig)
    elif isinstance(func, types.FunctionType):
        cfunc = numba.cfunc(sig)(func)
        addr = cfunc._wrapper_address
    elif isinstance(func, CFunc):
        # TODO: check that func signature matches sif
        addr = func._wrapper_address
    else:
        raise NotImplementedError(
            f'get wrapper address of {type(func)} instance with {sig!r}')
    if not isinstance(addr, int):
        raise TypeError(
            f'wrapper address must be integer, got {type(addr)} instance')
    if addr <= 0:
        raise ValueError(f'wrapper address of {type(func)} instance must be'
                         f' positive integer but got {addr}')
    return addr


@unbox(FunctionType)
def unbox_function_type(typ, obj, c):
    sfunc = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    # Get obj wrapper address. The code below trusts that a function
    # numba.function._get_wrapper_address exists and can be called
    # with two arguments. However, a failure raised in the function
    # will be catched and propagated to the caller.
    modname = c.context.insert_const_string(c.builder.module, 'numba.function')
    numba_mod = c.pyapi.import_module_noblock(modname)
    numba_func = c.pyapi.object_getattr_string(
        numba_mod, '_get_wrapper_address')
    c.pyapi.decref(numba_mod)

    ctyp = c.context.insert_const_string(c.builder.module, typ.name)
    styp = c.pyapi.string_from_string(ctyp)
    addr = c.pyapi.call_function_objargs(numba_func, (obj, styp))
    c.pyapi.decref(numba_func)
    c.pyapi.decref(styp)

    with c.builder.if_else(cgutils.is_null(c.builder, addr),
                           likely=False) as (then, orelse):
        with then:
            # propagate any errors in getting pyaddr to the caller
            c.builder.ret(c.pyapi.get_null_object())
        with orelse:
            # _get_wrapper_address checks that pyaddr is int and
            # nonzero, so no need to check it here. But it will be
            # impossible to tell if the addr value actually
            # corresponds to a memory location of a valid function.
            sfunc.addr = c.pyapi.long_as_voidptr(addr)
            c.pyapi.decref(addr)

            # TODO: the following does not work on 32-bit systems
            # TODO: is incref(obj) needed? where the corresponding
            # decref should be called?
            # TODO: see
            # https://github.com/numba/numba/blob/77cb53ba8966a38bfbd3413559e4f04b12812535/numba/targets/boxing.py#L505
            # as an alternative way for boxing
            llty = c.context.get_value_type(nbtypes.voidptr)
            sfunc.pyaddr = c.builder.ptrtoint(obj, llty)

    return NativeValue(sfunc._getvalue())


@box(FunctionType)
def box_function_type(typ, val, c):
    # print('BOX_function_type({}, {}, {})'.format(typ, val, type(val)))
    sfunc = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)

    # TODO: reconsider the boxing model, see the limitations in the
    # unbox function above.
    pyaddr_ptr = cgutils.alloca_once(c.builder, c.pyapi.pyobj)
    raw_ptr = c.builder.inttoptr(sfunc.pyaddr, c.pyapi.pyobj)
    with c.builder.if_else(cgutils.is_null(c.builder, raw_ptr),
                           likely=False) as (then, orelse):
        with then:
            cstr = f"first-class function {typ} parent object not set"
            c.pyapi.err_set_string("PyExc_MemoryError", cstr)
            c.builder.ret(c.pyapi.get_null_object())
        with orelse:
            pass
    c.builder.store(raw_ptr, pyaddr_ptr)
    cfunc = c.builder.load(pyaddr_ptr)
    c.pyapi.incref(cfunc)
    return cfunc
