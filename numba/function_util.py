# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

def external_call(context, llvm_module, name, args=(), temp_name=None):
    extfn = context.external_library.get(name)
    return external_call_func(context, llvm_module, extfn, args, temp_name)

def utility_call(context, llvm_module, name, args=(), temp_name=None):
    extfn = context.utility_library.get(name)
    return external_call_func(context, llvm_module, extfn, args, temp_name)

def external_call_func(context, llvm_module, extfn, args=(), temp_name=None):
    '''Build a call node to the specified external function.

    context --- A numba context
    llvm_module --- A LLVM llvm_module
    name --- Name of the external function
    args --- [optional] argument of for the call
    temp_name --- [optional] Name of the temporary value in LLVM IR.
    '''
    from numba import nodes
    temp_name = temp_name or extfn.name
    assert temp_name is not None

    sig = extfn.signature
    lfunc = extfn.declare_lfunc(context, llvm_module)

    exc_check = dict(badval   = extfn.badval,
                     goodval  = extfn.goodval,
                     exc_msg  = extfn.exc_msg,
                     exc_type = extfn.exc_type,
                     exc_args = extfn.exc_args)

    result = nodes.NativeCallNode(sig, args, lfunc, name=temp_name, **exc_check)

    if extfn.check_pyerr_occurred:
        result = nodes.PyErr_OccurredNode(result)

    return result
