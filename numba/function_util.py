
def external_call(context, module, name, args=(), temp_name=None):
    '''Build a call node to the specified external function.

    context --- A numba context
    module --- A LLVM module
    name --- Name of the external function
    args --- [optional] argument of for the call
    temp_name --- [optional] Name of the temporary value in LLVM IR.
    '''
    from numba import nodes
    temp_name = temp_name or name
    extfn = context.external_library.get(name)

    sig = extfn.signature
    lfunc_type = sig.to_llvm(context)

    lfunc = module.get_or_insert_function(lfunc_type, name=name)
    
    exc_check = dict(badval   = extfn.badval,
                     goodval  = extfn.goodval,
                     exc_msg  = extfn.exc_msg,
                     exc_type = extfn.exc_type,
                     exc_args = extfn.exc_args)

    result = nodes.NativeCallNode(sig, args, lfunc, name=temp_name, **exc_check)
    return result
