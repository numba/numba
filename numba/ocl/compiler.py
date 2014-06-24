from __future__ import print_function, absolute_import
from numba import types, compiler


def compile_ocl(pyfunc, return_type, args, debug):
    # First compilation will trigger the initialization of the CUDA backend.
    from .descriptor import OCLTargetDesc

    typingctx = OCLTargetDesc.typingctx
    targetctx = OCLTargetDesc.targetctx
    # TODO handle debug flag
    flags = compiler.Flags()
    # Do not compile (generate native code), just lower (to LLVM)
    flags.set('no_compile')
    # Run compilation pipeline
    cres = compiler.compile_extra(typingctx=typingctx,
                                  targetctx=targetctx,
                                  func=pyfunc,
                                  args=args,
                                  return_type=return_type,
                                  flags=flags,
                                  locals={})

    # Fix global naming
    for gv in cres.llvm_module.global_variables:
        if '.' in gv.name:
            gv.name = gv.name.replace('.', '_')

    return cres


def compile_kernel(pyfunc, args, debug=False):
    cres = compile_ocl(pyfunc, types.void, args, debug=debug)
    kernel = cres.target_context.prepare_ocl_kernel(cres.llvm_func,
                                                    cres.signature.args)
    cres = cres._replace(llvm_func=kernel)
    return cres
