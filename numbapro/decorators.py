import numba
from numbapro.cudadrv.initialize import last_error


def autojit(*args, **kwds):
    """Creates a type specialized function that lazily compiles to native code
    at the first invocation for any function type.  Each function type is
    only compiled once.
    
    NumbaPro adds "gpu" target to ``numba.autojit`` to target CUDA GPUs.
    
    Please see docs for ``numbapro.cuda.autojit`` for details.

    Refer to http://docs.continuum.io/numbapro/quickstart.html for usage
    """
    return numba.autojit(*args, **kwds)


def jit(*args, **kwds):
    """Compile a function given the parameter and return types.
    
    NumbaPro adds CUDA GPU support to ``numba.jit``.
    
    .. py:function:: jit(signature[, target="cpu"])
    
    :param signature: Specifies the function type.  A function type
                      can be created by calling a Numba type: 
                      ``int32(int32, float32)`` creates a function type that
                      takes a 32-bit integer and a single-precision float,
                      and returns a 32-bit integer.
                    
                      The argument can also be specified as a string.  
                      For example: ``jit("int32(int32, float32)")``
    
    :param target: Specifies the hardware target for the generated code.
                   The default target is "cpu". NumbaPro adds the "gpu" target 
                   for compiling for execution on a CUDA GPU.
    
    Please see docs for ``numbapro.cuda.jit`` for details about additional usage
    for the GPU target.

    Refer to http://docs.continuum.io/numbapro/quickstart.html for usage
    """
    return numba.jit(*args, **kwds)
