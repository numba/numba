import logging
import inspect

from ndtypes import ndt
from xnd import xnd
from gumath import unsafe_add_kernel
from llvmlite import ir
from llvmlite.ir import PointerType as ptr, LiteralStructType as struct

from .. import jit
from .. import types as numba_types
from .llvm import build_kernel_wrapper

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


def jit_xnd(ndt_sig_or_fn=None, **kwargs):
    if callable(ndt_sig_or_fn):
        return GumathDispatcher(None, kwargs, ndt_sig_or_fn)
    return lambda fn: GumathDispatcher(ndt_sig_or_fn, kwargs, fn)


class GumathDispatcher:
    """
    Add gumath kernels based on Python functions.
    """
    i = 0

    def __init__(self, ndt_sig, kwargs, fn):
        if not ndt_sig:
            nargs = len(inspect.signature(fn).parameters)
            ndt_sig = ', '.join(['... * D'] * nargs) + ' -> ... * D'
        logger.info('Creating dispatcher for %s', ndt_sig)
        self.cache = {}
        self.ndt_sig = ndt_sig
        self.dimensions = list(dimensions_from_ndt(ndt_sig))
        logger.info('Dimensions: %s', self.dimensions)
        self.returns_scalar = self.dimensions[-1] == 0
        logger.info('Returns scalar: %s', self.returns_scalar)
        logger.info('Jitting with kwargs: %s', kwargs)
        self.dispatcher =  jit(**kwargs)(fn)
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__

        self._install_type(self)

    def __call__(self, *args):
        xnd_args = [a if isinstance(a, xnd) else xnd.from_buffer(a) for a in args]
        logger.info('Calling dispatcher with %s', xnd_args)
        dtypes = tuple(str(a.type.hidden_dtype) for a in xnd_args)
        logger.info('Dtypes: %s', dtypes)
        kernel, cres = self.get_or_create_kernel(dtypes)
        return kernel(*xnd_args)

    def get_or_create_kernel(self, dtypes):
        if dtypes in self.cache:
            logger.info('Found existing kernel')
            return self.cache[dtypes]
        logger.info('Creating new kernel')
        numba_sig = self.generate_numba_sig(dtypes)
        logger.info('Numba sig: %s', numba_sig)

        # numba gives us back the function, but we want the compile result
        # so we search for it
        entry_point = self.dispatcher.compile(numba_sig)
        cres = [cres for cres in self.dispatcher.overloads.values() if cres.entry_point == entry_point][0]

        if cres.objectmode:
            raise NotImplementedError('Creating gumath kernel in object mode not supported.')
        name = f'numba.{GumathDispatcher.i}'
        logger.info('Gumath kernel name: %s', name)
        # gumath kernel name needs to be unique
        kernel = unsafe_add_kernel(
            name=name,
            sig=self.ndt_sig,
            ptr=build_kernel_wrapper(cres, self.dimensions),
            tag='Xnd'
        )
        GumathDispatcher.i += 1
        logger.info('Storing kernel in cache with dtypes: %s', dtypes)
        self.cache[dtypes] = (kernel, cres)
        return (kernel, cres)

    def generate_numba_sig(self, dtypes):
        if not self.returns_scalar:
            dtypes = list(dtypes) + [infer_return_dtype(self.ndt_sig, dtypes)]
    
        return tuple(numba_argument(dtype, ndim) for dtype, ndim in zip(dtypes, self.dimensions))

    def _install_type(self):
        _ty_cls = type('GumathTyping_' + self.ufunc.__name__,
                       (AbstractTemplate,),
                       dict(key=self, generic=self._type_me))
        self.dispatchertargetdescr.typing_context.typingctx.insert_user_function(self, _ty_cls)
def ndt_fn_to_dims(ndt_sig):
    """
    Returns the inputs and return value of the ndt signature, split by dimension

        >>> ndt_fn_to_dims("... * float64, ... * D -> D")
        (('... float64', '... D'), ('D'))
    """
    for args in ndt_sig.split(" -> "):
        yield [arg for arg in args.split(", ")]


def dimensions_from_ndt(ndt_sig):
    inputs, returns = ndt_fn_to_dims(ndt_sig)
    if len(returns) != 1:
        raise NotImplementedError("Only supports one return vale in gumath signature")
    for arg in inputs + returns:
        yield len([None for dim in arg.split(' * ') if '...' not in dim]) - 1


def numba_argument(ndt_dtype, ndim):
    numba_type = getattr(numba_types, str(ndt_dtype))
    if ndim == 0:
        return numba_type
    return numba_types.npytypes.Array(numba_type, ndim, 'A')

def infer_return_dtype(ndt_sig, input_dtypes):
    """
    Determines the return dtype based on the input dtypes.

        >>> infer_return_dtype('... * D, ... * K -> ... * D', ('float64', 'int32'))
        float64
    """
    inputs, returns = ndt_fn_to_dims(ndt_sig)
    *input_types, return_type = [ndt(arg).hidden_dtype for arg in inputs + returns]
    if return_type.isconcrete():
        return str(return_type)


    for i, input_type in enumerate(input_types):
        if input_type == return_type:
            return input_dtypes[i]
    
    raise NotImplementedError(f'Cannot infer return dtype for {ndt_sig} based on inputs {input_dtypes}')
