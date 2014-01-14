from __future__ import print_function, division, absolute_import
import numpy
from numba import _dispatcher, types, compiler, targets, typing


class GlobalContext(object):
    """
    Singleton object
    """
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            inst = object.__new__(cls)
            inst._init()
            cls.__instance = inst
        return cls.__instance

    def _init(self):
        self.target_context = targets.CPUContext()
        self.typing_context = typing.Context()


# TODO
# The dispatcher to use python type object to determine which version to
# call and use numpy.dtype for ndarray.
# int default to int32
# long default to pyobject?
class Overloaded(object):
    def __init__(self, py_func):
        self.dispatcher = _dispatcher.Dispatcher()
        self.py_func = py_func
        self.overloads = {}

    def add_overload(self, cres):
        self.dispatcher.insert(cres.argtypes, cres.entry_point_addr)
        self.overloads[cres.argtypes] = cres

    def jit(self, sig, **kws):
        flags = compiler.Flags()
        read_flags(flags, kws)

        glctx = GlobalContext()
        typingctx = glctx.typing_context
        targetctx = glctx.target_context

        if isinstance(sig, types.Prototype):
            args = sig.args
            return_type = sig.return_type
        else:
            args = sig
            return_type = None

        cres = compiler.compile_extra(typingctx, targetctx, self.py_func,
                                      args=args, return_type=return_type,
                                      flags=flags)

        # Check typing error if nopython mode is used
        if cres.typing_error is not None and not flags.enable_pyobject:
            raise cres.typing_error

        self.add_overload(cres)

    def __call__(self, *args, **kws):
        assert not kws, "Keyword arguments are not supported"
        tys = [None] * len(args)
        for i, a in enumerate(args):
            tys[i] = typeof_pyval(a)

        return self.dispatcher(tuple(tys), args)


def _jit(args, return_type, flags):
    glctx =  GlobalContext()
    typingctx = glctx.typing_context
    targetctx = glctx.target_context

    def wrapper(func):
        disp = dispatcher.Overloaded(py_func=func)

        cres = compiler.compile_extra(typingctx, targetctx, func, args=args,
                                      return_type=return_type, flags=flags)

        # Check typing error if nopython mode is used
        if cres.typing_error is not None and not flags.enable_pyobject:
            raise cres.typing_error

        disp.add_overload(cres)
        return disp

    return wrapper


def read_flags(flags, kws):
    if kws.pop('nopython', False) == False:
        flags.set("enable_pyobject")

    if kws.pop("forceobj", False) == True:
        flags.set("force_pyobject")

    if kws:
        # Unread options?
        raise NameError("Unrecognized options: %s" % k.keys())



DTYPE_MAPPING = {}


def typeof_pyval(val):
    # TODO make this faster
    if isinstance(val, (int, long)):
        return types.int32

    elif isinstance(val, float):
        return types.float64

    elif isinstance(val, numpy.ndarray):
        # TODO complete dtype mapping
        dtype = FROM_DTYPE[val.dtype]
        ndim = val.ndim
        layout = 'A'
        aryty = types.Array(dtype, ndim, layout)
        return aryty

    else:
        raise TypeError(type(val), val)



FROM_DTYPE = {
    numpy.dtype('int8'): types.int8,
    numpy.dtype('int16'): types.int16,
    numpy.dtype('int32'): types.int32,
    numpy.dtype('int64'): types.int64,

    numpy.dtype('uint8'): types.uint8,
    numpy.dtype('uint16'): types.uint16,
    numpy.dtype('uint32'): types.uint32,
    numpy.dtype('uint64'): types.uint64,

    numpy.dtype('float32'): types.float32,
    numpy.dtype('float64'): types.float64,

    # numpy.dtype('complex64'): types.complex64,
    # numpy.dtype('complex128'): types.complex128,

}

