from __future__ import print_function, division, absolute_import
import inspect
import warnings
import numpy
from numba.config import PYVERSION
from numba import _dispatcher, compiler, typing, utils
from numba.typeconv.rules import default_type_manager
from numba.typing.templates import resolve_overload
from numba import types, sigutils
from numba.targets import cpu


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
        self.typing_context = typing.Context()
        self.target_context = cpu.CPUContext(self.typing_context)


class Overloaded(_dispatcher.Dispatcher):
    """
    Abstract class. Subclass should define targetdescr class attribute.
    """
    def __init__(self, py_func, targetoptions={}):
        self.tm = default_type_manager

        argspec = inspect.getargspec(py_func)
        argct = len(argspec.args)

        super(Overloaded, self).__init__(self.tm.get_pointer(), argct)

        self.py_func = py_func
        self.overloads = {}

        self.targetoptions = targetoptions
        self.doc = py_func.__doc__

    def disable_compile(self, val=True):
        """Disable the compilation of new signatures at call time.
        """
        self._disable_compile(int(val))

    def add_overload(self, cres):
        sig = [a._code for a in cres.signature.args]
        self._insert(sig, cres.entry_point_addr)
        self.overloads[cres.signature] = cres

        # Add native function for correct typing the code generation
        typing = cres.typing_context
        target = cres.target_context
        cfunc = cres.entry_point
        if cfunc in target.native_funcs:
            target.dynamic_map_function(cfunc)
            calltemplate = target.get_user_function(cfunc)
            typing.insert_user_function(cfunc, calltemplate)
            typing.extend_user_function(self, calltemplate)

    def get_overload(self, *tys):
        return self.overloads[tys].entry_point

    def compile(self, sig, **targetoptions):
        topt = self.targetoptions.copy()
        topt.update(targetoptions)

        flags = compiler.Flags()
        self.targetdescr.options.parse_as_flags(flags, topt)

        glctx = GlobalContext()
        typingctx = glctx.typing_context
        targetctx = glctx.target_context

        args, return_type = sigutils.normalize_signature(sig)

        cres = compiler.compile_extra(typingctx, targetctx, self.py_func,
                                      args=args, return_type=return_type,
                                      flags=flags)

        # Check typing error if object mode is used
        if cres.typing_error is not None and not flags.enable_pyobject:
            raise cres.typing_error

        self.add_overload(cres)
        return cres.entry_point

    def jit(self, sig, **kws):
        """Alias of compile(sig, **kws)
        """
        return self.compile(sig, **kws)

    def _compile_and_call(self, *args, **kws):
        assert not kws
        sig = tuple([typeof_pyval(a) for a in args])
        self.jit(sig)
        return self(*args, **kws)

    def inspect_types(self):
        for ver, res in utils.dict_iteritems(self.overloads):
            print("%s %s" % (self.py_func.__name__, ver))
            print('-' * 80)
            print(res.type_annotation)
            print('=' * 80)

    # def __call__(self, *args, **kws):
    #     assert not kws, "Keyword arguments are not supported"
    #     tys = []
    #     for i, a in enumerate(args):
    #         tys.append(typeof_pyval(a))
    #
    #     sig = [self.tm.get(t) for t in tys]
    #     ptr = self.find(sig)
    #     return super(Overloaded, self).__call__(ptr, args)

    def _explain_ambiguous(self, *args, **kws):
        assert not kws, "kwargs not handled"
        args = tuple([typeof_pyval(a) for a in args])
        resolve_overload(GlobalContext().typing_context, self.py_func,
                         tuple(self.overloads.keys()), args, kws)


#
#
# class NPMOverloaded(Overloaded):
#     def compile(self, sig, **kws):
#         flags = compiler.Flags()
#         read_flags(flags, kws)
#         if flags.enable_pyobject or flags.force_pyobject:
#             raise TypeError("Object mode enabled for nopython target")
#
#         glctx = GlobalContext()
#         typingctx = glctx.typing_context
#         targetctx = glctx.target_context
#
#         args, return_type = sigutils.normalize_signature(sig)
#
#         cres = compiler.compile_extra(typingctx, targetctx, self.py_func,
#                                       args=args, return_type=return_type,
#                                       flags=flags)
#
#         # Check typing error if object mode is used
#         if cres.typing_error is not None and not flags.enable_pyobject:
#             raise cres.typing_error
#
#         self.add_overload(cres)
#         return cres.entry_point


DTYPE_MAPPING = {}


INT_TYPES = (int,)
if PYVERSION < (3, 0):
    INT_TYPES += (long,)


def typeof_pyval(val):
    """
    This is called from numba._dispatcher as a fallback if the native code
    cannot decide the type.
    """
    if isinstance(val, numpy.ndarray):
        # TODO complete dtype mapping
        dtype = FROM_DTYPE[val.dtype]
        ndim = val.ndim
        if ndim == 0:
            # is array scalar
            return FROM_DTYPE[val.dtype]

        if val.flags['C_CONTIGUOUS']:
            layout = 'C'
        elif val.flags['F_CONTIGUOUS']:
            layout = 'F'
        else:
            layout = 'A'
        aryty = types.Array(dtype, ndim, layout)
        return aryty

    # The following are handled in the C version for exact type match
    # So test these later
    elif isinstance(val, INT_TYPES):
        return types.int32

    elif isinstance(val, float):
        return types.float64

    elif isinstance(val, complex):
        return types.complex128

    elif numpy.dtype(type(val)) in FROM_DTYPE:
        # Array scalar
        return FROM_DTYPE[numpy.dtype(type(val))]

    # Other object
    else:
        return types.pyobject



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

    numpy.dtype('complex64'): types.complex64,
    numpy.dtype('complex128'): types.complex128,
}


# Initialize dispatcher
_dispatcher.init_types(dict((str(t), t._code) for t in types.number_domain))
