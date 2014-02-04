from __future__ import print_function, division, absolute_import
import inspect
import contextlib
import numpy
from numba.config import PYVERSION
from numba import _dispatcher, compiler, utils
from numba.typeconv.rules import default_type_manager
from numba.typing.templates import resolve_overload
from numba import types, sigutils
from numba import numpy_support
from numba.bytecode import get_code_object


class Overloaded(_dispatcher.Dispatcher):
    """
    Abstract class. Subclass should define targetdescr class attribute.
    """
    __numba__ = compile

    def __init__(self, py_func, locals={}, targetoptions={}):
        self.tm = default_type_manager

        argspec = inspect.getargspec(py_func)
        argct = len(argspec.args)

        super(Overloaded, self).__init__(self.tm.get_pointer(), argct)

        self.py_func = py_func
        self.func_code = get_code_object(py_func)
        self.overloads = {}
        self.fallback = None

        self.targetoptions = targetoptions
        self.locals = locals
        self.doc = py_func.__doc__
        self._compiling = False

        self.targetdescr.typing_context.insert_overloaded(self)

    @property
    def signatures(self):
        """
        Returns a list of compiled function signatures.
        """
        return list(self.overloads.keys())

    def disable_compile(self, val=True):
        """Disable the compilation of new signatures at call time.
        """
        self._disable_compile(int(val))

    def add_overload(self, cres):
        sig = [a._code for a in cres.signature.args]
        self._insert(sig, cres.entry_point_addr, cres.objectmode)
        if cres.objectmode:
            self.fallback = cres.entry_point
        self.overloads[cres.signature] = cres

        # Add native function for correct typing the code generation
        typing = cres.typing_context
        target = cres.target_context
        cfunc = cres.entry_point
        if cfunc in target.native_funcs:
            target.dynamic_map_function(cfunc)
            calltemplate = target.get_user_function(cfunc)
            typing.insert_user_function(cfunc, calltemplate)

    def get_overload(self, sig):
        args, return_type = sigutils.normalize_signature(sig)
        return self.overloads[tuple(args)].entry_point

    @contextlib.contextmanager
    def _compile_lock(self):
        if self._compiling:
            raise RuntimeError("Compiler re-entrant")
        self._compiling = True
        yield
        self._compiling = False

    @property
    def is_compiling(self):
        return self._compiling

    def compile(self, sig, locals={}, **targetoptions):
        with self._compile_lock():
            locs = self.locals.copy()
            locs.update(locals)

            topt = self.targetoptions.copy()
            topt.update(targetoptions)

            flags = compiler.Flags()
            self.targetdescr.options.parse_as_flags(flags, topt)

            glctx = self.targetdescr
            typingctx = glctx.typing_context
            targetctx = glctx.target_context


            args, return_type = sigutils.normalize_signature(sig)

            # Don't recompile if signature already exist.
            existing = self.overloads.get(tuple(args))
            if existing is not None:
                return existing.entry_point

            cres = compiler.compile_extra(typingctx, targetctx, self.py_func,
                                          args=args, return_type=return_type,
                                          flags=flags, locals=locs)

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

    def _explain_ambiguous(self, *args, **kws):
        assert not kws, "kwargs not handled"
        args = tuple([typeof_pyval(a) for a in args])
        resolve_overload(self.targetdescr.typing_context, self.py_func,
                         tuple(self.overloads.keys()), args, kws)

    def __repr__(self):
        return "%s(%s)" % (type(self).__name__, self.py_func)



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
        dtype = numpy_support.from_dtype(val.dtype)
        ndim = val.ndim
        if ndim == 0:
            # is array scalar
            return numpy_support.from_dtype(val.dtype)
        layout = numpy_support.map_layout(val)
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

    elif numpy_support.is_arrayscalar(val):
        # Array scalar
        return numpy_support.from_dtype(numpy.dtype(type(val)))

    # Other object
    else:
        return types.pyobject


# Initialize dispatcher
_dispatcher.init_types(dict((str(t), t._code) for t in types.number_domain))
