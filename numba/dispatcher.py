from __future__ import print_function, division, absolute_import
import inspect
from collections import namedtuple
import contextlib
import numpy
import functools

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
    __numba__ = "py_func"

    def __init__(self, py_func, locals={}, targetoptions={}):
        """
        Parameters
        ----------
        py_func: function object to be compiled
        locals: dict, optional
            Mapping of local variable names to Numba types.  Used to override
            the types deduced by the type inference engine.
        targetoptions: dict, optional
            Target-specific config options.
        """
        self.tm = default_type_manager

        argspec = inspect.getargspec(py_func)
        argct = len(argspec.args)

        super(Overloaded, self).__init__(self.tm.get_pointer(), argct)

        self.py_func = py_func
        functools.update_wrapper(self, py_func)

        # other parts of Numba assume the old Python 2 name for code object
        self.func_code = get_code_object(py_func)
        # but newer python uses a different name
        self.__code__ = self.func_code

        # A mapping of signatures to entry points
        self.overloads = {}

        # A mapping of signatures to compile results
        self._compileinfos = {}

        self.targetoptions = targetoptions
        self.locals = locals
        self._compiling = False

        self.targetdescr.typing_context.insert_overloaded(self)

    @property
    def signatures(self):
        """
        Returns a list of compiled function signatures.
        """
        return list(self.overloads)

    def disable_compile(self, val=True):
        """Disable the compilation of new signatures at call time.
        """
        self._disable_compile(int(val))

    def add_overload(self, cres):
        args = tuple(cres.signature.args)
        sig = [a._code for a in args]
        self._insert(sig, cres.entry_point, cres.objectmode)
        self.overloads[args] = cres.entry_point
        self._compileinfos[args] = cres

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
        return self.overloads[tuple(args)]

    @contextlib.contextmanager
    def _compile_lock(self):
        if self._compiling:
            raise RuntimeError("Compiler re-entrant")
        self._compiling = True
        try:
            yield
        finally:
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
                return existing

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

    def _compile_for_args(self, *args, **kws):
        """
        For internal use.  Compile a specialized version of the function
        for the given *args* and *kws*, and return the resulting callable.
        """
        assert not kws
        sig = tuple([self.typeof_pyval(a) for a in args])
        return self.jit(sig)

    def inspect_types(self):
        for ver, res in utils.dict_iteritems(self._compileinfos):
            print("%s %s" % (self.py_func.__name__, ver))
            print('-' * 80)
            print(res.type_annotation)
            print('=' * 80)

    def _explain_ambiguous(self, *args, **kws):
        assert not kws, "kwargs not handled"
        args = tuple([self.typeof_pyval(a) for a in args])
        resolve_overload(self.targetdescr.typing_context, self.py_func,
                         tuple(self.overloads.keys()), args, kws)

    def __repr__(self):
        return "%s(%s)" % (type(self).__name__, self.py_func)

    @classmethod
    def typeof_pyval(cls, val):
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
                return dtype
            layout = numpy_support.map_layout(val)
            aryty = types.Array(dtype, ndim, layout)
            return aryty

        elif isinstance(val, numpy.record):
            return numpy_support.from_dtype(val.dtype)

        # The following are handled in the C version for exact type match
        # So test these later
        elif isinstance(val, utils.INT_TYPES):
            return types.int64

        elif isinstance(val, float):
            return types.float64

        elif isinstance(val, complex):
            return types.complex128

        elif numpy_support.is_arrayscalar(val):
            # Array scalar
            return numpy_support.from_dtype(numpy.dtype(type(val)))

        # Other object
        else:
            return getattr(val, "_numba_type_", types.pyobject)


class LiftedLoop(Overloaded):
    def __init__(self, bytecode, typingctx, targetctx, locals, flags):
        self.tm = default_type_manager

        argspec = bytecode.argspec
        argct = len(argspec.args)

        _dispatcher.Dispatcher.__init__(self, self.tm.get_pointer(), argct)

        self.bytecode = bytecode
        self.typingctx = typingctx
        self.targetctx = targetctx
        self.locals = locals
        self.flags = flags

        self.py_func = bytecode.func
        self.overloads = {}
        self._compileinfos = {}

        self.doc = self.py_func.__doc__
        self._compiling = False

    def compile(self, sig):
        with self._compile_lock():
            # FIXME this is mostly duplicated from Overloaded
            flags = self.flags
            args, return_type = sigutils.normalize_signature(sig)

            # Don't recompile if signature already exist.
            existing = self.overloads.get(tuple(args))
            if existing is not None:
                return existing.entry_point

            assert not flags.enable_looplift, "Enable looplift flags is on"
            cres = compiler.compile_bytecode(typingctx=self.typingctx,
                                             targetctx=self.targetctx,
                                             bc=self.bytecode,
                                             args=args,
                                             return_type=return_type,
                                             flags=flags,
                                             locals=self.locals)

            # Check typing error if object mode is used
            if cres.typing_error is not None and not flags.enable_pyobject:
                raise cres.typing_error

            self.add_overload(cres)
            return cres.entry_point


# Initialize dispatcher
_dispatcher.init_types(dict((str(t), t._code) for t in types.number_domain))
