from __future__ import print_function, division, absolute_import

import contextlib
import functools
import inspect
import sys

from numba import _dispatcher, compiler, utils
from numba.typeconv.rules import default_type_manager
from numba import sigutils, serialize, types, typing
from numba.typing.templates import resolve_overload
from numba.bytecode import get_code_object
from numba.six import create_bound_method, next


class _OverloadedBase(_dispatcher.Dispatcher):
    """
    Common base class for dispatcher Implementations.
    """

    __numba__ = "py_func"

    def __init__(self, arg_count, py_func):
        self._tm = default_type_manager
        #_dispatcher.Dispatcher.__init__(self, self._tm.get_pointer(), arg_count)

        # A mapping of signatures to entry points
        self.overloads = {}
        # A mapping of signatures to compile results
        self._compileinfos = {}
        # A list of nopython signatures
        self._npsigs = []

        self.py_func = py_func
        # other parts of Numba assume the old Python 2 name for code object
        self.func_code = get_code_object(py_func)
        # but newer python uses a different name
        self.__code__ = self.func_code

        self._pysig = utils.pysignature(self.py_func)
        _argnames = tuple(self._pysig.parameters)
        _dispatcher.Dispatcher.__init__(self, self._tm.get_pointer(),
                                        arg_count, _argnames)

        self.doc = py_func.__doc__
        self._compile_lock = utils.NonReentrantLock()

        utils.finalize(self, self._make_finalizer())

    def _reset_overloads(self):
        self._clear()
        self.overloads.clear()
        self._compileinfos.clear()
        self._npsigs[:] = []

    def _make_finalizer(self):
        """
        Return a finalizer function that will release references to
        related compiled functions.
        """
        overloads = self.overloads
        targetctx = self.targetctx
        # Early-bind utils.shutting_down() into the function's local namespace
        # (see issue #689)
        def finalizer(shutting_down=utils.shutting_down):
            # The finalizer may crash at shutdown, skip it (resources
            # will be cleared by the process exiting, anyway).
            if shutting_down():
                return
            # This function must *not* hold any reference to self:
            # we take care to bind the necessary objects in the closure.
            for func in overloads.values():
                try:
                    targetctx.remove_user_function(func)
                    targetctx.remove_native_function(func)
                except KeyError as e:
                    # Not a native function (object mode presumably)
                    pass

        return finalizer

    @property
    def signatures(self):
        """
        Returns a list of compiled function signatures.
        """
        return list(self.overloads)

    @property
    def nopython_signatures(self):
        return self._npsigs

    def disable_compile(self, val=True):
        """Disable the compilation of new signatures at call time.
        """
        # If disabling compilation then there must be at least one signature
        assert val or len(self.signatures) > 0
        self._can_compile = not val

    def add_overload(self, cres):
        args = tuple(cres.signature.args)
        sig = [a._code for a in args]
        self._insert(sig, cres.entry_point, cres.objectmode, cres.interpmode)
        self.overloads[args] = cres.entry_point
        self._compileinfos[args] = cres

        # Add native function for correct typing the code generation
        if not cres.objectmode and not cres.interpmode:
            self._npsigs.append(cres.signature)

    def get_call_template(self, args, kws):
        """
        Get a typing.ConcreteTemplate for this dispatcher and the given *args*
        and *kws*.  This allows to resolve the return type.
        """
        # Fold keyword arguments
        if kws:
            ba = self._pysig.bind(*args, **kws)
            if ba.kwargs:
                # There's a remaining keyword argument, e.g. if omitting
                # some argument with a default value before it.
                raise NotImplementedError("unhandled keyword argument: %s"
                                          % list(ba.kwargs))
            args = ba.args
            kws = {}
        # Ensure an overload is available, but avoid compiler re-entrance
        if self._can_compile and not self.is_compiling:
            self.compile(tuple(args))

        # Create function type for typing
        func_name = self.py_func.__name__
        name = "CallTemplate({0})".format(func_name)
        # The `key` isn't really used except for diagnosis here,
        # so avoid keeping a reference to `cfunc`.
        call_template = typing.make_concrete_template(
            name, key=func_name, signatures=self.nopython_signatures)
        return call_template, args, kws

    def get_overload(self, sig):
        args, return_type = sigutils.normalize_signature(sig)
        return self.overloads[tuple(args)]

    @property
    def is_compiling(self):
        """
        Whether a specialization is currently being compiled.
        """
        return self._compile_lock.is_owned()

    def _compile_for_args(self, *args, **kws):
        """
        For internal use.  Compile a specialized version of the function
        for the given *args* and *kws*, and return the resulting callable.
        """
        assert not kws
        sig = tuple([self.typeof_pyval(a) for a in args])
        return self.compile(sig)

    def inspect_types(self, file=None):
        if file is None:
            file = sys.stdout

        for ver, res in utils.iteritems(self._compileinfos):
            print("%s %s" % (self.py_func.__name__, ver), file=file)
            print('-' * 80, file=file)
            print(res.type_annotation, file=file)
            print('=' * 80, file=file)

    def _explain_ambiguous(self, *args, **kws):
        assert not kws, "kwargs not handled"
        args = tuple([self.typeof_pyval(a) for a in args])
        sigs = [cr.signature for cr in self._compileinfos.values()]
        resolve_overload(self.typingctx, self.py_func, sigs, args, kws)

    def __repr__(self):
        return "%s(%s)" % (type(self).__name__, self.py_func)

    def typeof_pyval(self, val):
        """
        Resolve the Numba type of Python value *val*.
        This is called from numba._dispatcher as a fallback if the native code
        cannot decide the type.
        """
        if isinstance(val, utils.INT_TYPES):
            # Ensure no autoscaling of integer type, to match the
            # typecode() function in _dispatcher.c.
            return types.int64

        tp = self.typingctx.resolve_data_type(val)
        if tp is None:
            tp = types.pyobject
        return tp


class Overloaded(_OverloadedBase):
    """
    Implementation of user-facing dispatcher objects (i.e. created using
    the @jit decorator).
    This is an abstract base class. Subclasses should define the targetdescr
    class attribute.
    """

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
        self.typingctx = self.targetdescr.typing_context
        self.targetctx = self.targetdescr.target_context

        argspec = inspect.getargspec(py_func)
        argct = len(argspec.args)

        _OverloadedBase.__init__(self, argct, py_func)

        functools.update_wrapper(self, py_func)

        self.targetoptions = targetoptions
        self.locals = locals

        self.typingctx.insert_overloaded(self)

    def __get__(self, obj, objtype=None):
        '''Allow a JIT function to be bound as a method to an object'''
        if obj is None:  # Unbound method
            return self
        else:  # Bound method
            return create_bound_method(self, obj)

    def __reduce__(self):
        """
        Reduce the instance for pickling.  This will serialize
        the original function as well the compilation options and
        compiled signatures, but not the compiled code itself.
        """
        if self._can_compile:
            sigs = []
        else:
            sigs = [cr.signature for cr in self._compileinfos.values()]
        return (serialize._rebuild_reduction,
                (self.__class__, serialize._reduce_function(self.py_func),
                 self.locals, self.targetoptions, self._can_compile, sigs))

    @classmethod
    def _rebuild(cls, func_reduced, locals, targetoptions, can_compile, sigs):
        """
        Rebuild an Overloaded instance after it was __reduce__'d.
        """
        py_func = serialize._rebuild_function(*func_reduced)
        self = cls(py_func, locals, targetoptions)
        for sig in sigs:
            self.compile(sig)
        self._can_compile = can_compile
        return self

    def compile(self, sig):
        with self._compile_lock:
            args, return_type = sigutils.normalize_signature(sig)
            # Don't recompile if signature already exists
            # (e.g. if another thread compiled it before we got the lock)
            existing = self.overloads.get(tuple(args))
            if existing is not None:
                return existing

            flags = compiler.Flags()
            self.targetdescr.options.parse_as_flags(flags, self.targetoptions)

            cres = compiler.compile_extra(self.typingctx, self.targetctx,
                                          self.py_func,
                                          args=args, return_type=return_type,
                                          flags=flags, locals=self.locals)

            # Check typing error if object mode is used
            if cres.typing_error is not None and not flags.enable_pyobject:
                raise cres.typing_error

            self.add_overload(cres)
            return cres.entry_point

    def recompile(self):
        """
        Recompile all signatures afresh.
        """
        sigs = [cr.signature for cr in self._compileinfos.values()]
        old_can_compile = self._can_compile
        # Ensure the old overloads are disposed of, including compiled functions.
        self._make_finalizer()()
        self._reset_overloads()
        self._can_compile = True
        try:
            for sig in sigs:
                self.compile(sig)
        finally:
            self._can_compile = old_can_compile


class LiftedLoop(_OverloadedBase):
    """
    Implementation of the hidden dispatcher objects used for lifted loop
    (a lifted loop is really compiled as a separate function).
    """

    def __init__(self, bytecode, typingctx, targetctx, locals, flags):
        self.typingctx = typingctx
        self.targetctx = targetctx

        argspec = bytecode.argspec
        argct = len(argspec.args)

        _OverloadedBase.__init__(self, argct, bytecode.func)

        self.locals = locals
        self.flags = flags
        self.bytecode = bytecode

    def get_source_location(self):
        """Return the starting line number of the loop.
        """
        return next(iter(self.bytecode)).lineno

    def compile(self, sig):
        with self._compile_lock:
            # FIXME this is mostly duplicated from Overloaded
            flags = self.flags
            args, return_type = sigutils.normalize_signature(sig)

            # Don't recompile if signature already exists
            # (e.g. if another thread compiled it before we got the lock)
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
