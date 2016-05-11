# -*- coding: utf8 -*-

from __future__ import print_function, division, absolute_import

import collections
import functools
import os
import struct
import sys
import uuid
import weakref

import numba
from numba import _dispatcher, compiler, utils, types, config
from numba.typeconv.rules import default_type_manager
from numba import sigutils, serialize, typing
from numba.typing.templates import fold_arguments
from numba.typing.typeof import typeof
from numba.bytecode import get_code_object
from numba.six import create_bound_method, next
from .caching import NullCache, FunctionCache


class OmittedArg(object):
    """
    A placeholder for omitted arguments with a default value.
    """

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return "omitted arg(%r)" % (self.value,)


class _FunctionCompiler(object):

    def __init__(self, py_func, targetdescr, targetoptions, locals):
        self.py_func = py_func
        self.targetdescr = targetdescr
        self.targetoptions = targetoptions
        self.locals = locals
        self.pysig = utils.pysignature(self.py_func)

    def fold_argument_types(self, args, kws):
        """
        Given positional and named argument types, fold keyword arguments
        and resolve defaults by inserting types.Omitted() instances.

        A (pysig, argument types) tuple is returned.
        """
        def normal_handler(index, param, value):
            return value
        def default_handler(index, param, default):
            return types.Omitted(default)
        def stararg_handler(index, param, values):
            return types.Tuple(values)
        # For now, we take argument values from the @jit function, even
        # in the case of generated jit.
        args = fold_arguments(self.pysig, args, kws,
                              normal_handler,
                              default_handler,
                              stararg_handler)
        return self.pysig, args

    def compile(self, args, return_type):
        flags = compiler.Flags()
        self.targetdescr.options.parse_as_flags(flags, self.targetoptions)
        flags = self._customize_flags(flags)

        impl = self._get_implementation(args, {})
        cres = compiler.compile_extra(self.targetdescr.typing_context,
                                      self.targetdescr.target_context,
                                      impl,
                                      args=args, return_type=return_type,
                                      flags=flags, locals=self.locals)
        # Check typing error if object mode is used
        if cres.typing_error is not None and not flags.enable_pyobject:
            raise cres.typing_error
        return cres

    def get_globals_for_reduction(self):
        return serialize._get_function_globals_for_reduction(self.py_func)

    def _get_implementation(self, args, kws):
        return self.py_func

    def _customize_flags(self, flags):
        return flags


class _GeneratedFunctionCompiler(_FunctionCompiler):

    def __init__(self, py_func, targetdescr, targetoptions, locals):
        super(_GeneratedFunctionCompiler, self).__init__(
            py_func, targetdescr, targetoptions, locals)
        self.impls = set()

    def get_globals_for_reduction(self):
        # This will recursively get the globals used by any nested
        # implementation function.
        return serialize._get_function_globals_for_reduction(self.py_func)

    def _get_implementation(self, args, kws):
        impl = self.py_func(*args, **kws)
        # Check the generating function and implementation signatures are
        # compatible, otherwise compiling would fail later.
        pysig = utils.pysignature(self.py_func)
        implsig = utils.pysignature(impl)
        ok = len(pysig.parameters) == len(implsig.parameters)
        if ok:
            for pyparam, implparam in zip(pysig.parameters.values(),
                                          implsig.parameters.values()):
                # We allow the implementation to omit default values, but
                # if it mentions them, they should have the same value...
                if (pyparam.name != implparam.name or
                    pyparam.kind != implparam.kind or
                    (implparam.default is not implparam.empty and
                     implparam.default != pyparam.default)):
                    ok = False
        if not ok:
            raise TypeError("generated implementation %s should be compatible "
                            "with signature '%s', but has signature '%s'"
                            % (impl, pysig, implsig))
        self.impls.add(impl)
        return impl


_CompileStats = collections.namedtuple(
    '_CompileStats', ('cache_path', 'cache_hits', 'cache_misses'))


class _DispatcherBase(_dispatcher.Dispatcher):
    """
    Common base class for dispatcher Implementations.
    """

    __numba__ = "py_func"

    def __init__(self, arg_count, py_func, pysig):
        self._tm = default_type_manager

        # A mapping of signatures to compile results
        self.overloads = collections.OrderedDict()

        self.py_func = py_func
        # other parts of Numba assume the old Python 2 name for code object
        self.func_code = get_code_object(py_func)
        # but newer python uses a different name
        self.__code__ = self.func_code

        argnames = tuple(pysig.parameters)
        defargs = tuple(OmittedArg(val)
                        for val in (self.py_func.__defaults__ or ()))
        try:
            lastarg = list(pysig.parameters.values())[-1]
        except IndexError:
            has_stararg = False
        else:
            has_stararg = lastarg.kind == lastarg.VAR_POSITIONAL
        _dispatcher.Dispatcher.__init__(self, self._tm.get_pointer(),
                                        arg_count, self._fold_args,
                                        argnames, defargs,
                                        has_stararg)

        self.doc = py_func.__doc__
        self._compile_lock = utils.NonReentrantLock()

        utils.finalize(self, self._make_finalizer())

    def _reset_overloads(self):
        self._clear()
        self.overloads.clear()

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
            for cres in overloads.values():
                try:
                    targetctx.remove_user_function(cres.entry_point)
                except KeyError:
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
        return [cres.signature for cres in self.overloads.values()
                if not cres.objectmode and not cres.interpmode]

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
        self.overloads[args] = cres

    def get_call_template(self, args, kws):
        """
        Get a typing.ConcreteTemplate for this dispatcher and the given
        *args* and *kws* types.  This allows to resolve the return type.

        A (template, pysig, args, kws) tuple is returned.
        """
        # XXX how about a dispatcher template class automating the
        # following?

        # Fold keyword arguments and resolve default values
        pysig, args = self._compiler.fold_argument_types(args, kws)
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
        return call_template, pysig, args, kws

    def get_overload(self, sig):
        """
        Return the compiled function for the given signature.
        """
        args, return_type = sigutils.normalize_signature(sig)
        return self.overloads[tuple(args)].entry_point

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
        real_args = []
        for a in args:
            if isinstance(a, OmittedArg):
                real_args.append(types.Omitted(a.value))
            else:
                real_args.append(self.typeof_pyval(a))
        return self.compile(tuple(real_args))

    def inspect_llvm(self, signature=None):
        if signature is not None:
            lib = self.overloads[signature].library
            return lib.get_llvm_str()

        return dict((sig, self.inspect_llvm(sig)) for sig in self.signatures)

    def inspect_asm(self, signature=None):
        if signature is not None:
            lib = self.overloads[signature].library
            return lib.get_asm_str()

        return dict((sig, self.inspect_asm(sig)) for sig in self.signatures)

    def inspect_types(self, file=None):
        if file is None:
            file = sys.stdout

        for ver, res in utils.iteritems(self.overloads):
            print("%s %s" % (self.py_func.__name__, ver), file=file)
            print('-' * 80, file=file)
            print(res.type_annotation, file=file)
            print('=' * 80, file=file)

    def _explain_ambiguous(self, *args, **kws):
        """
        Callback for the C _Dispatcher object.
        """
        assert not kws, "kwargs not handled"
        args = tuple([self.typeof_pyval(a) for a in args])
        # The order here must be deterministic for testing purposes, which
        # is ensured by the OrderedDict.
        sigs = self.nopython_signatures
        # This will raise
        self.typingctx.resolve_overload(self.py_func, sigs, args, kws,
                                        allow_ambiguous=False)

    def _explain_matching_error(self, *args, **kws):
        """
        Callback for the C _Dispatcher object.
        """
        assert not kws, "kwargs not handled"
        args = [self.typeof_pyval(a) for a in args]
        msg = ("No matching definition for argument type(s) %s"
               % ', '.join(map(str, args)))
        raise TypeError(msg)

    def _search_new_conversions(self, *args, **kws):
        """
        Callback for the C _Dispatcher object.
        Search for approximately matching signatures for the given arguments,
        and ensure the corresponding conversions are registered in the C++
        type manager.
        """
        assert not kws, "kwargs not handled"
        args = [self.typeof_pyval(a) for a in args]
        found = False
        for sig in self.nopython_signatures:
            conv = self.typingctx.install_possible_conversions(args, sig.args)
            if conv:
                found = True
        return found

    def __repr__(self):
        return "%s(%s)" % (type(self).__name__, self.py_func)

    def typeof_pyval(self, val):
        """
        Resolve the Numba type of Python value *val*.
        This is called from numba._dispatcher as a fallback if the native code
        cannot decide the type.
        """
        # Not going through the resolve_argument_type() indirection
        # can shape a couple µs.
        tp = typeof(val)
        if tp is None:
            tp = types.pyobject
        return tp


class Dispatcher(_DispatcherBase):
    """
    Implementation of user-facing dispatcher objects (i.e. created using
    the @jit decorator).
    This is an abstract base class. Subclasses should define the targetdescr
    class attribute.
    """
    _fold_args = True
    _impl_kinds = {
        'direct': _FunctionCompiler,
        'generated': _GeneratedFunctionCompiler,
        }
    # A {uuid -> instance} mapping, for deserialization
    _memo = weakref.WeakValueDictionary()
    __uuid = None

    def __init__(self, py_func, locals={}, targetoptions={}, impl_kind='direct'):
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

        pysig = utils.pysignature(py_func)
        arg_count = len(pysig.parameters)

        _DispatcherBase.__init__(self, arg_count, py_func, pysig)

        functools.update_wrapper(self, py_func)

        self.targetoptions = targetoptions
        self.locals = locals
        self._cache = NullCache()
        compiler_class = self._impl_kinds[impl_kind]
        self._impl_kind = impl_kind
        self._compiler = compiler_class(py_func, self.targetdescr,
                                        targetoptions, locals)
        self._cache_hits = collections.Counter()
        self._cache_misses = collections.Counter()

        self.typingctx.insert_global(self, types.Dispatcher(self))

    def enable_caching(self):
        self._cache = FunctionCache(self.py_func)

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
            sigs = [cr.signature for cr in self.overloads.values()]
        globs = self._compiler.get_globals_for_reduction()
        return (serialize._rebuild_reduction,
                (self.__class__, str(self._uuid),
                 serialize._reduce_function(self.py_func, globs),
                 self.locals, self.targetoptions, self._impl_kind,
                 self._can_compile, sigs))

    @classmethod
    def _rebuild(cls, uuid, func_reduced, locals, targetoptions, impl_kind,
                 can_compile, sigs):
        """
        Rebuild an Dispatcher instance after it was __reduce__'d.
        """
        try:
            return cls._memo[uuid]
        except KeyError:
            pass
        py_func = serialize._rebuild_function(*func_reduced)
        self = cls(py_func, locals, targetoptions, impl_kind)
        # Make sure this deserialization will be merged with subsequent ones
        self._set_uuid(uuid)
        for sig in sigs:
            self.compile(sig)
        self._can_compile = can_compile
        return self

    @property
    def _uuid(self):
        """
        An instance-specific UUID, to avoid multiple deserializations of
        a given instance.

        Note this is lazily-generated, for performance reasons.
        """
        u = self.__uuid
        if u is None:
            u = str(uuid.uuid1())
            self._set_uuid(u)
        return u

    def _set_uuid(self, u):
        assert self.__uuid is None
        self.__uuid = u
        self._memo[u] = self

    def compile(self, sig):
        if not self._can_compile:
            raise RuntimeError("compilation disabled")
        with self._compile_lock:
            args, return_type = sigutils.normalize_signature(sig)
            # Don't recompile if signature already exists
            existing = self.overloads.get(tuple(args))
            if existing is not None:
                return existing.entry_point

            # Try to load from disk cache
            cres = self._cache.load_overload(sig, self.targetctx)
            if cres is not None:
                self._cache_hits[sig] += 1
                # XXX fold this in add_overload()? (also see compiler.py)
                if not cres.objectmode and not cres.interpmode:
                    self.targetctx.insert_user_function(cres.entry_point,
                                                   cres.fndesc, [cres.library])
                self.add_overload(cres)
                return cres.entry_point

            self._cache_misses[sig] += 1
            cres = self._compiler.compile(args, return_type)
            self.add_overload(cres)
            self._cache.save_overload(sig, cres)
            return cres.entry_point

    def recompile(self):
        """
        Recompile all signatures afresh.
        """
        sigs = list(self.overloads)
        old_can_compile = self._can_compile
        # Ensure the old overloads are disposed of, including compiled functions.
        self._make_finalizer()()
        self._reset_overloads()
        self._cache.flush()
        self._can_compile = True
        try:
            for sig in sigs:
                self.compile(sig)
        finally:
            self._can_compile = old_can_compile

    @property
    def stats(self):
        return _CompileStats(
            cache_path=self._cache.cache_path,
            cache_hits=self._cache_hits,
            cache_misses=self._cache_misses,
            )


class LiftedLoop(_DispatcherBase):
    """
    Implementation of the hidden dispatcher objects used for lifted loop
    (a lifted loop is really compiled as a separate function).
    """
    _fold_args = False

    def __init__(self, bytecode, typingctx, targetctx, locals, flags):
        self.typingctx = typingctx
        self.targetctx = targetctx

        _DispatcherBase.__init__(self, bytecode.arg_count, bytecode.func,
                                 bytecode.pysig)

        self.locals = locals
        self.flags = flags
        self.bytecode = bytecode
        self.lifted_from = None

    def get_source_location(self):
        """Return the starting line number of the loop.
        """
        return next(iter(self.bytecode)).lineno

    def compile(self, sig):
        with self._compile_lock:
            # XXX this is mostly duplicated from Dispatcher.
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
                                             locals=self.locals,
                                             lifted=(), lifted_from=self.lifted_from)

            # Check typing error if object mode is used
            if cres.typing_error is not None and not flags.enable_pyobject:
                raise cres.typing_error

            self.add_overload(cres)
            return cres.entry_point


# Initialize typeof machinery
_dispatcher.typeof_init(dict((str(t), t._code) for t in types.number_domain))
