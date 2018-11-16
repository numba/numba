# -*- coding: utf8 -*-

from __future__ import print_function, division, absolute_import

import collections
import functools
import os
import struct
import sys
import uuid
import weakref

from numba import _dispatcher, compiler, utils, types, config, errors
from numba.typeconv.rules import default_type_manager
from numba import sigutils, serialize, typing
from numba.typing.templates import fold_arguments
from numba.typing.typeof import Purpose, typeof
from numba.bytecode import get_code_object
from numba.six import create_bound_method, reraise
from .caching import NullCache, FunctionCache


class OmittedArg(object):
    """
    A placeholder for omitted arguments with a default value.
    """

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return "omitted arg(%r)" % (self.value,)

    @property
    def _numba_type_(self):
        return types.Omitted(self.value)


class _FunctionCompiler(object):

    def __init__(self, py_func, targetdescr, targetoptions, locals,
                 pipeline_class):
        self.py_func = py_func
        self.targetdescr = targetdescr
        self.targetoptions = targetoptions
        self.locals = locals
        self.pysig = utils.pysignature(self.py_func)
        self.pipeline_class = pipeline_class

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
                                      flags=flags, locals=self.locals,
                                      pipeline_class=self.pipeline_class)
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

    def __init__(self, py_func, targetdescr, targetoptions, locals,
                 pipeline_class):
        super(_GeneratedFunctionCompiler, self).__init__(
            py_func, targetdescr, targetoptions, locals, pipeline_class)
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


class _CompilingCounter(object):
    """
    A simple counter that increment in __enter__ and decrement in __exit__.
    """

    def __init__(self):
        self.counter = 0

    def __enter__(self):
        assert self.counter >= 0
        self.counter += 1

    def __exit__(self, *args, **kwargs):
        self.counter -= 1
        assert self.counter >= 0

    def __bool__(self):
        return self.counter > 0

    __nonzero__ = __bool__


class _DispatcherBase(_dispatcher.Dispatcher):
    """
    Common base class for dispatcher Implementations.
    """

    __numba__ = "py_func"

    def __init__(self, arg_count, py_func, pysig, can_fallback):
        self._tm = default_type_manager

        # A mapping of signatures to compile results
        self.overloads = collections.OrderedDict()

        self.py_func = py_func
        # other parts of Numba assume the old Python 2 name for code object
        self.func_code = get_code_object(py_func)
        # but newer python uses a different name
        self.__code__ = self.func_code

        argnames = tuple(pysig.parameters)
        default_values = self.py_func.__defaults__ or ()
        defargs = tuple(OmittedArg(val) for val in default_values)
        try:
            lastarg = list(pysig.parameters.values())[-1]
        except IndexError:
            has_stararg = False
        else:
            has_stararg = lastarg.kind == lastarg.VAR_POSITIONAL
        _dispatcher.Dispatcher.__init__(self, self._tm.get_pointer(),
                                        arg_count, self._fold_args,
                                        argnames, defargs,
                                        can_fallback,
                                        has_stararg)

        self.doc = py_func.__doc__
        self._compiling_counter = _CompilingCounter()
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
        assert (not val) or len(self.signatures) > 0
        self._can_compile = not val

    def add_overload(self, cres):
        args = tuple(cres.signature.args)
        sig = [a._code for a in args]
        self._insert(sig, cres.entry_point, cres.objectmode, cres.interpmode)
        self.overloads[args] = cres

    def fold_argument_types(self, args, kws):
        return self._compiler.fold_argument_types(args, kws)

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
        # Ensure an overload is available
        if self._can_compile:
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
        return self._compiling_counter

    def _compile_for_args(self, *args, **kws):
        """
        For internal use.  Compile a specialized version of the function
        for the given *args* and *kws*, and return the resulting callable.
        """
        assert not kws

        def error_rewrite(e, issue_type):
            """
            Rewrite and raise Exception `e` with help supplied based on the
            specified issue_type.
            """
            if config.SHOW_HELP:
                help_msg = errors.error_extras[issue_type]
                e.patch_message(''.join(e.args) + help_msg)
            if config.FULL_TRACEBACKS:
                raise e
            else:
                reraise(type(e), e, None)

        argtypes = []
        for a in args:
            if isinstance(a, OmittedArg):
                argtypes.append(types.Omitted(a.value))
            else:
                argtypes.append(self.typeof_pyval(a))
        try:
            return self.compile(tuple(argtypes))
        except errors.TypingError as e:
            # Intercept typing error that may be due to an argument
            # that failed inferencing as a Numba type
            failed_args = []
            for i, arg in enumerate(args):
                val = arg.value if isinstance(arg, OmittedArg) else arg
                try:
                    tp = typeof(val, Purpose.argument)
                except ValueError as typeof_exc:
                    failed_args.append((i, str(typeof_exc)))
                else:
                    if tp is None:
                        failed_args.append(
                            (i,
                             "cannot determine Numba type of value %r" % (val,)))
            if failed_args:
                # Patch error message to ease debugging
                msg = str(e).rstrip() + (
                    "\n\nThis error may have been caused by the following argument(s):\n%s\n"
                    % "\n".join("- argument %d: %s" % (i, err)
                                for i, err in failed_args))
                e.patch_message(msg)

            error_rewrite(e, 'typing')
        except errors.UnsupportedError as e:
            # Something unsupported is present in the user code, add help info
            error_rewrite(e, 'unsupported_error')
        except (errors.NotDefinedError, errors.RedefinedError,
                errors.VerificationError) as e:
            # These errors are probably from an issue with either the code supplied
            # being syntactically or otherwise invalid
            error_rewrite(e, 'interpreter')
        except errors.ConstantInferenceError as e:
            # this is from trying to infer something as constant when it isn't
            # or isn't supported as a constant
            error_rewrite(e, 'constant_inference')
        except Exception as e:
            if config.SHOW_HELP:
                if hasattr(e, 'patch_message'):
                    help_msg = errors.error_extras['reportable']
                    e.patch_message(''.join(e.args) + help_msg)
            # ignore the FULL_TRACEBACKS config, this needs reporting!
            raise e

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

    def inspect_types(self, file=None, **kwargs):
        """
        print or return annotated source with Numba intermediate IR

        Pass `pretty=True` to attempt color highlighting, and HTML rendering in
        Jupyter and IPython by returning an Annotate Object. `file` must be
        None if used in conjunction with `pretty=True`.
        """
        pretty = kwargs.get('pretty', False)
        style = kwargs.get('style', 'default')

        if not pretty:
            if file is None:
                file = sys.stdout

            for ver, res in utils.iteritems(self.overloads):
                print("%s %s" % (self.py_func.__name__, ver), file=file)
                print('-' * 80, file=file)
                print(res.type_annotation, file=file)
                print('=' * 80, file=file)
        else:
            if file is not None:
                raise ValueError("`file` must be None if `pretty=True`")
            from .pretty_annotate import Annotate
            return Annotate(self, style=style)

    def inspect_cfg(self, signature=None, show_wrapper=None):
        """
        For inspecting the CFG of the function.

        By default the CFG of the user function is showed.  The *show_wrapper*
        option can be set to "python" or "cfunc" to show the python wrapper
        function or the *cfunc* wrapper function, respectively.
        """
        if signature is not None:
            cres = self.overloads[signature]
            lib = cres.library
            if show_wrapper == 'python':
                fname = cres.fndesc.llvm_cpython_wrapper_name
            elif show_wrapper == 'cfunc':
                fname = cres.fndesc.llvm_cfunc_wrapper_name
            else:
                fname = cres.fndesc.mangled_name
            return lib.get_function_cfg(fname)

        return dict((sig, self.inspect_cfg(sig, show_wrapper=show_wrapper))
                    for sig in self.signatures)

    def get_annotation_info(self, signature=None):
        """
        Gets the annotation information for the function specified by
        signature. If no signature is supplied a dictionary of signature to
        annotation information is returned.
        """
        if signature is not None:
            cres = self.overloads[signature]
            return cres.type_annotation.annotate_raw()
        return dict((sig, self.annotate(sig)) for sig in self.signatures)

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
        # can save a couple µs.
        try:
            tp = typeof(val, Purpose.argument)
        except ValueError:
            tp = types.pyobject
        else:
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
    # hold refs to last N functions deserialized, retaining them in _memo
    # regardless of whether there is another reference
    _recent = collections.deque(maxlen=config.FUNCTION_CACHE_SIZE)
    __uuid = None
    __numba__ = 'py_func'

    def __init__(self, py_func, locals={}, targetoptions={},
                 impl_kind='direct', pipeline_class=compiler.Pipeline):
        """
        Parameters
        ----------
        py_func: function object to be compiled
        locals: dict, optional
            Mapping of local variable names to Numba types.  Used to override
            the types deduced by the type inference engine.
        targetoptions: dict, optional
            Target-specific config options.
        impl_kind: str
            Select the compiler mode for `@jit` and `@generated_jit`
        pipeline_class: type numba.compiler.BasePipeline
            The compiler pipeline type.
        """
        self.typingctx = self.targetdescr.typing_context
        self.targetctx = self.targetdescr.target_context

        pysig = utils.pysignature(py_func)
        arg_count = len(pysig.parameters)
        can_fallback = not targetoptions.get('nopython', False)
        _DispatcherBase.__init__(self, arg_count, py_func, pysig, can_fallback)

        functools.update_wrapper(self, py_func)

        self.targetoptions = targetoptions
        self.locals = locals
        self._cache = NullCache()
        compiler_class = self._impl_kinds[impl_kind]
        self._impl_kind = impl_kind
        self._compiler = compiler_class(py_func, self.targetdescr,
                                        targetoptions, locals, pipeline_class)
        self._cache_hits = collections.Counter()
        self._cache_misses = collections.Counter()

        self._type = types.Dispatcher(self)
        self.typingctx.insert_global(self, self._type)

    @property
    def _numba_type_(self):
        return types.Dispatcher(self)

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
        self._recent.append(self)

    @compiler.global_compiler_lock
    def compile(self, sig):
        if not self._can_compile:
            raise RuntimeError("compilation disabled")
        # Use counter to track recursion compilation depth
        with self._compiling_counter:
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

    def parallel_diagnostics(self, signature=None, level=1):
        """
        Print parallel diagnostic information for the given signature. If no
        signature is present it is printed for all known signatures. level is
        used to adjust the verbosity, level=1 (default) is minimal verbosity,
        and 2, 3, and 4 provide increasing levels of verbosity.
        """
        def dump(sig):
            ol = self.overloads[sig]
            pfdiag = ol.metadata.get('parfor_diagnostics', None)
            if pfdiag is None:
                msg = "No parfors diagnostic available, is 'parallel=True' set?"
                raise ValueError(msg)
            pfdiag.dump(level)
        if signature is not None:
            dump(signature)
        else:
            [dump(sig) for sig in self.signatures]

    def get_metadata(self, signature=None):
        """
        Obtain the compilation metadata for a given signature.
        """
        if signature is not None:
            return self.overloads[signature].metadata
        else:
            return dict((sig, self.overloads[sig].metadata) for sig in self.signatures)


class LiftedCode(_DispatcherBase):
    """
    Implementation of the hidden dispatcher objects used for lifted code
    (a lifted loop is really compiled as a separate function).
    """
    _fold_args = False

    def __init__(self, func_ir, typingctx, targetctx, flags, locals):
        self.func_ir = func_ir
        self.lifted_from = None

        self.typingctx = typingctx
        self.targetctx = targetctx
        self.flags = flags
        self.locals = locals

        _DispatcherBase.__init__(self, self.func_ir.arg_count,
                                 self.func_ir.func_id.func,
                                 self.func_ir.func_id.pysig,
                                 can_fallback=True)

    def get_source_location(self):
        """Return the starting line number of the loop.
        """
        return self.func_ir.loc.line

    def _pre_compile(self, args, return_type, flags):
        """Pre-compile actions
        """
        pass

    @compiler.global_compiler_lock
    def compile(self, sig):
        # Use counter to track recursion compilation depth
        with self._compiling_counter:
            # XXX this is mostly duplicated from Dispatcher.
            flags = self.flags
            args, return_type = sigutils.normalize_signature(sig)

            # Don't recompile if signature already exists
            # (e.g. if another thread compiled it before we got the lock)
            existing = self.overloads.get(tuple(args))
            if existing is not None:
                return existing.entry_point

            self._pre_compile(args, return_type, flags)

            # Clone IR to avoid mutation in rewrite pass
            cloned_func_ir = self.func_ir.copy()
            cres = compiler.compile_ir(typingctx=self.typingctx,
                                        targetctx=self.targetctx,
                                        func_ir=cloned_func_ir,
                                        args=args, return_type=return_type,
                                        flags=flags, locals=self.locals,
                                        lifted=(),
                                        lifted_from=self.lifted_from)

            # Check typing error if object mode is used
            if cres.typing_error is not None and not flags.enable_pyobject:
                raise cres.typing_error

            self.add_overload(cres)
            return cres.entry_point


class LiftedLoop(LiftedCode):
    def _pre_compile(self, args, return_type, flags):
        assert not flags.enable_looplift, "Enable looplift flags is on"


class LiftedWith(LiftedCode):
    @property
    def _numba_type_(self):
        return types.Dispatcher(self)

    def get_call_template(self, args, kws):
        """
        Get a typing.ConcreteTemplate for this dispatcher and the given
        *args* and *kws* types.  This enables the resolving of the return type.

        A (template, pysig, args, kws) tuple is returned.
        """
        # Ensure an overload is available
        if self._can_compile:
            self.compile(tuple(args))

        pysig = None
        # Create function type for typing
        func_name = self.py_func.__name__
        name = "CallTemplate({0})".format(func_name)
        # The `key` isn't really used except for diagnosis here,
        # so avoid keeping a reference to `cfunc`.
        call_template = typing.make_concrete_template(
            name, key=func_name, signatures=self.nopython_signatures)
        return call_template, pysig, args, kws


class ObjModeLiftedWith(LiftedWith):
    def __init__(self, *args, **kwargs):
        self.output_types = kwargs.pop('output_types', None)
        super(LiftedWith, self).__init__(*args, **kwargs)
        if not self.flags.force_pyobject:
            raise ValueError("expecting `flags.force_pyobject`")
        if self.output_types is None:
            raise TypeError('`output_types` must be provided')

    @property
    def _numba_type_(self):
        return types.ObjModeDispatcher(self)

    def get_call_template(self, args, kws):
        """
        Get a typing.ConcreteTemplate for this dispatcher and the given
        *args* and *kws* types.  This enables the resolving of the return type.

        A (template, pysig, args, kws) tuple is returned.
        """
        assert not kws
        self._legalize_arg_types(args)
        # Coerce to object mode
        args = [types.ffi_forced_object] * len(args)

        if self._can_compile:
            self.compile(tuple(args))

        signatures = [typing.signature(self.output_types, *args)]
        pysig = None
        func_name = self.py_func.__name__
        name = "CallTemplate({0})".format(func_name)
        call_template = typing.make_concrete_template(
            name, key=func_name, signatures=signatures)

        return call_template, pysig, args, kws

    def _legalize_arg_types(self, args):
        for i, a in enumerate(args, start=1):
            if isinstance(a, types.List):
                msg = (
                    'Does not support list type inputs into '
                    'with-context for arg {}'
                )
                raise errors.TypingError(msg.format(i))
            elif isinstance(a, types.Dispatcher):
                msg = (
                    'Does not support function type inputs into '
                    'with-context for arg {}'
                )
                raise errors.TypingError(msg.format(i))


# Initialize typeof machinery
_dispatcher.typeof_init(
    OmittedArg,
    dict((str(t), t._code) for t in types.number_domain))
