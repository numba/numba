# -*- coding: utf8 -*-

from __future__ import print_function, division, absolute_import

import collections
import contextlib
import functools
import errno
import hashlib
import itertools
import inspect
import os
from .six.moves import cPickle as pickle
import struct
import sys
import tempfile
import warnings

from .appdirs import AppDirs

import numba
from numba import _dispatcher, compiler, utils, types, config
from numba.typeconv.rules import default_type_manager
from numba import sigutils, serialize, typing
from numba.typing.templates import fold_arguments
from numba.typing.typeof import typeof
from numba.bytecode import get_code_object
from numba.six import create_bound_method, next
from .errors import NumbaWarning


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

def _cache_log(msg, *args):
    if config.DEBUG_CACHE:
        msg = msg % args
        print(msg)


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
                (self.__class__, serialize._reduce_function(self.py_func, globs),
                 self.locals, self.targetoptions, self._impl_kind,
                 self._can_compile, sigs))

    @classmethod
    def _rebuild(cls, func_reduced, locals, targetoptions, impl_kind,
                 can_compile, sigs):
        """
        Rebuild an Dispatcher instance after it was __reduce__'d.
        """
        py_func = serialize._rebuild_function(*func_reduced)
        self = cls(py_func, locals, targetoptions, impl_kind)
        for sig in sigs:
            self.compile(sig)
        self._can_compile = can_compile
        return self

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


class NullCache(object):

    @property
    def cache_path(self):
        return None

    def load_overload(self, sig, target_context):
        pass

    def save_overload(self, sig, cres):
        pass

    def enable(self):
        pass

    def disable(self):
        pass

    def flush(self):
        pass


class _CacheLocator(object):

    def ensure_cache_path(self):
        path = self.get_cache_path()
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        # Ensure the directory is writable by trying to write a temporary file
        tempfile.TemporaryFile(dir=path).close()

    def get_cache_path(self):
        raise NotImplementedError

    def get_source_stamp(self):
        raise NotImplementedError

    def get_disambiguator(self):
        raise NotImplementedError

    @classmethod
    def from_function(cls, py_func, py_file):
        raise NotImplementedError


class _SourceFileBackedLocatorMixin(object):
    """
    A cache locator mixin for functions which are backed by a well-known
    Python source file.
    """

    def get_source_stamp(self):
        st = os.stat(self._py_file)
        # We use both timestamp and size as some filesystems only have second
        # granularity.
        return st.st_mtime, st.st_size

    def get_disambiguator(self):
        return str(self._lineno)

    @classmethod
    def from_function(cls, py_func, py_file):
        if not os.path.exists(py_file):
            # Perhaps a placeholder (e.g. "<ipython-XXX>")
            return
        self = cls(py_func, py_file)
        try:
            self.ensure_cache_path()
        except OSError:
            # Cannot ensure the cache directory exists or is writable
            return
        return self


class _InTreeCacheLocator(_SourceFileBackedLocatorMixin, _CacheLocator):
    """
    A locator for functions backed by a regular Python module with a
    writable __pycache__ directory.
    """

    def __init__(self, py_func, py_file):
        self._py_file = py_file
        self._lineno = py_func.__code__.co_firstlineno
        self._cache_path = os.path.join(os.path.dirname(self._py_file), '__pycache__')

    def get_cache_path(self):
        return self._cache_path


class _UserWideCacheLocator(_SourceFileBackedLocatorMixin, _CacheLocator):
    """
    A locator for functions backed by a regular Python module, cached
    into a user-wide cache directory.
    """

    def __init__(self, py_func, py_file):
        self._py_file = py_file
        self._lineno = py_func.__code__.co_firstlineno
        appdirs = AppDirs(appname="numba", appauthor=False)
        cache_dir = appdirs.user_cache_dir
        cache_subpath = os.path.dirname(py_file)
        if os.name != "nt":
            # On non-Windows, further disambiguate by appending the entire
            # absolute source path to the cache dir, e.g.
            # "$HOME/.cache/numba/usr/lib/.../mypkg/mysubpkg"
            # On Windows, this is undesirable because of path length limitations
            cache_subpath = os.path.abspath(cache_subpath).lstrip(os.path.sep)
        self._cache_path = os.path.join(cache_dir, cache_subpath)

    def get_cache_path(self):
        return self._cache_path


class _IPythonCacheLocator(_CacheLocator):
    """
    A locator for functions entered at the IPython prompt (notebook or other).
    """

    def __init__(self, py_func, py_file):
        self._py_file = py_file
        # Note IPython enhances the linecache module to be able to
        # inspect source code of functions defined on the interactive prompt.
        source = inspect.getsource(py_func)
        if isinstance(source, bytes):
            self._bytes_source = source
        else:
            self._bytes_source = source.encode('utf-8')

    def get_cache_path(self):
        # We could also use jupyter_core.paths.jupyter_runtime_dir()
        # In both cases this is a user-wide directory, so we need to
        # be careful when disambiguating if we don't want too many
        # conflicts (see below).
        try:
            from IPython.paths import get_ipython_cache_dir
        except ImportError:
            # older IPython version
            from IPython.utils.path import get_ipython_cache_dir
        return os.path.join(get_ipython_cache_dir(), 'numba')

    def get_source_stamp(self):
        return hashlib.sha256(self._bytes_source).hexdigest()

    def get_disambiguator(self):
        # Heuristic: we don't want too many variants being saved, but
        # we don't want similar named functions (e.g. "f") to compete
        # for the cache, so we hash the first two lines of the function
        # source (usually this will be the @jit decorator + the function
        # signature).
        firstlines = b''.join(self._bytes_source.splitlines(True)[:2])
        return hashlib.sha256(firstlines).hexdigest()[:10]

    @classmethod
    def from_function(cls, py_func, py_file):
        if not py_file.startswith("<ipython-"):
            return
        try:
            self.ensure_cache_path()
        except OSError:
            # Cannot ensure the cache directory exists
            return
        return cls(py_func, py_file)


class FunctionCache(object):
    """
    A per-function compilation cache.  The cache saves data in separate
    data files and maintains information in an index file.

    There is one index file per function and Python version
    ("function_name-<lineno>.pyXY.nbi") which contains a mapping of
    signatures and architectures to data files.
    It is prefixed by a versioning key and a timestamp of the Python source
    file containing the function.

    There is one data file ("function_name-<lineno>.pyXY.<number>.nbc")
    per function, function signature, target architecture and Python version.

    Separate index and data files per Python version avoid pickle
    compatibility problems.
    """

    _source_stamp = None
    _locator_classes = [_InTreeCacheLocator, _UserWideCacheLocator,
                        _IPythonCacheLocator]

    def __init__(self, py_func):
        try:
            qualname = py_func.__qualname__
        except AttributeError:
            qualname = py_func.__name__
        # Keep the last dotted component, since the package name is already
        # encoded in the directory.
        modname = py_func.__module__.split('.')[-1]
        self._funcname = qualname.split('.')[-1]
        self._fullname = "%s.%s" % (modname, qualname)
        self._is_closure = bool(py_func.__closure__)
        self._lineno = py_func.__code__.co_firstlineno
        abiflags = getattr(sys, 'abiflags', '')

        # Find a locator
        self._source_path = inspect.getfile(py_func)
        for cls in self._locator_classes:
            self._locator = cls.from_function(py_func, self._source_path)
            if self._locator is not None:
                break
        else:
            raise RuntimeError("cannot cache function %r: no locator available "
                               "for file %r" % (qualname, self._source_path))
        self._cache_path = self._locator.get_cache_path()

        # '<' and '>' can appear in the qualname (e.g. '<locals>') but
        # are forbidden in Windows filenames
        fixed_fullname = self._fullname.replace('<', '').replace('>', '')
        filename_base = (
            '%s-%s.py%d%d%s' % (fixed_fullname, self._locator.get_disambiguator(),
                                sys.version_info[0], sys.version_info[1],
                                abiflags)
            )
        self._index_name = '%s.nbi' % (filename_base,)
        self._index_path = os.path.join(self._cache_path, self._index_name)
        self._data_name_pattern = '%s.{number:d}.nbc' % (filename_base,)

        self.enable()

    def __repr__(self):
        return "<%s fullname=%r>" % (self.__class__.__name__, self._fullname)

    @property
    def cache_path(self):
        return self._cache_path

    def enable(self):
        self._enabled = True
        # This may be a bit strict but avoids us maintaining a magic number
        self._version = numba.__version__
        self._source_stamp = self._locator.get_source_stamp()

    def disable(self):
        self._enabled = False

    def flush(self):
        self._save_index({})

    def load_overload(self, sig, target_context):
        """
        Load and recreate the cached CompileResult for the given signature,
        using the *target_context*.
        """
        if not self._enabled:
            return
        overloads = self._load_index()
        key = self._index_key(sig, target_context.codegen())
        data_name = overloads.get(key)
        if data_name is None:
            return
        try:
            return self._load_data(data_name, target_context)
        except EnvironmentError:
            # File could have been removed while the index still refers it.
            return

    def save_overload(self, sig, cres):
        """
        Save the CompileResult for the given signature in the cache.
        """
        if not self._enabled:
            return
        if not self._check_cachable(cres):
            return
        self._locator.ensure_cache_path()
        overloads = self._load_index()
        key = self._index_key(sig, cres.library.codegen)
        try:
            # If key already exists, we will overwrite the file
            data_name = overloads[key]
        except KeyError:
            # Find an available name for the data file
            existing = set(overloads.values())
            for i in itertools.count(1):
                data_name = self._data_name(i)
                if data_name not in existing:
                    break
            overloads[key] = data_name
            self._save_index(overloads)

        self._save_data(data_name, cres)

    def _check_cachable(self, cres):
        """
        Check cachability of the given compile result.
        """
        cannot_cache = None
        if self._is_closure:
            cannot_cache = "as it uses outer variables in a closure"
        elif cres.lifted:
            cannot_cache = "as it uses lifted loops"
        elif cres.has_dynamic_globals:
            cannot_cache = "as it uses dynamic globals (such as ctypes pointers)"
        if cannot_cache:
            msg = ('Cannot cache compiled function "%s" %s'
                   % (self._funcname, cannot_cache))
            warnings.warn_explicit(msg, NumbaWarning,
                                   self._source_path, self._lineno)
            return False
        return True

    def _index_key(self, sig, codegen):
        """
        Compute index key for the given signature and codegen.
        It includes a description of the OS and target architecture.
        """
        return (sig, codegen.magic_tuple())

    def _data_name(self, number):
        return self._data_name_pattern.format(number=number)

    def _data_path(self, name):
        return os.path.join(self._cache_path, name)

    @contextlib.contextmanager
    def _open_for_write(self, filepath):
        """
        Open *filepath* for writing in a race condition-free way
        (hopefully).
        """
        tmpname = '%s.tmp.%d' % (filepath, os.getpid())
        try:
            with open(tmpname, "wb") as f:
                yield f
            utils.file_replace(tmpname, filepath)
        except Exception:
            # In case of error, remove dangling tmp file
            try:
                os.unlink(tmpname)
            except OSError:
                pass
            raise

    def _load_index(self):
        """
        Load the cache index and return it as a dictionary (possibly
        empty if cache is empty or obsolete).
        """
        try:
            with open(self._index_path, "rb") as f:
                version = pickle.load(f)
                data = f.read()
        except EnvironmentError as e:
            # Index doesn't exist yet?
            if e.errno in (errno.ENOENT,):
                return {}
            raise
        if version != self._version:
            # This is another version.  Avoid trying to unpickling the
            # rest of the stream, as that may fail.
            return {}
        stamp, overloads = pickle.loads(data)
        _cache_log("[cache] index loaded from %r", self._index_path)
        if stamp != self._source_stamp:
            # Cache is not fresh.  Stale data files will be eventually
            # overwritten, since they are numbered in incrementing order.
            return {}
        else:
            return overloads

    def _load_data(self, name, target_context):
        path = self._data_path(name)
        with open(path, "rb") as f:
            data = f.read()
        tup = pickle.loads(data)
        _cache_log("[cache] data loaded from %r", path)
        return compiler.CompileResult._rebuild(target_context, *tup)

    def _save_index(self, overloads):
        data = self._source_stamp, overloads
        data = self._dump(data)
        with self._open_for_write(self._index_path) as f:
            pickle.dump(self._version, f, protocol=-1)
            f.write(data)
        _cache_log("[cache] index saved to %r", self._index_path)

    def _save_data(self, name, cres):
        data = cres._reduce()
        data = self._dump(data)
        path = self._data_path(name)
        with self._open_for_write(path) as f:
            f.write(data)
        _cache_log("[cache] data saved to %r", path)

    def _dump(self, obj):
        return pickle.dumps(obj, protocol=-1)
