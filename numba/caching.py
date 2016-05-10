"""
Caching mechanism for compiled functions.
"""

from __future__ import print_function, division, absolute_import

from abc import ABCMeta, abstractmethod, abstractproperty
import contextlib
import errno
import hashlib
import inspect
import itertools
import os
from .six.moves import cPickle as pickle
import sys
import tempfile
import warnings

from .appdirs import AppDirs
from .six import add_metaclass

import numba
from . import compiler, config, utils
from .errors import NumbaWarning


def _cache_log(msg, *args):
    if config.DEBUG_CACHE:
        msg = msg % args
        print(msg)


@add_metaclass(ABCMeta)
class _Cache(object):

    @abstractproperty
    def cache_path(self):
        """
        The base filesystem path of this cache (for example its root folder).
        """

    @abstractmethod
    def load_overload(self, sig, target_context):
        """
        Load an overload for the given signature using the target context.
        A CompileResult must be returned if successful, None if not found
        in the cache.
        """

    @abstractmethod
    def save_overload(self, sig, cres):
        """
        Save the overload for the given signature.
        """

    @abstractmethod
    def enable(self):
        """
        Enable the cache.
        """

    @abstractmethod
    def disable(self):
        """
        Disable the cache.
        """

    @abstractmethod
    def flush(self):
        """
        Flush the cache.
        """


class NullCache(_Cache):

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


@add_metaclass(ABCMeta)
class _CacheLocator(object):
    """
    A filesystem locator for caching a given function.
    """

    def ensure_cache_path(self):
        path = self.get_cache_path()
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        # Ensure the directory is writable by trying to write a temporary file
        tempfile.TemporaryFile(dir=path).close()

    @abstractmethod
    def get_cache_path(self):
        """
        Return the directory the function is cached in.
        """

    @abstractmethod
    def get_source_stamp(self):
        """
        Get a timestamp representing the source code's freshness.
        Can return any picklable Python object.
        """

    @abstractmethod
    def get_disambiguator(self):
        """
        Get a string disambiguator for this locator's function.
        It should allow disambiguating different but similarly-named functions.
        """

    @classmethod
    def from_function(cls, py_func, py_file):
        """
        Create a locator instance for the given function located in the
        given file.
        """
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


class FunctionCache(_Cache):
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
