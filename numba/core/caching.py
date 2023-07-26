"""
Caching mechanism for compiled functions.
"""


from abc import ABCMeta, abstractmethod, abstractproperty
import contextlib
import errno
import hashlib
import inspect
import itertools
import os
import pickle
import sys
import tempfile
import uuid
import warnings
import typing as pt

from numba.misc.appdirs import AppDirs

import numba
from numba.core.errors import NumbaWarning
from numba.core.base import BaseContext
from numba.core.codegen import CodeLibrary, JITCodeLibrary
from numba.core.compiler import CompileResult
from numba.core import config, compiler, types
from numba.core.serialize import dumps
if pt.TYPE_CHECKING:
    from numba.core.typing import Signature
else:
    Signature = pt.Any


# CodegenMagicTuple is different for CPU (3-tuple) and GPU (2-tuple)
CodegenMagicTuple = pt.Tuple[str, ...]
# IndexKey : sig, codegen.magictuple, hashed code, hashed cells
# the sig argument sometimes is a Signature and sometimes a tuple of types
SignatureLike = pt.Union[Signature, pt.Tuple[types.Type, ...], str]
IndexKey = pt.Tuple[
    SignatureLike, CodegenMagicTuple, pt.Tuple[str, str]
]
# FileStamp: tuple of file timestamp and file size
FileStamp = pt.Tuple[float, int]
# IndexData: Tuple of filename for cached code and Dict of file names to FileStamps
IndexOverloadData = pt.Tuple[str, pt.Dict[str, FileStamp]]
# Index data: what gets pickled and saved.
# Tuple of Timestamp and size of main file + IndexOverloadData
IndexData = pt.Tuple[pt.Tuple[float, int], pt.Dict[IndexKey, IndexOverloadData]]
# This is the output of CompileResult._reduce
ReducedCompileResult = pt.Tuple
# CompileResult
OverloadData = pt.Union[JITCodeLibrary, CompileResult]


def _cache_log(msg, *args):
    if config.DEBUG_CACHE:
        msg = msg % args
        print(msg)


class _Cache(metaclass=ABCMeta):

    @abstractproperty
    def cache_path(self):
        """
        The base filesystem path of this cache (for example its root folder).
        """

    @abstractmethod
    def load_overload(self, sig, target_context):
        """
        Load an overload for the given signature using the target context.
        The saved object must be returned if successful, None if not found
        in the cache.
        """

    @abstractmethod
    def save_overload(self, sig, data):
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


class _CacheLocator(metaclass=ABCMeta):
    """
    A filesystem locator for caching a given function.
    """

    def ensure_cache_path(self):
        path = self.get_cache_path()
        os.makedirs(path, exist_ok=True)
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

    @classmethod
    def get_suitable_cache_subpath(cls, py_file):
        """Given the Python file path, compute a suitable path inside the
        cache directory.

        This will reduce a file path that is too long, which can be a problem
        on some operating system (i.e. Windows 7).
        """
        path = os.path.abspath(py_file)
        subpath = os.path.dirname(path)
        parentdir = os.path.split(subpath)[-1]
        # Use SHA1 to reduce path length.
        # Note: windows doesn't like long path.
        hashed = hashlib.sha1(subpath.encode()).hexdigest()
        # Retain parent directory name for easier debugging
        return '_'.join([parentdir, hashed])


class _SourceFileBackedLocatorMixin(object):
    """
    A cache locator mixin for functions which are backed by a well-known
    Python source file.
    """

    def get_source_stamp(self) -> FileStamp:
        if getattr(sys, 'frozen', False):
            st = os.stat(sys.executable)
        else:
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


class _UserProvidedCacheLocator(_SourceFileBackedLocatorMixin, _CacheLocator):
    """
    A locator that always point to the user provided directory in
    `numba.config.CACHE_DIR`
    """
    def __init__(self, py_func, py_file):
        self._py_file = py_file
        self._lineno = py_func.__code__.co_firstlineno
        cache_subpath = self.get_suitable_cache_subpath(py_file)
        self._cache_path = os.path.join(config.CACHE_DIR, cache_subpath)

    def get_cache_path(self):
        return self._cache_path

    @classmethod
    def from_function(cls, py_func, py_file):
        if not config.CACHE_DIR:
            return
        parent = super(_UserProvidedCacheLocator, cls)
        return parent.from_function(py_func, py_file)


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
    A locator for functions backed by a regular Python module or a
    frozen executable, cached into a user-wide cache directory.
    """

    def __init__(self, py_func, py_file):
        self._py_file = py_file
        self._lineno = py_func.__code__.co_firstlineno
        appdirs = AppDirs(appname="numba", appauthor=False)
        cache_dir = appdirs.user_cache_dir
        cache_subpath = self.get_suitable_cache_subpath(py_file)
        self._cache_path = os.path.join(cache_dir, cache_subpath)

    def get_cache_path(self):
        return self._cache_path

    @classmethod
    def from_function(cls, py_func, py_file):
        if not (os.path.exists(py_file) or getattr(sys, 'frozen', False)):
            # Perhaps a placeholder (e.g. "<ipython-XXX>")
            # stop function exit if frozen, since it uses a temp placeholder
            return
        self = cls(py_func, py_file)
        try:
            self.ensure_cache_path()
        except OSError:
            # Cannot ensure the cache directory exists or is writable
            return
        return self


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
        return os.path.join(get_ipython_cache_dir(), 'numba_cache')

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
        if not (
            py_file.startswith("<ipython-")
            or os.path.basename(os.path.dirname(py_file)).startswith("ipykernel_")
        ):
            return
        self = cls(py_func, py_file)
        try:
            self.ensure_cache_path()
        except OSError:
            # Cannot ensure the cache directory exists
            return
        return self


class CacheImpl(metaclass=ABCMeta):
    """
    Provides the core machinery for caching.
    - implement how to serialize and deserialize the data in the cache.
    - control the filename of the cache.
    - provide the cache locator
    """
    _locator_classes = [_UserProvidedCacheLocator,
                        _InTreeCacheLocator,
                        _UserWideCacheLocator,
                        _IPythonCacheLocator]

    def __init__(self, py_func):
        self._lineno = py_func.__code__.co_firstlineno
        # Get qualname
        try:
            qualname = py_func.__qualname__
        except AttributeError:
            qualname = py_func.__name__
        # Find a locator
        source_path = inspect.getfile(py_func)
        for cls in self._locator_classes:
            locator = cls.from_function(py_func, source_path)
            if locator is not None:
                break
        else:
            raise RuntimeError("cannot cache function %r: no locator available "
                               "for file %r" % (qualname, source_path))
        self._locator = locator
        # Use filename base name as module name to avoid conflict between
        # foo/__init__.py and foo/foo.py
        filename = inspect.getfile(py_func)
        modname = os.path.splitext(os.path.basename(filename))[0]
        fullname = "%s.%s" % (modname, qualname)
        abiflags = getattr(sys, 'abiflags', '')
        self._filename_base = self.get_filename_base(fullname, abiflags)

    def get_filename_base(self, fullname, abiflags):
        # '<' and '>' can appear in the qualname (e.g. '<locals>') but
        # are forbidden in Windows filenames
        fixed_fullname = fullname.replace('<', '').replace('>', '')
        fmt = '%s-%s.py%d%d%s'
        return fmt % (fixed_fullname, self.locator.get_disambiguator(),
                      sys.version_info[0], sys.version_info[1], abiflags)

    @property
    def filename_base(self):
        return self._filename_base

    @property
    def locator(self):
        return self._locator

    @abstractmethod
    def reduce(self, data):
        "Returns the serialized form the data"
        pass

    @abstractmethod
    def rebuild(self, target_context, reduced_data):
        "Returns the de-serialized form of the *reduced_data*"
        pass

    @abstractmethod
    def check_cachable(self, data):
        "Returns True if the given data is cachable; otherwise, returns False."
        pass


class CompileResultCacheImpl(CacheImpl):
    """
    Implements the logic to cache CompileResult objects.
    """

    def reduce(self, cres):
        """
        Returns a serialized CompileResult
        """
        return cres._reduce()

    def rebuild(self, target_context, payload):
        """
        Returns the unserialized CompileResult
        """
        return compiler.CompileResult._rebuild(target_context, *payload)

    def check_cachable(self, cres):
        """
        Check cachability of the given compile result.
        """
        cannot_cache = None
        if any(not x.can_cache for x in cres.lifted):
            cannot_cache = "as it uses lifted code"
        elif cres.library.has_dynamic_globals:
            cannot_cache = ("as it uses dynamic globals "
                            "(such as ctypes pointers and large global arrays)")
        if cannot_cache:
            msg = ('Cannot cache compiled function "%s" %s'
                   % (cres.fndesc.qualname.split('.')[-1], cannot_cache))
            warnings.warn_explicit(msg, NumbaWarning,
                                   self._locator._py_file, self._lineno)
            return False
        return True


class CodeLibraryCacheImpl(CacheImpl):
    """
    Implements the logic to cache CodeLibrary objects.
    """

    _filename_prefix = None  # must be overridden

    def reduce(self, codelib):
        """
        Returns a serialized CodeLibrary
        """
        return codelib.serialize_using_object_code()

    def rebuild(self, target_context, payload):
        """
        Returns the unserialized CodeLibrary
        """
        return target_context.codegen().unserialize_library(payload)

    def check_cachable(self, codelib):
        """
        Check cachability of the given CodeLibrary.
        """
        return not codelib.has_dynamic_globals

    def get_filename_base(self, fullname, abiflags):
        parent = super(CodeLibraryCacheImpl, self)
        res = parent.get_filename_base(fullname, abiflags)
        return '-'.join([self._filename_prefix, res])


class IndexDataCacheFile(object):
    """
    Implements the logic for the index file and data file used by a cache.
    """
    def __init__(self, cache_path, filename_base, source_stamp):
        self._cache_path = cache_path
        self._index_name = '%s.nbi' % (filename_base,)
        self._index_path = os.path.join(self._cache_path, self._index_name)
        self._data_name_pattern = '%s.{number:d}.nbc' % (filename_base,)
        self._source_stamp = source_stamp
        self._version = numba.__version__

    def flush(self):
        self._save_index({})

    def save(self,
             key: IndexKey,
             data: ReducedCompileResult,
             deps_filestamps: pt.Dict[str, FileStamp]
             ):
        """
        Save a new cache entry with *key* and *data*.
        """
        overloads = self._load_index()
        try:
            # If key already exists, we will overwrite the file
            # but we will update the filestamps in the index
            data_filename, old_deps_filestamps = overloads[key]
        except KeyError:
            # Find an available name for the data file
            existing = set((name for name, filestamp in overloads.values()))
            for i in itertools.count(1):
                data_filename = self._data_name(i)
                if data_filename not in existing:
                    break
        overloads[key] = data_filename, deps_filestamps
        self._save_index(overloads)
        self._save_data(data_filename, data)

    def load(self, key: IndexKey):
        """
        Load a cache entry with *key*.
        """
        overloads = self._load_index()
        data_name, deps_filestamps = overloads.get(key, (None, None))
        if data_name is None:
            return
        if config.DEBUG_CACHE:
            files = list(deps_filestamps.keys())
            _cache_log(f'[cache] dependencies of cache file "{data_name}": {files}')
        if not are_filestamps_valid(deps_filestamps):
            _cache_log("Some files have changed, cache has been invalidated")
            return
        try:
            cache_data = self._load_data(data_name)
        except OSError:
            # File could have been removed while the index still refers it.
            return
        return cache_data
    def load_deps(self,  key: IndexKey) -> pt.Dict[str, FileStamp]:
        """
        Load the cached dependencies' FileStamps for the given key
        """
        overloads = self._load_index()
        data_name, deps_filestamps = overloads.get(key, (None, None))
        if config.DEBUG_CACHE:
            files = list(deps_filestamps.keys())
            _cache_log(f'[cache] Loading dependencies of "{key}": {files}')
        return deps_filestamps

    def _load_index(self) -> pt.Dict[IndexKey, IndexOverloadData]:
        """
        Load the cache index and return it as a dictionary (possibly
        empty if cache is empty or obsolete).
        """
        try:
            with open(self._index_path, "rb") as f:
                version = pickle.load(f)
                data = f.read()
        except FileNotFoundError:
            # Index doesn't exist yet?
            return {}
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

    def _save_index(self, overloads: pt.Dict[IndexKey, IndexOverloadData]) -> None:
        data: IndexData  # for python 3.7, otherwise put in next line
        data = self._source_stamp, overloads
        data_bytes = self._dump(data)
        with self._open_for_write(self._index_path) as f:
            pickle.dump(self._version, f, protocol=-1)
            f.write(data_bytes)
        _cache_log("[cache] index saved to %r", self._index_path)

    def _load_data(self, name):
        path = self._data_path(name)
        with open(path, "rb") as f:
            data = f.read()
        tup = pickle.loads(data)
        _cache_log("[cache] data loaded from %r", path)
        return tup

    def _save_data(self, name, data):
        data = self._dump(data)
        path = self._data_path(name)
        with self._open_for_write(path) as f:
            f.write(data)
        _cache_log("[cache] data saved to %r", path)

    def _data_name(self, number):
        return self._data_name_pattern.format(number=number)

    def _data_path(self, name):
        return os.path.join(self._cache_path, name)

    def _dump(self, obj):
        return dumps(obj)

    @contextlib.contextmanager
    def _open_for_write(self, filepath):
        """
        Open *filepath* for writing in a race condition-free way (hopefully).
        uuid4 is used to try and avoid name collisions on a shared filesystem.
        """
        uid = uuid.uuid4().hex[:16]  # avoid long paths
        tmpname = '%s.tmp.%s' % (filepath, uid)
        try:
            with open(tmpname, "wb") as f:
                yield f
            os.replace(tmpname, filepath)
        except Exception:
            # In case of error, remove dangling tmp file
            try:
                os.unlink(tmpname)
            except OSError:
                pass
            raise


class Cache(_Cache):
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

    Note:
    This contains the driver logic only.  The core logic is provided
    by a subclass of ``CacheImpl`` specified as *_impl_class* in the subclass.
    """

    # The following class variables must be overridden by subclass.
    _impl_class = None

    def __init__(self, py_func):
        self._name = repr(py_func)
        self._py_func = py_func
        self._impl = self._impl_class(py_func)
        self._cache_path = self._impl.locator.get_cache_path()
        # This may be a bit strict but avoids us maintaining a magic number
        source_stamp = self._impl.locator.get_source_stamp()
        filename_base = self._impl.filename_base
        self._cache_file = IndexDataCacheFile(cache_path=self._cache_path,
                                              filename_base=filename_base,
                                              source_stamp=source_stamp)
        self.enable()

    def __repr__(self):
        return "<%s py_func=%r>" % (self.__class__.__name__, self._name)

    @property
    def cache_path(self):
        return self._cache_path

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def flush(self):
        self._cache_file.flush()

    def load_overload(self, sig, target_context):
        """
        Load and recreate the cached object for the given signature,
        using the *target_context*.
        """
        # Refresh the context to ensure it is initialized
        target_context.refresh()
        with self._guard_against_spurious_io_errors():
            return self._load_overload(sig, target_context)
        # None returned if the `with` block swallows an exception

    def _load_overload(self, sig, target_context):
        if not self._enabled:
            return
        key = self._index_key(sig, target_context.codegen())
        data = self._cache_file.load(key)
        if data is not None:
            data = self._impl.rebuild(target_context, data)
        return data

    def save_overload(self, sig: SignatureLike, data: OverloadData) -> None:
        """
        Save the data for the given signature in the cache.

        sig: numba types of input arguments as one of SignatureLike instances
        data: object containing compiled code to be cached
        """
        with self._guard_against_spurious_io_errors():
            self._save_overload(sig, data)

    def _save_overload(self, sig: SignatureLike, data: OverloadData) -> None:
        if not self._enabled:
            return
        if not self._impl.check_cachable(data):
            return
        self._impl.locator.ensure_cache_path()
        # get timestamps of all dependency files
        deps_filestamps = get_function_dependencies(data)
        # get index key and reduce CompileResult
        key = self._index_key(sig, data.codegen)
        data = self._impl.reduce(data)
        self._cache_file.save(key, data, deps_filestamps)

    @contextlib.contextmanager
    def _guard_against_spurious_io_errors(self):
        if os.name == 'nt':
            # Guard against permission errors due to accessing the file
            # from several processes (see #2028)
            try:
                yield
            except OSError as e:
                if e.errno != errno.EACCES:
                    raise
        else:
            # No such conditions under non-Windows OSes
            yield

    def _index_key(self, sig: SignatureLike, codegen) -> IndexKey:
        """
        Compute index key for the given signature and codegen.
        It includes a description of the OS, target architecture and hashes of
        the bytecode for the function and, if the function has a __closure__,
        a hash of the cell_contents.
        """
        codebytes = self._py_func.__code__.co_code
        if self._py_func.__closure__ is not None:
            cvars = tuple([x.cell_contents for x in self._py_func.__closure__])
            # Note: cloudpickle serializes a function differently depending
            #       on how the process is launched; e.g. multiprocessing.Process
            cvarbytes = dumps(cvars)
        else:
            cvarbytes = b''

        hasher = lambda x: hashlib.sha256(x).hexdigest()
        return (sig, codegen.magic_tuple(), (hasher(codebytes),
                                             hasher(cvarbytes),))

    def load_cached_deps(self, sig_args, target_context) -> pt.Dict[str, FileStamp]:
        key = self._index_key(sig_args, target_context.codegen())
        deps = self._cache_file.load_deps(key)
        return deps


class FunctionCache(Cache):
    """
    Implements Cache that saves and loads CompileResult objects.
    """
    _impl_class = CompileResultCacheImpl


# Remember used cache filename prefixes.
_lib_cache_prefixes = set([''])


def make_library_cache(prefix):
    """
    Create a Cache class for additional compilation features to cache their
    result for reuse.  The cache is saved in filename pattern like
    in ``FunctionCache`` but with additional *prefix* as specified.
    """
    # avoid cache prefix reuse
    assert prefix not in _lib_cache_prefixes
    _lib_cache_prefixes.add(prefix)

    class CustomCodeLibraryCacheImpl(CodeLibraryCacheImpl):
        _filename_prefix = prefix

    class LibraryCache(Cache):
        """
        Implements Cache that saves and loads CodeLibrary objects for additional
        feature for the specified python function.
        """
        _impl_class = CustomCodeLibraryCacheImpl

    return LibraryCache


# list of types for which the cache invalidation is extended to their
# definitions
dep_types = (types.Dispatcher, types.Function)


def get_function_dependencies(overload: OverloadData
                              ) -> pt.Dict[str, FileStamp]:
    """ Returns functions on which the overload depends, and their file stamps

    Not every function dependency is returned. The list of types returned
    if found being called by the overload are in ``dep_types`` global.
    """
    deps = {}
    if not isinstance(overload, CompileResult):
        return {}
    typemap = overload.type_annotation.typemap
    calltypes = overload.type_annotation.calltypes
    for call_op in calltypes:
        name = call_op.list_vars()[0].name
        if name not in typemap:
            # for parfor generated callables which cannot be found in  typemap
            continue
        fc_ty = typemap[name]
        sig = calltypes[call_op]
        if not isinstance(fc_ty, dep_types):
            continue
        # get every file where the function is defined
        py_files = get_impl_filenames(fc_ty)
        for py_file in py_files:
            deps[py_file] = get_source_stamp(py_file)
        # retrieve all dependencies of the function in fc_ty
        indirect_deps = get_deps_info(fc_ty, sig)
        deps.update(indirect_deps)

    if config.DEBUG_CACHE:
        files = list(deps.keys())
        func = overload.library.name
        _cache_log(f'[cache] dependencies of "{func}": {files}')
    return deps


def get_deps_info(fc_ty: pt.Union[types.Dispatcher, types.Function], sig
                      ) -> pt.Dict[str, FileStamp]:
    """
    Retrieve dependency information for the implementation of the function
    represented in the given function type
    :param fc_ty:
    :return: dictionary of filenames to FileStamp
    """
    if isinstance(fc_ty, types.Dispatcher):
        dispatcher = fc_ty.dispatcher
        deps_stamps = dispatcher.cache_deps_info(sig)
    elif isinstance(fc_ty, types.Function):
        if hasattr(fc_ty.typing_key, "_dispatcher"):
            # this case captures DUFuncs and GUFuncs
            dispatcher = fc_ty.key[0]._dispatcher
            deps_stamps = dispatcher.cache_deps_info(sig)
        else:
            # a type of Function with a dispatcher associated. Probably an
            # overload
            # If the template does not have `get_cache_deps_info` it might be
            # a generated class for a global value in Registry.register_global
            deps_stamps = [tmplt.get_cache_deps_info(tmplt, sig, get_function_dependencies)
                           for tmplt in fc_ty.templates
                           if hasattr(tmplt, 'get_cache_deps_info')]
            deps_stamps = {k: v for d in deps_stamps for k, v in d.items()}
    return deps_stamps


def get_impl_filenames(fc_ty: pt.Union[types.Dispatcher, types.Function]
                      ) -> pt.List[str]:
    """
    Return the filename where the implementation of the function
    represented by the function type `fc_ty` is.
    For dispatchers, this will be a single file, where the function is written.
    For overloads, this will be one or more files, where *every* overload to
    that function is written.

    :param func: Dispatcher or FunctionType
    :return: list of filenames
    """
    py_files = None
    if isinstance(fc_ty, types.Dispatcher):
        dispatcher = fc_ty.dispatcher
        py_func = dispatcher.py_func
        py_files = [py_func.__code__.co_filename]
    elif isinstance(fc_ty, types.Function):
        if hasattr(fc_ty.typing_key, "_dispatcher"):
            # this case captures DUFuncs and GUFuncs
            dispatcher = fc_ty.key[0]._dispatcher
            py_func = dispatcher.py_func
            py_files = [py_func.__code__.co_filename]
        else:
            # a type of Function with a dispatcher associated. Probably an
            # overload
            py_files = [tmplt.get_template_info(tmplt)["filename"]
                        for tmplt in fc_ty.templates]

            # the base path depends on what tmplt.get_template_info is doing
            # in this case, the filenames returned by get_template_info are
            # relative to numba.__file__
            basepath = os.path.dirname(os.path.dirname(numba.__file__))

            def make_abs(basepath, rel_path):
                return os.path.realpath(os.path.join(basepath, rel_path))

            py_files = [make_abs(basepath, f) for f in py_files]

            # `get_template_info` can return invalid file paths such as
            # "unknown" or "<unknown> (built from string?)` so non-paths are
            # filtered out
            py_files = [file for file in py_files if os.path.isfile(file)]
    else:
        raise TypeError
    return py_files


def are_filestamps_valid(filestamps: pt.Dict[str, FileStamp]) -> bool:
    """
    Checks whether the input FileStamps for certain files match freshly
    calculated FileStamps for those files
    """
    for filepath, source_stamp in filestamps.items():
        if get_source_stamp(filepath) != source_stamp:
            return False
    return True


def get_source_stamp(filepath: str) -> FileStamp:
    """
    Calculates the current FileStamp of a given file
    :param filepath: full filepath of a file
    :return: FileStamp (combination of size and timestamp
    """
    if getattr(sys, 'frozen', False):
        st = os.stat(sys.executable)
    else:
        st = os.stat(filepath)
    # We use both timestamp and size as some filesystems only have second
    # granularity.
    return st.st_mtime, st.st_size