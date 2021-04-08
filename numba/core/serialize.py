"""
Serialization support for compiled functions.
"""

from importlib.util import MAGIC_NUMBER as bc_magic

import abc
import io
import marshal
import sys
import copyreg
from types import FunctionType, ModuleType, CodeType

from numba.core.utils import PYVERSION


if PYVERSION >= (3, 8):
    from types import CellType

    import pickle
    pickle38 = pickle
else:
    try:
        import pickle5 as pickle38
    except ImportError:
        pickle38 = None
        import pickle
    else:
        pickle = pickle38


#
# Pickle support
#

def _rebuild_reduction(cls, *args):
    """
    Global hook to rebuild a given class from its __reduce__ arguments.
    """
    return cls._rebuild(*args)


class _ModuleRef(object):

    def __init__(self, name):
        self.name = name

    def __reduce__(self):
        return _rebuild_module, (self.name,)


def _rebuild_module(name):
    if name is None:
        raise ImportError("cannot import None")
    __import__(name)
    return sys.modules[name]


def _get_function_globals_for_reduction(func):
    """
    Analyse *func* and return a dictionary of global values suitable for
    reduction.
    """
    from numba.core import bytecode   # needed here to avoid cyclic import

    func_id = bytecode.FunctionIdentity.from_function(func)
    bc = bytecode.ByteCode(func_id)
    globs = bc.get_used_globals()
    # Remember the module name so that the function gets a proper __module__
    # when rebuilding.  This is used to recreate the environment.
    globs['__name__'] = func.__module__
    return globs


def _reduce_function(func, globs):
    """
    Reduce a Python function and its globals to picklable components.
    If there are cell variables (i.e. references to a closure), their
    values will be frozen.
    """
    if func.__closure__:
        cells = [cell.cell_contents for cell in func.__closure__]
    else:
        cells = None
    return (_reduce_code(func.__code__), globs, func.__name__, cells,
            func.__defaults__)


def _reduce_code(code):
    """
    Reduce a code object to picklable components.
    """
    return marshal.version, bc_magic, marshal.dumps(code)


def _make_cell(x):
    """
    A dummy function allowing us to build cell objects because
    types.CellType doesn't exist until Python3.8.
    """
    return (lambda: x).__closure__[0]


def _rebuild_function(code_reduced, globals, name, cell_values, defaults):
    """
    Rebuild a function from its _reduce_function() results.
    """
    if cell_values:
        cells = tuple(_make_cell(v) for v in cell_values)
    else:
        cells = ()
    code = _rebuild_code(*code_reduced)
    modname = globals['__name__']
    try:
        _rebuild_module(modname)
    except ImportError:
        # If the module can't be found, avoid passing it (it would produce
        # errors when lowering).
        del globals['__name__']
    return FunctionType(code, globals, name, defaults, cells)


def _rebuild_code(marshal_version, bytecode_magic, marshalled):
    """
    Rebuild a code object from its _reduce_code() results.
    """
    if marshal.version != marshal_version:
        raise RuntimeError("incompatible marshal version: "
                           "interpreter has %r, marshalled code has %r"
                           % (marshal.version, marshal_version))
    if bc_magic != bytecode_magic:
        raise RuntimeError("incompatible bytecode version")
    return marshal.loads(marshalled)


# Keep unpickled object via `numba_unpickle` alive.
_unpickled_memo = {}


def _numba_unpickle(address, bytedata, hashed):
    """Used by `numba_unpickle` from _helperlib.c

    Parameters
    ----------
    address : int
    bytedata : bytes
    hashed : bytes

    Returns
    -------
    obj : object
        unpickled object
    """
    key = (address, hashed)
    try:
        obj = _unpickled_memo[key]
    except KeyError:
        _unpickled_memo[key] = obj = pickle.loads(bytedata)
    return obj


def dumps(obj):
    """Similar to `pickle.dumps()`. Returns the serialized object in bytes.
    """
    pickler = NumbaPickler
    with io.BytesIO() as buf:
        p = pickler(buf)
        p.dump(obj)
        pickled = buf.getvalue()

    return pickled


# Alias to pickle.loads to allow `serialize.loads()`
loads = pickle.loads


class _CustomPickled:
    """A wrapper for objects that must be pickled with `NumbaPickler`.

    Standard `pickle` will pick up the implementation registered via `copyreg`.
    This will spawn a `NumbaPickler` instance to serialize the data.

    `NumbaPickler` overrides the handling of this type so as not to spawn a
    new pickler for the object when it is already being pickled by a
    `NumbaPickler`.
    """

    __slots__ = 'ctor', 'states'

    def __init__(self, ctor, states):
        self.ctor = ctor
        self.states = states

    def _reduce(self):
        return _CustomPickled._rebuild, (self.ctor, self.states)

    @classmethod
    def _rebuild(cls, ctor, states):
        return cls(ctor, states)


def _unpickle__CustomPickled(serialized):
    """standard unpickling for `_CustomPickled`.

    Uses `NumbaPickler` to load.
    """
    ctor, states = loads(serialized)
    return _CustomPickled(ctor, states)


def _pickle__CustomPickled(cp):
    """standard pickling for `_CustomPickled`.

    Uses `NumbaPickler` to dump.
    """
    serialized = dumps((cp.ctor, cp.states))
    return _unpickle__CustomPickled, (serialized,)


# Register custom pickling for the standard pickler.
copyreg.pickle(_CustomPickled, _pickle__CustomPickled)


def custom_reduce(cls, states):
    """For customizing object serialization in `__reduce__`.

    Object states provided here are used as keyword arguments to the
    `._rebuild()` class method.

    Parameters
    ----------
    states : dict
        Dictionary of object states to be serialized.

    Returns
    -------
    result : tuple
        This tuple conforms to the return type requirement for `__reduce__`.
    """
    return custom_rebuild, (_CustomPickled(cls, states),)


def custom_rebuild(custom_pickled):
    """Customized object deserialization.

    This function is referenced internally by `custom_reduce()`.
    """
    cls, states = custom_pickled.ctor, custom_pickled.states
    return cls._rebuild(**states)


def is_serialiable(obj):
    """Check if *obj* can be serialized.

    Parameters
    ----------
    obj : object

    Returns
    --------
    can_serialize : bool
    """
    with io.BytesIO() as fout:
        pickler = NumbaPickler(fout)
        try:
            pickler.dump(obj)
        except pickle.PicklingError:
            return False
        else:
            return True


class _TracedPicklingError(pickle.PicklingError):
    """A custom pickling error used internally in NumbaPickler.
    """
    pass


class SlowNumbaPickler(pickle._Pickler):
    """Extends the pure-python Pickler to support the pickling need in Numba.

    Adds pickling for closure functions, modules.
    Adds customized pickling for _CustomPickled to avoid invoking a new
    Pickler instance.

    Note: this is used on Python < 3.8 unless `pickle5` is installed.

    Note: This is good for debugging because the C-pickler hides the traceback
    """

    dispatch = pickle._Pickler.dispatch.copy()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__trace = []
        self.__memo = {}

    def save(self, obj):
        """
        Overrides parent's `.save()` to provide better error messages.
        """
        self.__trace.append(f"{type(obj)}: {id(obj)}" )
        try:
            return super().save(obj)
        except _TracedPicklingError:
            raise
        except Exception as e:
            def perline(items):
                return '\n'.join(f" [{depth}]: {it}"
                                 for depth, it in enumerate(items))
            m = (f"Failed to pickle because of\n  {type(e).__name__}: {e}"
                 f"\ntracing... \n{perline(self.__trace)}")
            raise _TracedPicklingError(m)
        finally:
            self.__trace.pop()

    def _save_function(self, func):
        """
        Override how functions are saved by serializing the bytecode and
        corresponding globals.
        """
        if _is_importable(func):
            return self.save_global(func)
        if id(func) in self.__memo:
            # Detect recursive pickling; i.e. function references itself.
            # NOTE: this is not ideal, but we prefer the fast pickler which
            #       does this properly.
            msg = f"Recursive function reference on {func}"
            raise pickle.PicklingError(msg)
        self.__memo[id(func)] = func
        try:
            gls = _get_function_globals_for_reduction(func)
            args = _reduce_function(func, gls)
            self.save_reduce(_rebuild_function, args, obj=func)
        finally:
            self.__memo.pop(id(func))

    # Override the dispatch table for functions
    dispatch[FunctionType] = _save_function

    def _save_module(self, mod):
        return self.save_reduce(_rebuild_module, (mod.__name__,))

    # Override the dispatch table for modules
    dispatch[ModuleType] = _save_module

    def _save_custom_pickled(self, cp):
        return self.save_reduce(*cp._reduce())

    # Override the dispatch table for _CustomPickled
    # thus override the copyreg registry
    dispatch[_CustomPickled] = _save_custom_pickled


class FastNumbaPickler(pickle.Pickler):
    """Faster version of the NumbaPickler through use of Python3.8+ features from
    the C Pickler, install `pickle5` for similar on older Python versions.
    """
    def reducer_override(self, obj):
        if isinstance(obj, FunctionType):
            return self._custom_reduce_func(obj)
        elif isinstance(obj, ModuleType):
            return self._custom_reduce_module(obj)
        elif isinstance(obj, CodeType):
            return self._custom_reduce_code(obj)
        elif isinstance(obj, _CustomPickled):
            return self._custom_reduce__custompickled(obj)
        return NotImplemented

    def _custom_reduce_func(self, func):
        if not _is_importable(func):
            gls = _get_function_globals_for_reduction(func)
            args, cells = _reduce_function_no_cells(func, gls)
            states = {'cells': cells}
            return (_rebuild_function, args, states, None, None,
                    _function_setstate)
        else:
            return NotImplemented

    def _custom_reduce_module(self, mod):
        return _rebuild_module, (mod.__name__,)

    def _custom_reduce_code(self, code):
        return _reduce_code(code)

    def _custom_reduce__custompickled(self, cp):
        return cp._reduce()


# Pick our preferred pickler
NumbaPickler = FastNumbaPickler if pickle38 else SlowNumbaPickler


class ReduceMixin(abc.ABC):
    """A mixin class for objects that should be reduced by the NumbaPickler instead
    of the standard pickler.
    """
    # Subclass MUST override the below methods

    @abc.abstractmethod
    def _reduce_states(self):
        raise NotImplementedError

    @abc.abstractclassmethod
    def _rebuild(cls, **kwargs):
        raise NotImplementedError

    # Subclass can override the below methods

    def _reduce_class(self):
        return self.__class__

    # Private methods

    def __reduce__(self):
        return custom_reduce(self._reduce_class(), self._reduce_states())


# ----------------------------------------------------------------------------
# The following code is adapted from cloudpickle as of
# https://github.com/cloudpipe/cloudpickle/commit/9518ae3cc71b7a6c14478a6881c0db41d73812b8    # noqa: E501
# Please see LICENSE.third-party file for full copyright information.

def _is_importable(obj):
    """Check if an object is importable.

    Parameters
    ----------
    obj :
        Must define `__module__` and `__qualname__`.
    """
    if obj.__module__ in sys.modules:
        ptr = sys.modules[obj.__module__]
        # Walk through the attributes
        parts = obj.__qualname__.split('.')
        if len(parts) > 1:
            # can't deal with function insides classes yet
            return False
        for p in parts:
            try:
                ptr = getattr(ptr, p)
            except AttributeError:
                return False
        return obj is ptr
    return False


def _function_setstate(obj, states):
    """The setstate function is executed after creating the function instance
    to add `cells` into it.
    """
    cells = states.pop('cells')
    for i, v in enumerate(cells):
        _cell_set(obj.__closure__[i], v)
    return obj


def _reduce_function_no_cells(func, globs):
    """_reduce_function() but return empty cells instead.

    """
    if func.__closure__:
        oldcells = [cell.cell_contents for cell in func.__closure__]
        cells = [None for _ in range(len(oldcells))] # idea from cloudpickle
    else:
        oldcells = ()
        cells = None
    rebuild_args = (_reduce_code(func.__code__), globs, func.__name__, cells,
                    func.__defaults__)
    return rebuild_args, oldcells


def _cell_rebuild(contents):
    """Rebuild a cell from cell contents
    """
    if contents is None:
        return CellType()
    else:
        return CellType(contents)


def _cell_reduce(obj):
    """Reduce a CellType
    """
    try:
        # .cell_contents attr lookup raises the wrong exception?
        obj.cell_contents
    except ValueError:
        # empty cell
        return _cell_rebuild, (None,)
    else:
        # non-empty cell
        return _cell_rebuild, (obj.cell_contents,)


def _cell_set(cell, value):
    """Set *value* into *cell* because `.cell_contents` is not writable
    before python 3.7.

    See https://github.com/cloudpipe/cloudpickle/blob/9518ae3cc71b7a6c14478a6881c0db41d73812b8/cloudpickle/cloudpickle.py#L298   # noqa: E501
    """
    if PYVERSION >= (3, 7):  # pragma: no branch
        cell.cell_contents = value
    else:
        _cell_set = FunctionType(
            _cell_set_template_code, {}, '_cell_set', (), (cell,),)
        _cell_set(value)


def _make_cell_set_template_code():
    """See _cell_set"""
    def _cell_set_factory(value):
        lambda: cell
        cell = value

    co = _cell_set_factory.__code__

    _cell_set_template_code = CodeType(
        co.co_argcount,
        co.co_kwonlyargcount,
        co.co_nlocals,
        co.co_stacksize,
        co.co_flags,
        co.co_code,
        co.co_consts,
        co.co_names,
        co.co_varnames,
        co.co_filename,
        co.co_name,
        co.co_firstlineno,
        co.co_lnotab,
        co.co_cellvars,  # co_freevars is initialized with co_cellvars
        (),  # co_cellvars is made empty
    )
    return _cell_set_template_code


if PYVERSION < (3, 7):
    _cell_set_template_code = _make_cell_set_template_code()

# End adapting from cloudpickle
# ----------------------------------------------------------------------------
