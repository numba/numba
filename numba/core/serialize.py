"""
Serialization support for compiled functions.
"""
import abc
import io
import copyreg


import pickle
from cloudpickle import cloudpickle_fast

cloudpickle = cloudpickle_fast


#
# Pickle support
#

def _rebuild_reduction(cls, *args):
    """
    Global hook to rebuild a given class from its __reduce__ arguments.
    """
    return cls._rebuild(*args)


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
loads = cloudpickle.loads


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


class NumbaPickler(cloudpickle_fast.CloudPickler):
    pass


def _custom_reduce__custompickled(cp):
    return cp._reduce()


NumbaPickler._dispatch_table[_CustomPickled] = _custom_reduce__custompickled


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
