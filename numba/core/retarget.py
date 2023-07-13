"""
Implement utils for supporting retargeting of dispatchers.

WARNING: Features defined in this file are experimental. The API may change
         without notice.
"""
import abc
import weakref

from numba.core import errors


class RetargetCache:
    """Cache for retargeted dispatchers.

    The cache uses the original dispatcher as the key.
    """
    container_type = weakref.WeakKeyDictionary

    def __init__(self):
        self._cache = self.container_type()
        self._stat_hit = 0
        self._stat_miss = 0

    def save_cache(self, orig_disp, new_disp):
        """Save a dispatcher associated with the given key.
        """
        self._cache[orig_disp] = new_disp

    def load_cache(self, orig_disp):
        """Load a dispatcher associated with the given key.
        """
        out = self._cache.get(orig_disp)
        if out is None:
            self._stat_miss += 1
        else:
            self._stat_hit += 1
        return out

    def items(self):
        """Returns the contents of the cache.
        """
        return self._cache.items()

    def stats(self):
        """Returns stats regarding cache hit/miss.
        """
        return {'hit': self._stat_hit, 'miss': self._stat_miss}


class BaseRetarget(abc.ABC):
    """Abstract base class for retargeting logic.
    """
    @abc.abstractmethod
    def check_compatible(self, orig_disp):
        """Check that the retarget is compatible.

        This method does not return anything meaningful (e.g. None)
        Incompatibility is signalled via raising an exception.
        """
        pass

    @abc.abstractmethod
    def retarget(self, orig_disp):
        """Retargets the given dispatcher and returns a new dispatcher-like
        callable. Or, returns the original dispatcher if the the target_backend
        will not change.
        """
        pass


class BasicRetarget(BaseRetarget):
    """A basic retargeting implementation for a single output target.

    This class has two abstract methods/properties that subclasses must define.

    - `output_target` must return output target name.
    - `compile_retarget` must define the logic to retarget the given dispatcher.

    By default, this class uses `RetargetCache` as the internal cache. This
    can be modified by overriding the `.cache_type` class attribute.

    """
    cache_type = RetargetCache

    def __init__(self):
        self.cache = self.cache_type()

    @abc.abstractproperty
    def output_target(self) -> str:
        """Returns the output target name.

        See numba/tests/test_retargeting.py for example usage.
        """
        pass

    @abc.abstractmethod
    def compile_retarget(self, orig_disp):
        """Returns the retargeted dispatcher.

        See numba/tests/test_retargeting.py for example usage.
        """
        pass

    def check_compatible(self, orig_disp):
        """
        This implementation checks that
        `self.output_target == orig_disp._required_target_backend`
        """
        required_target = orig_disp._required_target_backend
        output_target = self.output_target
        if required_target is not None:
            if output_target != required_target:
                m = ("The output target does match the required target: "
                     f"{output_target} != {required_target}.")
                raise errors.CompilerError(m)

    def retarget(self, orig_disp):
        """Apply retargeting to orig_disp.

        The retargeted dispatchers are cached for future use.
        """
        cache = self.cache
        opts = orig_disp.targetoptions
        # Skip if the original dispatcher is targeting the same output target
        if opts.get('target_backend') == self.output_target:
            return orig_disp
        cached = cache.load_cache(orig_disp)
        # No cache?
        if cached is None:
            out = self.compile_retarget(orig_disp)
            cache.save_cache(orig_disp, out)
        else:
            out = cached
        return out
