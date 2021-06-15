import abc

from numba.core import errors


class RetargetCache:
    """Cache for retargeted dispatcher.

    The key is the original dispatcher.
    """
    container_type = dict

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
        """Retarget the givern dispatcher and returns a new dispatcher-like
        callable. Or, return the original dispatcher if the the target_backend
        will not change.
        """
        pass


class BasicRetarget(BaseRetarget):
    """Retarget to a new backend.
    """
    def __init__(self):
        self.cache = RetargetCache()

    @abc.abstractproperty
    def output_target(self):
        """This is used in `.check_compatible()`
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
                m = f"{output_target} != {required_target}"
                raise errors.CompilerError(m)

    def retarget(self, orig_disp):
        cache = self.cache
        cached = cache.load_cache(orig_disp)
        opts = orig_disp.targetoptions
        if opts.get('target_backend') == self.output_target:
            return orig_disp
        if cached is None:
            out = self.compile_retarget(orig_disp)
            cache.save_cache(orig_disp, out)
        else:
            out = cached
        return out
