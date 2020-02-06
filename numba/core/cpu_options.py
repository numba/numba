"""
Defines CPU Options for use in the CPU target
"""


class FastMathOptions(object):
    """
    Options for controlling fast math optimization.
    """

    def __init__(self, value):
        # https://releases.llvm.org/7.0.0/docs/LangRef.html#fast-math-flags
        valid_flags = {
            'fast',
            'nnan', 'ninf', 'nsz', 'arcp',
            'contract', 'afn', 'reassoc',
        }

        if value is True:
            self.flags = {'fast'}
        elif value is False:
            self.flags = set()
        elif isinstance(value, set):
            invalid = value - valid_flags
            if invalid:
                raise ValueError("Unrecognized fastmath flags: %s" % invalid)
            self.flags = value
        elif isinstance(value, dict):
            invalid = set(value.keys()) - valid_flags
            if invalid:
                raise ValueError("Unrecognized fastmath flags: %s" % invalid)
            self.flags = {v for v, enable in value.items() if enable}
        else:
            msg = "Expected fastmath option(s) to be either a bool, dict or set"
            raise ValueError(msg)

    def __bool__(self):
        return bool(self.flags)

    __nonzero__ = __bool__


class ParallelOptions(object):
    """
    Options for controlling auto parallelization.
    """

    def __init__(self, value):
        if isinstance(value, bool):
            self.enabled = value
            self.comprehension = value
            self.reduction = value
            self.setitem = value
            self.numpy = value
            self.stencil = value
            self.fusion = value
            self.prange = value
        elif isinstance(value, dict):
            self.enabled = True
            self.comprehension = value.pop('comprehension', True)
            self.reduction = value.pop('reduction', True)
            self.setitem = value.pop('setitem', True)
            self.numpy = value.pop('numpy', True)
            self.stencil = value.pop('stencil', True)
            self.fusion = value.pop('fusion', True)
            self.prange = value.pop('prange', True)
            if value:
                msg = "Unrecognized parallel options: %s" % value.keys()
                raise NameError(msg)
        else:
            msg = "Expect parallel option to be either a bool or a dict"
            raise ValueError(msg)


class InlineOptions(object):
    """
    Options for controlling inlining
    """

    def __init__(self, value):
        ok = False
        if isinstance(value, str):
            if value in ('always', 'never'):
                ok = True
        else:
            ok = hasattr(value, '__call__')

        if ok:
            self._inline = value
        else:
            msg = ("kwarg 'inline' must be one of the strings 'always' or "
                   "'never', or it can be a callable that returns True/False. "
                   "Found value %s" % value)
            raise ValueError(msg)

    @property
    def is_never_inline(self):
        """
        True if never inline
        """
        return self._inline == 'never'

    @property
    def is_always_inline(self):
        """
        True if always inline
        """
        return self._inline == 'always'

    @property
    def has_cost_model(self):
        """
        True if a cost model is provided
        """
        return not (self.is_always_inline or self.is_never_inline)

    @property
    def value(self):
        """
        The raw value
        """
        return self._inline
