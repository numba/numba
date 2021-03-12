from types import MappingProxyType


class Option:
    __slots__ = "_type", "_default", "_doc"

    def __init__(self, type, *, default, doc):
        self._type = type
        self._default = default
        self._doc = doc

    @property
    def type(self):
        return self._type

    @property
    def default(self):
        return self._default

    @property
    def doc(self):
        return self._doc


class _MetaTargetConfig(type):
    def __init__(cls, name, bases, dct):
        # Gather options from base classes and class dict
        opts = {}
        # Reversed scan into the base classes to follow MRO ordering such that
        # the closest base class is overriding
        for base_cls in reversed(bases):
            opts.update(base_cls.options)
        opts.update(cls.find_options(dct))
        cls.options = MappingProxyType(opts)

        # Make properties for each of the options
        def make_prop(name, option):
            def getter(self):
                return self._values.get(name, option.default)

            def setter(self, val):
                self._values[name] = option.type(val)

            def delter(self):
                del self._values[name]

            return property(getter, setter, delter, option.doc)

        for name, option in cls.options.items():
            setattr(cls, name, make_prop(name, option))

    def find_options(cls, dct):
        """Returns a new dict with all the items that is mapping to a
        ``Option``.
        """
        return {k: v for k, v in dct.items() if isinstance(v, Option)}


class TargetConfig(metaclass=_MetaTargetConfig):
    def __init__(self, copy_from=None):
        self._values = {}
        if copy_from is not None:
            assert isinstance(copy_from, TargetConfig)
            self._values.update(copy_from._values)

    def __repr__(self):
        # NOTE: defaulted options will be placed in the back and grouped inside
        #       a square bracket; i.e. [optname=optval, ...]
        args = []
        defs = []
        for k in self.options:
            msg = f"{k}={getattr(self, k)}"
            if not self.is_set(k):
                defs.append(msg)
            else:
                args.append(msg)
        clsname = self.__class__.__name__
        return f"{clsname}({', '.join(args)}, [{', '.join(defs)}])"

    def __hash__(self):
        return hash(tuple(sorted(self.values())))

    def __eq__(self, other):
        if isinstance(other, TargetConfig):
            return self.values() == other.values()
        else:
            return NotImplemented

    def __ne__(self, other):
        eq = self.__eq__(other)
        if eq is NotImplemented:
            return NotImplemented
        else:
            return eq

    def values(self):
        """Returns a dict of all the values
        """
        return {k: getattr(self, k) for k in self.options}

    def is_set(self, name):
        """Is the option set?
        """
        self._guard_option(name)
        return name in self._values

    def discard(self, name):
        self._guard_option(name)
        self._values.pop(name, None)

    def copy(self):
        return type(self)(self)

    def summary(self):
        args = []
        for k in self.options:
            msg = f"{k}={getattr(self, k)}"
            if self.is_set(k):
                args.append(msg)
        clsname = self.__class__.__name__
        return f"{clsname}({', '.join(args)})"

    def _guard_option(self, name):
        if name not in self.options:
            msg = f"{name!r} is not a valid option for {type(self)}"
            raise ValueError(msg)
