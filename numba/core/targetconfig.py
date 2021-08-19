"""
This module contains utils for manipulating target configurations such as
compiler flags.
"""

from types import MappingProxyType
from numba.core import utils


class Option:
    """An option to be used in ``TargetConfig``.
    """
    __slots__ = "_type", "_default", "_doc"

    def __init__(self, type, *, default, doc):
        """
        Parameters
        ----------
        type :
            Type of the option value. It can be a callable.
            The setter always calls ``self._type(value)``.
        default :
            The default value for the option.
        doc : str
            Docstring for the option.
        """
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
    """Metaclass for ``TargetConfig``.

    When a subclass of ``TargetConfig`` is created, all ``Option`` defined
    as class members will be parsed and corresponding getters, setters, and
    delters will be inserted.
    """
    def __init__(cls, name, bases, dct):
        """Invoked when subclass is created.

        Insert properties for each ``Option`` that are class members.
        All the options will be grouped inside the ``.options`` class
        attribute.
        """
        # Gather options from base classes and class dict
        opts = {}
        # Reversed scan into the base classes to follow MRO ordering such that
        # the closest base class is overriding
        for base_cls in reversed(bases):
            opts.update(base_cls.options)
        opts.update(cls.find_options(dct))
        # Store the options into class attribute as a ready-only mapping.
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
        """Returns a new dict with all the items that are a mapping to an
        ``Option``.
        """
        return {k: v for k, v in dct.items() if isinstance(v, Option)}


class _NotSetType:
    def __repr__(self):
        return "<NotSet>"


_NotSet = _NotSetType()


class TargetConfig(metaclass=_MetaTargetConfig):
    """Base class for ``TargetConfig``.

    Subclass should fill class members with ``Option``. For example:

    >>> class MyTargetConfig(TargetConfig):
    >>>     a_bool_option = Option(type=bool, default=False, doc="a bool")
    >>>     an_int_option = Option(type=int, default=0, doc="an int")

    The metaclass will insert properties for each ``Option``. For exapmle:

    >>> tc = MyTargetConfig()
    >>> tc.a_bool_option = True  # invokes the setter
    >>> print(tc.an_int_option)  # print the default
    """
    def __init__(self, copy_from=None):
        """
        Parameters
        ----------
        copy_from : TargetConfig or None
            if None, creates an empty ``TargetConfig``.
            Otherwise, creates a copy.
        """
        self._values = {}
        if copy_from is not None:
            assert isinstance(copy_from, TargetConfig)
            self._values.update(copy_from._values)

    def __repr__(self):
        # NOTE: default options will be placed at the end and grouped inside
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
        """Remove the option by name if it is defined.

        After this, the value for the option will be set to its default value.
        """
        self._guard_option(name)
        self._values.pop(name, None)

    def inherit_if_not_set(self, name, default=_NotSet):
        """Inherit flag from ``ConfigStack``.

        Parameters
        ----------
        name : str
            Option name.
        default : optional
            When given, it overrides the default value.
            It is only used when the flag is not defined locally and there is
            no entry in the ``ConfigStack``.
        """
        self._guard_option(name)
        if not self.is_set(name):
            cstk = utils.ConfigStack()
            if cstk:
                # inherit
                top = cstk.top()
                setattr(self, name, getattr(top, name))
            elif default is not _NotSet:
                setattr(self, name, default)

    def copy(self):
        """Clone this instance.
        """
        return type(self)(self)

    def summary(self):
        """Returns a ``str`` that summarizes this instance.

        In contrast to ``__repr__``, only options that are explicitly set will
        be shown.
        """
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
