import warnings

from numba.core import types, config, errors
from numba.experimental.jitclass.base import register_class_type, ClassBuilder


def jitclass(cls_or_spec=None, spec=None):
    """
    A function for creating a jitclass.
    Can be used as a decorator or function.

    Different use cases will cause different arguments to be set.

    1) cls_or_spec = None, spec = None

        @jitclass()
        class Foo:
            ...

    2) cls_or_spec = None, spec = spec

        @jitclass(spec=spec)
        class Foo:
            ...

    3) cls_or_spec = Foo, spec = None

        @jitclass
        class Foo:
            ...

    4) cls_or_spec = spec, spec = None

        @jitclass(spec)
        class Foo:
            ...

        In this case we update `cls_or_spec, spec = None, cls_or_spec`.

    5) cls_or_spec = Foo, spec = spec

        JitFoo = jitclass(Foo, spec)

    Args
    ----
    spec:
        Specifies the types of class fields.
        Must be a dictionary or sequence.
        With a dictionary, use collections.OrderedDict for stable ordering.
        With a sequence, it must contain 2-tuples of (fieldname, fieldtype).

        Any class annotations for field names not listed in spec will be added.
        For class annotation
            x: T
        we will append
            ("x", numba.typeof(T))
        to spec if x is not already a key in spec.

    Returns
    -------
    If used as a decorator, returns a callable that takes a class object and
    returns a compiled version.
    If used as a function, returns the compiled class.
    """

    if (cls_or_spec is not None and
        spec is None and
            not isinstance(cls_or_spec, type)):
        # Used like
        # @jitclass([("x", intp)])
        # class Foo:
        #     ...
        spec = cls_or_spec
        cls_or_spec = None

    def wrap(cls):
        if config.DISABLE_JIT:
            return cls
        else:
            return register_class_type(cls, spec, types.ClassType, ClassBuilder)

    if cls_or_spec is None:
        return wrap
    else:
        return wrap(cls_or_spec)


def _warning_jitclass(spec):
    """
    Re-export of numba.experimental.jitclass with a warning.
    To be used in numba/__init__.py.
    This endpoint is deprecated.
    """
    url = ("http://numba.pydata.org/numba-doc/latest/reference/"
           "deprecation.html#change-of-jitclass-location")

    msg = ("The 'numba.jitclass' decorator has moved to "
           "'numba.experimental.jitclass' to better reflect the experimental "
           "nature of the functionality. Please update your imports to "
           "accommodate this change and see {} for the time frame.".format(url))

    warnings.warn(msg, category=errors.NumbaDeprecationWarning,
                  stacklevel=2)

    return jitclass(spec=spec)
