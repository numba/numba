from __future__ import annotations

import typing as pt

from numba.core import types, config
from numba.type_hints import ClassSpecType, FuncSpecType

C = pt.TypeVar("C", bound=pt.Callable)
T = pt.TypeVar("T", bound=pt.Type)

if pt.TYPE_CHECKING:
    from numba.experimental.jitclass.base import JitMethod


@pt.overload
def jitmethod(
    func_or_spec: pt.Optional[FuncSpecType] = None,
    spec: pt.Optional[FuncSpecType] = None,
    **njit_options: bool
) -> pt.Callable[[C], JitMethod[C]]:
    ...


@pt.overload
def jitmethod(  # noqa: F811
    func_or_spec: C,
    spec: pt.Optional[FuncSpecType] = None,
    **njit_options: bool
) -> JitMethod[C]:
    ...


def jitmethod(  # noqa: F811
    func_or_spec: pt.Union[C, FuncSpecType, None] = None,
    spec: pt.Optional[FuncSpecType] = None,
    **njit_options: bool,
) -> pt.Union[JitMethod[C], pt.Callable[[C], JitMethod[C]]]:
    """"""
    if (
        func_or_spec is not None
        and spec is None
        and not callable(func_or_spec)
    ):
        # Used like
        # @jitmethod([("x", intp)])
        # def foo():
        #     ...
        spec = pt.cast(pt.Optional[FuncSpecType], func_or_spec)
        func_or_spec = None

    def wrap(func: C) -> JitMethod[C]:
        from numba.experimental.jitclass.base import JitMethod

        return JitMethod(implementation=func, njit_options=njit_options)

    if func_or_spec is None:
        return wrap
    else:
        return wrap(pt.cast(C, func_or_spec))


@pt.overload
def jitclass(  # noqa: F811
    cls_or_spec: pt.Optional[ClassSpecType] = None,
    spec: pt.Optional[ClassSpecType] = None,
    **njit_options: bool
) -> pt.Callable[[T], T]:
    ...


@pt.overload
def jitclass(  # noqa: F811
    cls_or_spec: T,
    spec: pt.Optional[ClassSpecType] = None,
    **njit_options: bool
) -> T:
    ...


def jitclass(  # noqa: F811
    cls_or_spec: pt.Union[T, ClassSpecType, None] = None,
    spec: pt.Optional[ClassSpecType] = None,
    **njit_options: bool
) -> pt.Union[T, pt.Callable[[T], T]]:
    """
    A function for creating a jitclass.
    Can be used as a decorator or function.

    Different use cases will cause different arguments to be set.

    If specified, ``spec`` gives the types of class fields.
    It must be a dictionary or sequence.
    With a dictionary, use collections.OrderedDict for stable ordering.
    With a sequence, it must contain 2-tuples of (fieldname, fieldtype).

    Any class annotations for field names not listed in spec will be added.
    For class annotation `x: T` we will append ``("x", as_numba_type(T))`` to
    the spec if ``x`` is not already a key in spec.


    Examples
    --------

    1) ``cls_or_spec = None``, ``spec = None``

    >>> @jitclass()
    ... class Foo:
    ...     ...

    2) ``cls_or_spec = None``, ``spec = spec``

    >>> @jitclass(spec=spec)
    ... class Foo:
    ...     ...

    3) ``cls_or_spec = Foo``, ``spec = None``

    >>> @jitclass
    ... class Foo:
    ...     ...

    4) ``cls_or_spec = spec``, ``spec = None``
    In this case we update ``cls_or_spec, spec = None, cls_or_spec``.

    >>> @jitclass(spec)
    ... class Foo:
    ...     ...

    5) ``cls_or_spec = Foo``, ``spec = spec``

    >>> JitFoo = jitclass(Foo, spec)

    Returns
    -------
    If used as a decorator, returns a callable that takes a class object and
    returns a compiled version.
    If used as a function, returns the compiled class (an instance of
    ``JitClassType``).
    """

    if (
        cls_or_spec is not None
        and spec is None
        and not isinstance(cls_or_spec, type)
    ):
        # Used like
        # @jitclass([("x", intp)])
        # class Foo:
        #     ...
        spec = cls_or_spec
        cls_or_spec = None

    def wrap(cls: T) -> T:
        if config.DISABLE_JIT:  # type: ignore[attr-defined]
            return cls
        else:
            from numba.experimental.jitclass.base import (
                register_class_type,
                ClassBuilder,
            )

            cls_jitted = register_class_type(
                cls, spec, types.ClassType, ClassBuilder, njit_options
            )

            # Preserve the module name of the original class
            cls_jitted.__module__ = cls.__module__

            return pt.cast(T, cls_jitted)

    if cls_or_spec is None:
        return wrap
    else:
        return wrap(pt.cast(T, cls_or_spec))
