import functools

from numba import typeof, types, njit


def typed_tuple(cls_or_spec=None, spec=None):
    """
    A function for creating tuple subclasses that are typed.

    By enforcing the types of the elements, we can statically define the numba
    type of this class.  This makes for significantly faster dispatch when
    passing such a tuple to a jitted function from normal python code.

    >>> class Point(typing.NamedTuple):
    ...     x: int
    ...     y: float
    >>>
    >>> @typed_tuple
    ... class TypedPoint(typing.NamedTuple):
    ...     x: int
    ...     y: float
    >>>
    >>> @njit
    >>> def point_mag(p):
    ...     return (p.x**2 + p.y**2)**0.5
    >>>
    >>> p = Point(3, 4)
    >>> tp = TypedPoint(3, 4)
    >>>
    >>> point_mag(p), point_mag(tp)
    (5.0, 5.0)
    >>> %timeit point_mag(p)
    9.75 µs ± 356 ns per loop
    >>> $timeit point_mag(tp)
    535 ns ± 10.2 ns per loop

    It makes no difference when used inside a jitted function.

    >>> @njit
    ... def make_point(x, y):
    ...     return Point(x, y)
    >>>
    >>> @njit
    ... def make_typed_point(x, y):
    ...     return TypedPoint(x, y)
    >>>
    >>> make_point(3, 4), make_point(3, 4)
    (Point(x=3, y=4), TypedPoint(x=3, y=4.0))
    >>> %timeit make_point(3, 4)
    1.73 µs ± 72.6 ns per loop
    >>> %timeit make_typed_point(3, 4)
    1.85 µs ± 40.6 ns per loop


    Examples
    --------

    1) ``cls_or_spec = None``, ``spec = None``

    >>> @typed_tuple()
    ... class Point1(typing.NamedTuple):
    ...     x: int
    ...     y: float
    >>> Point1(0, 0)
    Point1(x=0, y=0.0)

    2) ``cls_or_spec = None``, ``spec = {"x": numba.int8}``

    >>> @typed_tuple(spec=dict(x=numba.int8))
    ... class Point2(typing.NamedTuple):
    ...     x: int
    ...     y: float
    >>> Point2(200, 1)
    Point2(x=-56, y=1.0)

    3) ``cls_or_spec = Point3``, ``spec = None``

    >>> @typed_tuple
    ... class Point3(typing.NamedTuple):
    ...     x: int
    ...     y: float
    >>> Point3(0, 0)
    Point3(x=0, y=0.0)

    4) ``cls_or_spec = Point4``,
       ``spec = {"x": numba.intp, "y": numba.float64}``

    >>> Point4 = typed_tuple(
    ...     collections.namedtuple("Point4", ("x", "y")),
    ...     [("x", numba.intp), ("y", numba.float64)],
    ... )
    >>> Point4(0, 0)
    Point4(x=0, y=0.0)

    5) ``cls_or_spec = {"x": numba.intp, "y": numba.float64}``,
       ``spec = None``

    >>> @typed_tuple(dict(x=numba.int8, y=numba.float64))
    ... class Point5(typing.NamedTuple):
    ...     x: int
    ...     y: float
    >>> Point5(2.718, 3.1415)
    Point5(x=2, y=3.1415)

    """
    if spec:
        spec = dict(spec)
    elif cls_or_spec and not issubclass(cls_or_spec, tuple):
        spec = dict(cls_or_spec)
        cls_or_spec = None
    else:
        spec = dict()

    _ctor_template = """
def ctor(cls, {args}):
    return _tuple_new(cls, typecheck_tuple(({args},)))
"""

    def add_typecheck(cls):
        assert issubclass(cls, tuple)

        # Get numba type for fields.
        annotations = getattr(cls, "__annotations__", dict())
        nb_types = []
        for f in cls._fields:
            if f in spec:
                nb_types.append(spec[f])
            elif f in annotations:
                nb_types.append(typeof(annotations[f]))
            else:
                raise RuntimeError(f"Field {f} has no spec or annotation")
        nb_types = tuple(nb_types)

        # Create a jit function that will enforce typing.
        raw_tuple_ty = types.BaseTuple.from_types(nb_types)
        typecheck_tuple = njit((raw_tuple_ty,))(lambda t: t)

        # Update cls.__new__ to use typecheck_tuple.
        ctor_args = ", ".join(cls._fields)
        ctor_source = _ctor_template.format(args=ctor_args)

        exec_namespace = {
            "__name__": f"typedtuple_{cls.__name__}",
            "_tuple_new": tuple.__new__,
            "typecheck_tuple": typecheck_tuple,
        }
        exec(ctor_source, exec_namespace)
        ctor = exec_namespace["ctor"]

        assignment_fields = functools.WRAPPER_ASSIGNMENTS + (
            "__annotations__",
            "__defaults__",
        )
        functools.update_wrapper(ctor, cls.__new__, assignment_fields)

        cls.__untyped_new__ = cls.__new__
        cls.__new__ = ctor
        cls._numba_type_ = types.BaseTuple.from_types(nb_types, cls)

        return cls

    if cls_or_spec is None:
        return add_typecheck
    else:
        return add_typecheck(cls_or_spec)
