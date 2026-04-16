from numba.core.pythonapi import box, unbox, NativeValue
from numba.np.types import NPDatetime, NPTimedelta, UniTuple, BaseTuple
from numba.core.extending import overload
from numba.cpython.builtins import max_vararg, min_vararg


@box(NPDatetime)
def box_npdatetime(typ, val, c):
    return c.pyapi.create_np_datetime(val, typ.unit_code)


@unbox(NPDatetime)
def unbox_npdatetime(typ, obj, c):
    val = c.pyapi.extract_np_datetime(obj)
    return NativeValue(val, is_error=c.pyapi.c_api_error())


@box(NPTimedelta)
def box_nptimedelta(typ, val, c):
    return c.pyapi.create_np_timedelta(val, typ.unit_code)


@unbox(NPTimedelta)
def unbox_nptimedelta(typ, obj, c):
    val = c.pyapi.extract_np_timedelta(obj)
    return NativeValue(val, is_error=c.pyapi.c_api_error())


@overload(max)
def ol_max(*x):
    if len(x) == 1 and (
        (isinstance(x[0], UniTuple) and isinstance(
            x[0].dtype, (NPDatetime, NPTimedelta)))

        or (isinstance(x[0], BaseTuple) and all(
            isinstance(ty, (NPDatetime, NPTimedelta)) for ty in x[0].types))
    ):
        def impl(*x):
            return max_vararg(x[0])
        return impl
    else:
        for ty in x:
            if not isinstance(ty, (NPDatetime, NPTimedelta)):
                return None

        def impl(*x):
            return max_vararg(x)
        return impl


@overload(min)
def ol_min(*x):
    if len(x) == 1 and (
        (isinstance(x[0], UniTuple) and isinstance(
            x[0].dtype, (NPDatetime, NPTimedelta)))

        or (isinstance(x[0], BaseTuple) and all(
            isinstance(ty, (NPDatetime, NPTimedelta)) for ty in x[0].types))
    ):
        def impl(*x):
            return min_vararg(x[0])
        return impl
    else:
        for ty in x:
            if not isinstance(ty, (NPDatetime, NPTimedelta)):
                return None

        def impl(*x):
            return min_vararg(x)
        return impl
