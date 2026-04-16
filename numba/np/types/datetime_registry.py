from numba.core.pythonapi import box, unbox, NativeValue
from numba.np.types import NPDatetime, NPTimedelta
from numba.core.extending import overload
from numba.cpython.builtins import cast_int
from numba.core import errors


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


@overload(int)
def ol_int(x):
    if not isinstance(x, (NPDatetime, NPTimedelta)):
        return

    if not (isinstance(x, NPDatetime) and x.unit == 'ns'):
        raise errors.NumbaTypeError(
            "Only datetime64[ns] can be converted,"
            f" but got datetime64[{x.unit}]"
        )

    def impl(x):
        return cast_int(x)

    return impl
