import numpy as np

from numba.core import types, errors
from numba.core.typing.templates import (
    infer_getattr, AttributeTemplate, make_callable_template
)
from numba.core.imputils import lower_builtin, impl_ret_untracked
from numba.core.types import Callable, DTypeSpec, Opaque


class NumberClass(Callable, DTypeSpec, Opaque):
    """
    Type class for number classes (e.g. "np.float64").
    """

    def __init__(self, instance_type):
        self.instance_type = instance_type
        name = "class(%s)" % (instance_type,)
        super(NumberClass, self).__init__(name)

    def get_call_type(self, context, args, kws):
        # Overridden by the __call__ constructor resolution in typing.builtins
        return None

    def get_call_signatures(self):
        return (), True

    def get_impl_key(self, sig):
        return type(self)

    @property
    def key(self):
        return self.instance_type

    @property
    def dtype(self):
        return self.instance_type


@lower_builtin(NumberClass, types.Any)
def number_constructor(context, builder, sig, args):
    """
    Call a number class, e.g. np.int32(...)
    """
    if isinstance(sig.return_type, types.Array):
        # Array constructor
        dt = sig.return_type.dtype

        def foo(*arg_hack):
            return np.array(arg_hack, dtype=dt)

        res = context.compile_internal(builder, foo, sig, args)
        return impl_ret_untracked(context, builder, sig.return_type, res)
    else:
        # Scalar constructor
        [val] = args
        [valty] = sig.args
        return context.cast(builder, val, valty, sig.return_type)


@infer_getattr
class NumberClassAttribute(AttributeTemplate):
    key = NumberClass

    def resolve___call__(self, classty):
        """
        Resolve a NumPy number class's constructor
        (e.g. calling numpy.int32(...))
        """
        ty = classty.instance_type

        def typer(val):
            # TODO: When we refactor NumberClass, we should move this logic
            # to the NumPy module. For now, we special case the datetime-like
            # types here.
            if isinstance(val, (types.BaseTuple, types.Sequence)):
                import numpy as np
                # Array constructor, e.g. np.int32([1, 2])
                fnty = self.context.resolve_value_type(np.array)
                sig = fnty.get_call_type(self.context, (val, types.DType(ty)),
                                         {})
                return sig.return_type
            elif isinstance(
                val, (types.Number, types.Boolean, types.IntEnumMember)
            ):
                # Scalar constructor, e.g. np.int32(42)
                return ty
            elif val.__class__.__name__ in ("NPDatetime", "NPTimedelta"):
                # Constructor cast from datetime-like, e.g.
                # > np.int64(np.datetime64("2000-01-01"))
                if ty.bitwidth == 64:
                    return ty
                else:
                    msg = (f"Cannot cast {val} to {ty} as {ty} is not 64 bits "
                           "wide.")
                    raise errors.TypingError(msg)
            else:
                if (
                    isinstance(val, types.Array) and val.ndim == 0 and
                    val.dtype == ty
                ):
                    # This is 0d array -> scalar degrading
                    return ty
                else:
                    # unsupported
                    msg = f"Casting {val} to {ty} directly is unsupported."
                    if isinstance(val, types.Array):
                        # array casts are supported a different way.
                        msg += f" Try doing '<array>.astype(np.{ty})' instead"
                    raise errors.TypingError(msg)

        return types.Function(make_callable_template(key=ty, typer=typer))
