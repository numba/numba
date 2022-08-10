"""
This implements the typing template for `dict()`.
"""

from .. import types, errors
from .templates import (
    AbstractTemplate,
    Registry,
    signature,
)

registry = Registry()
infer = registry.register
infer_global = registry.register_global
infer_getattr = registry.register_attr


_message_dict_support = """
Unsupported use of `dict()` with positional or keyword argument(s). \
The only supported uses are `dict()` or `dict(*iterable)`.
""".strip()


@infer_global(dict)
class DictBuiltin(AbstractTemplate):
    def generic(self, args, kws):
        if kws:
            raise errors.TypingError(_message_dict_support)
        if args:
            iterable, = args
            if isinstance(iterable, types.IterableType):
                dtype = iterable.iterator_type.yield_type
                if isinstance(dtype, types.UniTuple):
                    k = v = dtype.key[0]
                elif isinstance(dtype, types.Tuple):
                    k, v = dtype.key
                else:
                    raise errors.TypingError(_message_dict_support)
                return signature(types.DictType(k, v), iterable)
            else:
                raise errors.TypingError(_message_dict_support)
        return signature(types.DictType(types.undefined, types.undefined))
