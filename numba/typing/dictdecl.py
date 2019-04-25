"""
This implements the typing template for `dict()`.
"""
from __future__ import absolute_import, print_function

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
The only supported use is `dict()`.
""".strip()


@infer_global(dict)
class DictBuiltin(AbstractTemplate):
    def generic(self, args, kws):
        if args or kws:
            raise errors.TypingError(_message_dict_support)
        return signature(types.DictType(types.undefined, types.undefined))
