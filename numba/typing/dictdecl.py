from __future__ import absolute_import, print_function

from .. import types
from .templates import (
    AbstractTemplate,
    Registry,
    signature,
)

registry = Registry()
infer = registry.register
infer_global = registry.register_global
infer_getattr = registry.register_attr


@infer_global(dict)
class DictBuiltin(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert not args
        return signature(types.DictType(types.undefined, types.undefined))
