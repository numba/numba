from __future__ import print_function, division, absolute_import

from .. import types
from .templates import (AttributeTemplate, ConcreteTemplate,
                        AbstractTemplate, builtin_global, builtin,
                        builtin_attr, signature, bound_function)
from .builtins import normalize_1d_index


# NOTE: "in" and "len" are defined on all sized containers, but we have
# no need for a more fine-grained hierarchy right now.

@builtin
class InSequence(AbstractTemplate):
    key = "in"

    def generic(self, args, kws):
        item, seq = args
        if isinstance(seq, types.Sequence):
            return signature(types.boolean, seq.dtype, seq)

@builtin
class SequenceLen(AbstractTemplate):
    key = types.len_type

    def generic(self, args, kws):
        assert not kws
        (val,) = args
        if isinstance(val, (types.Sequence)):
            return signature(types.intp, val)

@builtin
class SequenceBool(AbstractTemplate):
    key = "is_true"

    def generic(self, args, kws):
        assert not kws
        (val,) = args
        if isinstance(val, (types.Sequence)):
            return signature(types.boolean, val)

@builtin
class GetItemSequence(AbstractTemplate):
    key = "getitem"

    def generic(self, args, kws):
        seq, idx = args
        if isinstance(seq, types.Sequence):
            idx = normalize_1d_index(idx)
            if idx == types.slice3_type:
                return signature(seq, seq, idx)
            elif isinstance(idx, types.Integer):
                return signature(seq.dtype, seq, idx)

@builtin
class SetItemSequence(AbstractTemplate):
    key = "setitem"

    def generic(self, args, kws):
        seq, idx, value = args
        if isinstance(seq, types.MutableSequence):
            idx = normalize_1d_index(idx)
            if idx == types.slice3_type:
                return signature(types.none, seq, idx, seq)
            elif isinstance(idx, types.Integer):
                return signature(types.none, seq, idx, seq.dtype)

@builtin
class DelItemSequence(AbstractTemplate):
    key = "delitem"

    def generic(self, args, kws):
        seq, idx = args
        if isinstance(seq, types.MutableSequence):
            idx = normalize_1d_index(idx)
            return signature(types.none, seq, idx)
