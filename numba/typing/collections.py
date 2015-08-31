from __future__ import print_function, division, absolute_import

from .. import types, utils
from .templates import (AttributeTemplate, ConcreteTemplate, AbstractTemplate,
                        builtin_global, builtin, builtin_attr,
                        signature, bound_function, make_callable_template)
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


# --------------------------------------------------------------------------
# named tuples

@builtin_attr
class NamedTupleAttribute(AttributeTemplate):
    key = types.BaseNamedTuple

    def resolve___class__(self, tup):
        return types.NamedTupleClass(tup.instance_class)

    def generic_resolve(self, tup, attr):
        # Resolution of other attributes
        try:
            index = tup.fields.index(attr)
        except ValueError:
            return
        return tup[index]


@builtin_attr
class NamedTupleClassAttribute(AttributeTemplate):
    key = types.NamedTupleClass

    def resolve___call__(self, classty):
        """
        Resolve the named tuple constructor, aka the class's __call__ method.
        """
        instance_class = classty.instance_class
        pysig = utils.pysignature(instance_class)

        def typer(*args, **kws):
            # Fold keyword args
            try:
                bound = pysig.bind(*args, **kws)
            except TypeError as e:
                msg = "In '%s': %s" % (instance_class, e)
                e.args = (msg,)
                raise
            assert not bound.kwargs
            return types.BaseTuple.from_types(bound.args, instance_class)

        # Override the typer's pysig to match the namedtuple constructor's
        typer.pysig = pysig
        return types.Function(make_callable_template(self.key, typer))
