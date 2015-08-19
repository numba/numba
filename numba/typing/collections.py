from __future__ import print_function, division, absolute_import

from .. import types
from .templates import (AttributeTemplate, ConcreteTemplate,
                        AbstractTemplate, CallableTemplate,
                        builtin_global, builtin,
                        builtin_attr, signature, bound_function)
from .builtins import normalize_1d_index
from numba import utils


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

    def resolve___call__(self, _classty):
        """
        Resolve the named tuple constructor, aka the class's __call__ method.
        """
        class Constructor(NamedTupleConstructor):
            classty = _classty
            key = _classty
        return types.Function(Constructor)


class NamedTupleConstructor(CallableTemplate):
    # The named tuple type class
    classty = None

    def generic(self):
        # Compute the pysig for the namedtuple constructor
        instance_class = self.classty.instance_class
        params = [utils.Parameter(name, kind=utils.Parameter.POSITIONAL_OR_KEYWORD)
                  for name in instance_class._fields]
        pysig = utils.Signature(params)

        def typer(*args, **kws):
            # Fold keyword args
            try:
                bound = pysig.bind(*args, **kws)
            except TypeError as e:
                msg = "In '%s': %s" % (instance_class, e)
                e.args = (msg,)
                raise
            tys = bound.args
            # In sync with typing.typeof
            # XXX dedup using a classmethod?
            homogenous = False
            if tys:
                first = tys[0]
                for ty in tys[1:]:
                    if ty != first:
                        break
                else:
                    homogenous = True
            if homogenous:
                return_type = types.NamedUniTuple(first, len(tys), instance_class)
            else:
                return_type = types.NamedTuple(tys, instance_class)
            return return_type

        # Override the typer's pysig to match the namedtuple constructor's
        typer.pysig = pysig
        return typer
