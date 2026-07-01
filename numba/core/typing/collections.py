from .. import types, utils, errors
import operator
from .templates import (AttributeTemplate, AbstractTemplate, Signature,
                        infer_global, infer_getattr,
                        signature, fold_arguments)
from .builtins import normalize_1d_index
from .typeof import typeof

@infer_global(operator.contains)
class InContainer(AbstractTemplate):
    key = operator.contains

    def generic(self, args, kws):
        cont, item = args
        if isinstance(cont, types.Container):
            return signature(types.boolean, cont, cont.dtype)

@infer_global(len)
class ContainerLen(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        (val,) = args
        if isinstance(val, (types.Container)):
            return signature(types.intp, val)


@infer_global(operator.truth)
class SequenceBool(AbstractTemplate):
    key = operator.truth

    def generic(self, args, kws):
        assert not kws
        (val,) = args
        if isinstance(val, (types.Sequence)):
            return signature(types.boolean, val)


@infer_global(operator.getitem)
class GetItemSequence(AbstractTemplate):
    key = operator.getitem

    def generic(self, args, kws):
        seq, idx = args
        if isinstance(seq, types.Sequence):
            idx = normalize_1d_index(idx)
            if isinstance(idx, types.SliceType):
                # Slicing a tuple only supported with static_getitem
                if not isinstance(seq, types.BaseTuple):
                    return signature(seq, seq, idx)
            elif isinstance(idx, types.Integer):
                return signature(seq.dtype, seq, idx)

@infer_global(operator.setitem)
class SetItemSequence(AbstractTemplate):
    def generic(self, args, kws):
        seq, idx, value = args
        if isinstance(seq, types.MutableSequence):
            idx = normalize_1d_index(idx)
            if isinstance(idx, types.SliceType):
                return signature(types.none, seq, idx, seq)
            elif isinstance(idx, types.Integer):
                if not self.context.can_convert(value, seq.dtype):
                    msg = "invalid setitem with value of {} to element of {}"
                    raise errors.TypingError(msg.format(types.unliteral(value), seq.dtype))
                return signature(types.none, seq, idx, seq.dtype)


@infer_global(operator.delitem)
class DelItemSequence(AbstractTemplate):
    def generic(self, args, kws):
        seq, idx = args
        if isinstance(seq, types.MutableSequence):
            idx = normalize_1d_index(idx)
            return signature(types.none, seq, idx)


# --------------------------------------------------------------------------
# named tuples

@infer_getattr
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


class NamedTupleCallTemplate(AbstractTemplate):
    """Base call template for NamedTuple class constructors.
    
    Subclasses should define class variables `instance_class` and `key`.
    """

    def generic(self, args, kws):
        pysig = utils.pysignature(self.instance_class)

        def stararg_handler(index, param, value):
            raise NotImplementedError(
                "stararg not implemented for NamedTuple constructors"
            )

        # Handle default arguments by looking up their type.
        folded = fold_arguments(
            pysig=pysig,
            args=args,
            kws=kws,
            normal_handler=lambda index, param, value: value,
            default_handler=lambda index, param, default: typeof(default),
            stararg_handler=stararg_handler,
        )

        return_type = types.BaseTuple.from_types(folded, self.instance_class)
        sig = Signature(return_type, args=folded, recvr=None, pysig=pysig)
        return sig


@infer_getattr
class NamedTupleClassAttribute(AttributeTemplate):
    key = types.NamedTupleClass

    def resolve___call__(self, classty):
        """
        Resolve the named tuple constructor, aka the class's __call__ method.
        """
        class SpecificNamedTupleCallTemplate(NamedTupleCallTemplate):
            key = classty.instance_class
            instance_class = classty.instance_class

        instance_class = classty.instance_class
        pysig = utils.pysignature(instance_class)

        return types.Function(SpecificNamedTupleCallTemplate)
