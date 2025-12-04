"""
Type declarations for CPU atomic operations
"""

from numba import types
from numba.core.atomic_stubs import atomic
from numba.core.typing.templates import signature, AbstractTemplate, Registry

# Create typing registry for CPU atomic operations (following CUDA pattern)
registry = Registry()
register = registry.register
register_attr = registry.register_attr

# Supported atomic types - integers only for now
atomic_integer_types = (
    types.int8,
    types.uint8,
    types.int16,
    types.uint16,
    types.int32,
    types.uint32,
    types.int64,
    types.uint64,
)

# Memory ordering types
memory_orderings = ("relaxed", "acquire", "release", "acq_rel", "seq_cst")


def _validate_memory_ordering(ordering):
    """Validate memory ordering string"""
    if ordering not in memory_orderings:
        raise ValueError(
            f"Invalid memory ordering: {ordering}. Must be one of {memory_orderings}"
        )


@register
class AtomicLoad(AbstractTemplate):
    key = atomic.load

    def generic(self, args, kws):
        # Handle both ptr-only and ptr+ordering signatures
        if len(args) == 1:
            # load(ptr) - default ordering
            ptr_type = args[0]
            ordering = "acquire"  # default
        elif len(args) == 2:
            # load(ptr, ordering)
            ptr_type, ordering_type = args
            if not isinstance(ordering_type, types.UnicodeType) and not isinstance(
                    ordering_type, types.StringLiteral
            ):
                return None
            ordering = getattr(ordering_type, "literal_value", "acquire")
        else:
            return None

        # Validate pointer type
        if not isinstance(ptr_type, types.CPointer):
            return None

        # Validate element type is atomic-compatible
        element_type = ptr_type.dtype
        if element_type not in atomic_integer_types:
            return None

        # Validate memory ordering
        try:
            _validate_memory_ordering(ordering)
        except ValueError:
            return None

        # Return signature: element_type load(ptr_type, [ordering])
        return signature(element_type, *args)


@register
class AtomicStore(AbstractTemplate):
    key = atomic.store

    def generic(self, args, kws):
        # Handle both ptr+val and ptr+val+ordering signatures
        if len(args) == 2:
            # store(ptr, val) - default ordering
            ptr_type, val_type = args
            ordering = "release"  # default
        elif len(args) == 3:
            # store(ptr, val, ordering)
            ptr_type, val_type, ordering_type = args
            if not isinstance(ordering_type, types.UnicodeType) and not isinstance(
                    ordering_type, types.StringLiteral
            ):
                return None
            ordering = getattr(ordering_type, "literal_value", "release")
        else:
            return None

        # Validate pointer type
        if not isinstance(ptr_type, types.CPointer):
            return None

        # Validate element type is atomic-compatible
        element_type = ptr_type.dtype
        if element_type not in atomic_integer_types:
            return None

        # Validate value type matches element type
        if element_type != val_type:
            return None

        # Validate memory ordering
        try:
            _validate_memory_ordering(ordering)
        except ValueError:
            return None

        # Return signature: void store(ptr_type, val_type, [ordering])
        return signature(types.void, *args)


@register
class AtomicAdd(AbstractTemplate):
    key = atomic.add

    def generic(self, args, kws):
        # Handle both ptr+val and ptr+val+ordering signatures
        if len(args) == 2:
            # add(ptr, val) - default ordering
            ptr_type, val_type = args
            ordering = "acq_rel"  # default
        elif len(args) == 3:
            # add(ptr, val, ordering)
            ptr_type, val_type, ordering_type = args
            if not isinstance(ordering_type, types.UnicodeType) and not isinstance(
                    ordering_type, types.StringLiteral
            ):
                return None
            ordering = getattr(ordering_type, "literal_value", "acq_rel")
        else:
            return None

        # Validate pointer type
        if not isinstance(ptr_type, types.CPointer):
            return None

        # Validate element type is atomic-compatible
        element_type = ptr_type.dtype
        if element_type not in atomic_integer_types:
            return None

        # Validate value type matches element type
        if element_type != val_type:
            return None

        # Validate memory ordering
        try:
            _validate_memory_ordering(ordering)
        except ValueError:
            return None

        # Return signature: element_type add(ptr_type, val_type, [ordering])
        return signature(element_type, *args)


@register
class AtomicSub(AbstractTemplate):
    key = atomic.sub

    def generic(self, args, kws):
        # Handle both ptr+val and ptr+val+ordering signatures
        if len(args) == 2:
            # sub(ptr, val) - default ordering
            ptr_type, val_type = args
            ordering = "acq_rel"  # default
        elif len(args) == 3:
            # sub(ptr, val, ordering)
            ptr_type, val_type, ordering_type = args
            if not isinstance(ordering_type, types.UnicodeType) and not isinstance(
                    ordering_type, types.StringLiteral
            ):
                return None
            ordering = getattr(ordering_type, "literal_value", "acq_rel")
        else:
            return None

        # Validate pointer type
        if not isinstance(ptr_type, types.CPointer):
            return None

        # Validate element type is atomic-compatible
        element_type = ptr_type.dtype
        if element_type not in atomic_integer_types:
            return None

        # Validate value type matches element type
        if element_type != val_type:
            return None

        # Validate memory ordering
        try:
            _validate_memory_ordering(ordering)
        except ValueError:
            return None

        # Return signature: element_type sub(ptr_type, val_type, [ordering])
        return signature(element_type, *args)


@register
class AtomicCompareAndSwap(AbstractTemplate):
    key = atomic.compare_and_swap

    def generic(self, args, kws):
        # Handle both ptr+expected+desired and ptr+expected+desired+ordering signatures
        if len(args) == 3:
            # compare_and_swap(ptr, expected, desired) - default ordering
            ptr_type, expected_type, desired_type = args
            ordering = "acq_rel"  # default
        elif len(args) == 4:
            # compare_and_swap(ptr, expected, desired, ordering)
            ptr_type, expected_type, desired_type, ordering_type = args
            if not isinstance(ordering_type, types.UnicodeType) and not isinstance(
                    ordering_type, types.StringLiteral
            ):
                return None
            ordering = getattr(ordering_type, "literal_value", "acq_rel")
        else:
            return None

        # Validate pointer type
        if not isinstance(ptr_type, types.CPointer):
            return None

        # Validate element type is atomic-compatible
        element_type = ptr_type.dtype
        if element_type not in atomic_integer_types:
            return None

        # Validate expected and desired types match element type
        if element_type != expected_type or element_type != desired_type:
            return None

        # Validate memory ordering
        try:
            _validate_memory_ordering(ordering)
        except ValueError:
            return None

        # Return signature: element_type compare_and_swap(ptr_type, element_type, element_type, [ordering])
        return signature(element_type, *args)


@register
class AtomicFetchAdd(AbstractTemplate):
    key = atomic.fetch_add

    def generic(self, args, kws):
        # Handle both ptr+val and ptr+val+ordering signatures
        if len(args) == 2:
            # fetch_add(ptr, val) - default ordering
            ptr_type, val_type = args
            ordering = "acq_rel"  # default
        elif len(args) == 3:
            # fetch_add(ptr, val, ordering)
            ptr_type, val_type, ordering_type = args
            if not isinstance(ordering_type, types.UnicodeType) and not isinstance(
                    ordering_type, types.StringLiteral
            ):
                return None
            ordering = getattr(ordering_type, "literal_value", "acq_rel")
        else:
            return None

        # Validate pointer type
        if not isinstance(ptr_type, types.CPointer):
            return None

        # Validate element type is atomic-compatible
        element_type = ptr_type.dtype
        if element_type not in atomic_integer_types:
            return None

        # Validate value type matches element type
        if element_type != val_type:
            return None

        # Validate memory ordering
        try:
            _validate_memory_ordering(ordering)
        except ValueError:
            return None

        # Return signature: element_type fetch_add(ptr_type, val_type, [ordering])
        return signature(element_type, *args)


# CPU atomic operations generator (following CUDA pattern)
def _gen_atomic_load(l_key, supported_types):
    @register
    class AtomicArray(AbstractTemplate):
        key = l_key

        def generic(self, args, kws):
            assert not kws
            if len(args) < 2:
                return None

            ary, idx = args[0], args[1]

            # Must be array
            if not isinstance(ary, types.Array):
                return None

            # Element type must be supported
            if ary.dtype not in supported_types:
                return None

            # Handle optional ordering argument
            if len(args) > 2:
                # Validate ordering is string literal
                ordering_arg = args[2]
                if not isinstance(
                        ordering_arg, (types.UnicodeType, types.StringLiteral)
                ):
                    return None

            # Follow CUDA pattern for index normalization
            if ary.ndim == 1:
                if len(args) == 2:
                    return signature(ary.dtype, ary, types.intp)
                else:
                    return signature(ary.dtype, ary, types.intp, args[2])
            else:
                return signature(ary.dtype, *args)

    return AtomicArray


def _gen_atomic_store(l_key, supported_types):
    @register
    class AtomicArray(AbstractTemplate):
        key = l_key

        def generic(self, args, kws):
            assert not kws
            if len(args) < 3:
                return None

            ary, idx, val = args[0], args[1], args[2]

            # Must be array
            if not isinstance(ary, types.Array):
                return None

            # Element type must be supported and match value type
            if ary.dtype not in supported_types or val not in supported_types:
                return None

            # Handle optional ordering argument
            if len(args) > 3:
                # Validate ordering is string literal
                ordering_arg = args[3]
                if not isinstance(
                        ordering_arg, (types.UnicodeType, types.StringLiteral)
                ):
                    return None

            # Follow CUDA pattern for index normalization
            if ary.ndim == 1:
                if len(args) == 3:
                    return signature(types.void, ary, types.intp, ary.dtype)
                else:
                    return signature(types.void, ary, types.intp, ary.dtype, args[3])
            else:
                return signature(types.void, *args)

    return AtomicArray


def _gen_atomic_rmw(l_key, supported_types):
    @register
    class AtomicArray(AbstractTemplate):
        key = l_key

        def generic(self, args, kws):
            assert not kws
            if len(args) < 3:
                return None

            ary, idx, val = args[0], args[1], args[2]

            # Must be array
            if not isinstance(ary, types.Array):
                return None

            # Element type must be supported and match value type
            if ary.dtype not in supported_types or ary.dtype != val:
                return None

            # Handle optional ordering argument
            if len(args) > 3:
                # Validate ordering is string literal
                ordering_arg = args[3]
                if not isinstance(
                        ordering_arg, (types.UnicodeType, types.StringLiteral)
                ):
                    return None

            # Follow CUDA pattern for index normalization
            if ary.ndim == 1:
                if len(args) == 3:
                    return signature(ary.dtype, ary, types.intp, ary.dtype)
                else:
                    return signature(ary.dtype, ary, types.intp, ary.dtype, args[3])
            else:
                return signature(ary.dtype, *args)

    return AtomicArray


def _gen_atomic_cas(l_key, supported_types):
    @register
    class AtomicArray(AbstractTemplate):
        key = l_key

        def generic(self, args, kws):
            assert not kws
            if len(args) < 4:
                return None

            ary, idx, expected, desired = args[0], args[1], args[2], args[3]

            # Must be array
            if not isinstance(ary, types.Array):
                return None

            # Element type must be supported and match expected/desired types
            if (
                    ary.dtype not in supported_types
                    or ary.dtype != expected
                    or ary.dtype != desired
            ):
                return None

            # Handle optional ordering argument
            if len(args) > 4:
                # Validate ordering is string literal
                ordering_arg = args[4]
                if not isinstance(
                        ordering_arg, (types.UnicodeType, types.StringLiteral)
                ):
                    return None

            # Follow CUDA pattern for index normalization
            if ary.ndim == 1:
                if len(args) == 4:
                    return signature(ary.dtype, ary, types.intp, ary.dtype, ary.dtype)
                else:
                    return signature(
                        ary.dtype, ary, types.intp, ary.dtype, ary.dtype, args[4]
                    )
            else:
                return signature(ary.dtype, *args)

    return AtomicArray


# Generate atomic operation templates
AtomicLoadArray = _gen_atomic_load(atomic.load, atomic_integer_types)
AtomicStoreArray = _gen_atomic_store(atomic.store, atomic_integer_types)
AtomicAddArray = _gen_atomic_rmw(atomic.add, atomic_integer_types)
AtomicSubArray = _gen_atomic_rmw(atomic.sub, atomic_integer_types)
AtomicFetchAddArray = _gen_atomic_rmw(atomic.fetch_add, atomic_integer_types)
AtomicCompareAndSwapArray = _gen_atomic_cas(
    atomic.compare_and_swap, atomic_integer_types
)

# Register CPU module and atomic as a global
from numba.core.typing.templates import AttributeTemplate

# Register the CPU module and atomic as globals
# Register atomic directly as a module so "import numba.atomic" works
registry.register_global(atomic, types.Module(atomic))


@register_attr
class AtomicTemplate(AttributeTemplate):
    key = types.Module(atomic)

    def resolve_load(self, mod):
        return types.Function(AtomicLoadArray)

    def resolve_store(self, mod):
        return types.Function(AtomicStoreArray)

    def resolve_add(self, mod):
        return types.Function(AtomicAddArray)

    def resolve_sub(self, mod):
        return types.Function(AtomicSubArray)

    def resolve_compare_and_swap(self, mod):
        return types.Function(AtomicCompareAndSwapArray)

    def resolve_fetch_add(self, mod):
        return types.Function(AtomicFetchAddArray)
