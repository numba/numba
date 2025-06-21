"""
Implementation of CPU atomic operations using LLVM atomic instructions
"""

from numba import types
from numba.core import cgutils
from numba.core.atomic_stubs import atomic
from numba.core.imputils import Registry, lower_builtin

# Create registry for CPU atomic implementations (following CUDA pattern)
registry = Registry('atomic')
lower = registry.lower
lower_attr = registry.lower_getattr


def _get_llvm_atomic_ordering(ordering_str):
    """Convert string memory ordering to LLVM ordering"""
    ordering_map = {
        "relaxed": "monotonic",  # LLVM uses 'monotonic' for relaxed
        "acquire": "acquire",
        "release": "release",
        "acq_rel": "acq_rel",
        "seq_cst": "seq_cst",
    }
    return ordering_map.get(ordering_str, "monotonic")


def _get_ordering_from_args(args, default_ordering):
    """Extract ordering from function arguments"""
    if len(args) > 2:  # Has ordering argument
        ordering_arg = args[-1]
        if hasattr(ordering_arg, "literal_value"):
            return ordering_arg.literal_value
    return default_ordering


@lower_builtin(atomic.load, types.CPointer)
@lower_builtin(atomic.load, types.CPointer, types.UnicodeType)
@lower_builtin(atomic.load, types.CPointer, types.StringLiteral)
def atomic_load_impl(context, builder, sig, args):
    """
    Implementation of atomic load operation
    """
    ptr_type = sig.args[0]
    element_type = ptr_type.dtype
    ptr_val = args[0]

    # Get memory ordering
    ordering = "acquire"  # default
    if len(args) > 1:
        ordering_arg = sig.args[1]
        if hasattr(ordering_arg, "literal_value"):
            ordering = ordering_arg.literal_value

    llvm_ordering = _get_llvm_atomic_ordering(ordering)

    # Get LLVM type for the element
    llvm_element_type = context.get_value_type(element_type)

    # Create atomic load instruction
    # LLVM atomic load: load atomic <ty>, <ty>* <pointer> <ordering>, align <alignment>
    loaded_val = builder.load_atomic(ptr_val, llvm_ordering, align=1)

    return loaded_val


@lower_builtin(atomic.store, types.CPointer, types.Any)
@lower_builtin(atomic.store, types.CPointer, types.Any, types.UnicodeType)
@lower_builtin(atomic.store, types.CPointer, types.Any, types.StringLiteral)
def atomic_store_impl(context, builder, sig, args):
    """
    Implementation of atomic store operation
    """
    ptr_type = sig.args[0]
    element_type = ptr_type.dtype
    ptr_val = args[0]
    val = args[1]

    # Get memory ordering
    ordering = "release"  # default
    if len(args) > 2:
        ordering_arg = sig.args[2]
        if hasattr(ordering_arg, "literal_value"):
            ordering = ordering_arg.literal_value

    llvm_ordering = _get_llvm_atomic_ordering(ordering)

    # Create atomic store instruction
    # LLVM atomic store: store atomic <ty> <value>, <ty>* <pointer> <ordering>, align <alignment>
    builder.store_atomic(val, ptr_val, llvm_ordering, align=1)

    # Return void (no return value for store)
    return context.get_dummy_value()


@lower_builtin(atomic.add, types.CPointer, types.Any)
@lower_builtin(atomic.add, types.CPointer, types.Any, types.UnicodeType)
@lower_builtin(atomic.add, types.CPointer, types.Any, types.StringLiteral)
def atomic_add_impl(context, builder, sig, args):
    """
    Implementation of atomic add operation using LLVM atomicrmw
    """
    ptr_type = sig.args[0]
    element_type = ptr_type.dtype
    ptr_val = args[0]
    val = args[1]

    # Get memory ordering
    ordering = "acq_rel"  # default
    if len(args) > 2:
        ordering_arg = sig.args[2]
        if hasattr(ordering_arg, "literal_value"):
            ordering = ordering_arg.literal_value

    llvm_ordering = _get_llvm_atomic_ordering(ordering)

    # Create atomic RMW add instruction
    # LLVM atomicrmw: atomicrmw <operation> <ty>* <pointer>, <ty> <value> <ordering>
    # Returns the previous value
    old_val = builder.atomic_rmw("add", ptr_val, val, llvm_ordering)

    return old_val


@lower_builtin(atomic.sub, types.CPointer, types.Any)
@lower_builtin(atomic.sub, types.CPointer, types.Any, types.UnicodeType)
@lower_builtin(atomic.sub, types.CPointer, types.Any, types.StringLiteral)
def atomic_sub_impl(context, builder, sig, args):
    """
    Implementation of atomic subtract operation using LLVM atomicrmw
    """
    ptr_type = sig.args[0]
    element_type = ptr_type.dtype
    ptr_val = args[0]
    val = args[1]

    # Get memory ordering
    ordering = "acq_rel"  # default
    if len(args) > 2:
        ordering_arg = sig.args[2]
        if hasattr(ordering_arg, "literal_value"):
            ordering = ordering_arg.literal_value

    llvm_ordering = _get_llvm_atomic_ordering(ordering)

    # Create atomic RMW sub instruction
    old_val = builder.atomic_rmw("sub", ptr_val, val, llvm_ordering)

    return old_val


@lower_builtin(atomic.compare_and_swap, types.CPointer, types.Any, types.Any)
@lower_builtin(
    atomic.compare_and_swap, types.CPointer, types.Any, types.Any, types.UnicodeType
)
@lower_builtin(
    atomic.compare_and_swap, types.CPointer, types.Any, types.Any, types.StringLiteral
)
def atomic_compare_and_swap_impl(context, builder, sig, args):
    """
    Implementation of atomic compare-and-swap operation using LLVM cmpxchg
    """
    ptr_type = sig.args[0]
    element_type = ptr_type.dtype
    ptr_val = args[0]
    expected_val = args[1]
    desired_val = args[2]

    # Get memory ordering
    ordering = "acq_rel"  # default
    if len(args) > 3:
        ordering_arg = sig.args[3]
        if hasattr(ordering_arg, "literal_value"):
            ordering = ordering_arg.literal_value

    llvm_ordering = _get_llvm_atomic_ordering(ordering)

    # Create atomic cmpxchg instruction
    # LLVM cmpxchg: cmpxchg <ty>* <pointer>, <ty> <cmp>, <ty> <new> <success_ordering> <failure_ordering>
    # Returns {<ty>, i1} - the old value and a boolean indicating success

    # For cmpxchg, failure ordering must be weaker than success ordering
    failure_ordering = (
        "acquire" if llvm_ordering in ["acq_rel", "seq_cst"] else llvm_ordering
    )
    result = builder.cmpxchg(
        ptr_val, expected_val, desired_val, llvm_ordering, failure_ordering
    )

    # Extract the old value (first element of the returned struct)
    old_val = builder.extract_value(result, 0)

    return old_val


# Helper functions for creating pointer operations on arrays/memory


def _get_array_element_ptr(context, builder, ary_type, ary_val, index_val):
    """Get pointer to array element for atomic operations"""
    # Create array structure
    ary_struct = context.make_array(ary_type)(context, builder, ary_val)

    # Get pointer to the indexed element
    ptr = cgutils.get_item_pointer(context, builder, ary_type, ary_struct, (index_val,))

    return ptr


# Additional lowering rules for array indexing versions of atomic operations
@lower_builtin(atomic.load, types.Array, types.intp)
@lower_builtin(atomic.load, types.Array, types.intp, types.UnicodeType)
@lower_builtin(atomic.load, types.Array, types.intp, types.StringLiteral)
def atomic_load_array_impl(context, builder, sig, args):
    """
    Implementation of atomic load from array element
    """
    ary_type = sig.args[0]
    element_type = ary_type.dtype
    ary_val = args[0]
    index_val = args[1]

    # Get pointer to array element
    ptr = _get_array_element_ptr(context, builder, ary_type, ary_val, index_val)

    # Get memory ordering
    ordering = "acquire"  # default
    if len(args) > 2:
        ordering_arg = sig.args[2]
        if hasattr(ordering_arg, "literal_value"):
            ordering = ordering_arg.literal_value

    llvm_ordering = _get_llvm_atomic_ordering(ordering)

    # Create atomic load instruction
    loaded_val = builder.load_atomic(ptr, llvm_ordering, align=1)

    return loaded_val


@lower_builtin(atomic.store, types.Array, types.intp, types.Any)
@lower_builtin(atomic.store, types.Array, types.intp, types.Any, types.UnicodeType)
@lower_builtin(atomic.store, types.Array, types.intp, types.Any, types.StringLiteral)
def atomic_store_array_impl(context, builder, sig, args):
    """
    Implementation of atomic store to array element
    """
    ary_type = sig.args[0]
    element_type = ary_type.dtype
    ary_val = args[0]
    index_val = args[1]
    val = args[2]

    # Get pointer to array element
    ptr = _get_array_element_ptr(context, builder, ary_type, ary_val, index_val)

    # Get memory ordering
    ordering = "release"  # default
    if len(args) > 3:
        ordering_arg = sig.args[3]
        if hasattr(ordering_arg, "literal_value"):
            ordering = ordering_arg.literal_value

    llvm_ordering = _get_llvm_atomic_ordering(ordering)

    # Create atomic store instruction
    builder.store_atomic(val, ptr, llvm_ordering, align=1)

    return context.get_dummy_value()


@lower_builtin(atomic.add, types.Array, types.intp, types.Any)
@lower_builtin(atomic.add, types.Array, types.intp, types.Any, types.UnicodeType)
@lower_builtin(atomic.add, types.Array, types.intp, types.Any, types.StringLiteral)
def atomic_add_array_impl(context, builder, sig, args):
    """
    Implementation of atomic add on array element
    """
    ary_type = sig.args[0]
    element_type = ary_type.dtype
    ary_val = args[0]
    index_val = args[1]
    val = args[2]

    # Get pointer to array element
    ptr = _get_array_element_ptr(context, builder, ary_type, ary_val, index_val)

    # Get memory ordering
    ordering = "acq_rel"  # default
    if len(args) > 3:
        ordering_arg = sig.args[3]
        if hasattr(ordering_arg, "literal_value"):
            ordering = ordering_arg.literal_value

    llvm_ordering = _get_llvm_atomic_ordering(ordering)

    # Create atomic RMW add instruction
    old_val = builder.atomic_rmw("add", ptr, val, llvm_ordering)

    return old_val


@lower_builtin(atomic.sub, types.Array, types.intp, types.Any)
@lower_builtin(atomic.sub, types.Array, types.intp, types.Any, types.UnicodeType)
@lower_builtin(atomic.sub, types.Array, types.intp, types.Any, types.StringLiteral)
def atomic_sub_array_impl(context, builder, sig, args):
    """
    Implementation of atomic subtract on array element
    """
    ary_type = sig.args[0]
    element_type = ary_type.dtype
    ary_val = args[0]
    index_val = args[1]
    val = args[2]

    # Get pointer to array element
    ptr = _get_array_element_ptr(context, builder, ary_type, ary_val, index_val)

    # Get memory ordering
    ordering = "acq_rel"  # default
    if len(args) > 3:
        ordering_arg = sig.args[3]
        if hasattr(ordering_arg, "literal_value"):
            ordering = ordering_arg.literal_value

    llvm_ordering = _get_llvm_atomic_ordering(ordering)

    # Create atomic RMW sub instruction
    old_val = builder.atomic_rmw("sub", ptr, val, llvm_ordering)

    return old_val


@lower_builtin(atomic.fetch_add, types.CPointer, types.Any)
@lower_builtin(atomic.fetch_add, types.CPointer, types.Any, types.UnicodeType)
@lower_builtin(atomic.fetch_add, types.CPointer, types.Any, types.StringLiteral)
def atomic_fetch_add_impl(context, builder, sig, args):
    """
    Implementation of atomic fetch_add operation using LLVM atomicrmw
    Same as add - atomically adds value and returns previous value
    """
    ptr_type = sig.args[0]
    element_type = ptr_type.dtype
    ptr_val = args[0]
    val = args[1]

    # Get memory ordering
    ordering = "acq_rel"  # default
    if len(args) > 2:
        ordering_arg = sig.args[2]
        if hasattr(ordering_arg, "literal_value"):
            ordering = ordering_arg.literal_value

    llvm_ordering = _get_llvm_atomic_ordering(ordering)

    # Create atomic RMW add instruction
    # LLVM atomicrmw: atomicrmw <operation> <ty>* <pointer>, <ty> <value> <ordering>
    # Returns the previous value
    old_val = builder.atomic_rmw("add", ptr_val, val, llvm_ordering)

    return old_val


@lower_builtin(atomic.fetch_add, types.Array, types.intp, types.Any)
@lower_builtin(atomic.fetch_add, types.Array, types.intp, types.Any, types.UnicodeType)
@lower_builtin(
    atomic.fetch_add, types.Array, types.intp, types.Any, types.StringLiteral
)
def atomic_fetch_add_array_impl(context, builder, sig, args):
    """
    Implementation of atomic fetch_add on array element
    Same as add - atomically adds value and returns previous value
    """
    ary_type = sig.args[0]
    element_type = ary_type.dtype
    ary_val = args[0]
    index_val = args[1]
    val = args[2]

    # Get pointer to array element
    ptr = _get_array_element_ptr(context, builder, ary_type, ary_val, index_val)

    # Get memory ordering
    ordering = "acq_rel"  # default
    if len(args) > 3:
        ordering_arg = sig.args[3]
        if hasattr(ordering_arg, "literal_value"):
            ordering = ordering_arg.literal_value

    llvm_ordering = _get_llvm_atomic_ordering(ordering)

    # Create atomic RMW add instruction
    old_val = builder.atomic_rmw("add", ptr, val, llvm_ordering)

    return old_val


@lower_builtin(atomic.compare_and_swap, types.Array, types.intp, types.Any, types.Any)
@lower_builtin(
    atomic.compare_and_swap,
    types.Array,
    types.intp,
    types.Any,
    types.Any,
    types.UnicodeType,
)
@lower_builtin(
    atomic.compare_and_swap,
    types.Array,
    types.intp,
    types.Any,
    types.Any,
    types.StringLiteral,
)
def atomic_compare_and_swap_array_impl(context, builder, sig, args):
    """
    Implementation of atomic compare_and_swap on array element
    """
    ary_type = sig.args[0]
    element_type = ary_type.dtype
    ary_val = args[0]
    index_val = args[1]
    expected = args[2]
    desired = args[3]

    # Get pointer to array element
    ptr = _get_array_element_ptr(context, builder, ary_type, ary_val, index_val)

    # Get memory ordering
    ordering = "acq_rel"  # default
    if len(args) > 4:
        ordering_arg = sig.args[4]
        if hasattr(ordering_arg, "literal_value"):
            ordering = ordering_arg.literal_value

    llvm_ordering = _get_llvm_atomic_ordering(ordering)

    # Create atomic compare and exchange instruction
    # LLVM cmpxchg returns {<type>, i1} where i1 indicates success
    # We want just the old value regardless of success
    # For cmpxchg, failure ordering must be weaker than success ordering
    failure_ordering = (
        "acquire" if llvm_ordering in ["acq_rel", "seq_cst"] else llvm_ordering
    )
    cmpxchg_result = builder.cmpxchg(
        ptr, expected, desired, llvm_ordering, failure_ordering
    )
    old_val = builder.extract_value(cmpxchg_result, 0)

    return old_val
