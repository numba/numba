"""
CPU atomic operation stubs for Numba

This module provides atomic operations for CPU targets, similar to
the CUDA atomic operations but using standard LLVM atomic instructions.
"""

from numba.cuda.stubs import Stub


class atomic(Stub):
    """Namespace for CPU atomic operations"""

    class load(Stub):
        """
        load(ptr, ordering='acquire')

        Atomically load a value from memory with specified memory ordering.

        Parameters:
        - ptr: pointer to memory location
        - ordering: memory ordering ('acquire', 'relaxed', 'seq_cst')

        Returns:
        - The loaded value

        Supported types: uint8, uint16, uint32, uint64, int8, int16, int32, int64
        """

        pass

    class store(Stub):
        """
        store(ptr, val, ordering='release')

        Atomically store a value to memory with specified memory ordering.

        Parameters:
        - ptr: pointer to memory location
        - val: value to store
        - ordering: memory ordering ('release', 'relaxed', 'seq_cst')

        Supported types: uint8, uint16, uint32, uint64, int8, int16, int32, int64
        """

        pass

    class add(Stub):
        """
        add(ptr, val, ordering='acq_rel')

        Atomically add a value to memory location and return the previous value.

        Parameters:
        - ptr: pointer to memory location
        - val: value to add
        - ordering: memory ordering ('acq_rel', 'seq_cst', 'relaxed')

        Returns:
        - The previous value before addition

        Supported types: uint8, uint16, uint32, uint64, int8, int16, int32, int64
        """

        pass

    class sub(Stub):
        """
        sub(ptr, val, ordering='acq_rel')

        Atomically subtract a value from memory location and return the previous value.

        Parameters:
        - ptr: pointer to memory location
        - val: value to subtract
        - ordering: memory ordering ('acq_rel', 'seq_cst', 'relaxed')

        Returns:
        - The previous value before subtraction

        Supported types: uint8, uint16, uint32, uint64, int8, int16, int32, int64
        """

        pass

    class compare_and_swap(Stub):
        """
        compare_and_swap(ptr, expected, desired, ordering='acq_rel')

        Atomically compare and swap values at memory location.

        Parameters:
        - ptr: pointer to memory location
        - expected: expected current value
        - desired: desired new value
        - ordering: memory ordering ('acq_rel', 'seq_cst', 'relaxed')

        Returns:
        - The actual previous value (use to check if swap succeeded)

        Supported types: uint8, uint16, uint32, uint64, int8, int16, int32, int64
        """

        pass

    class fetch_add(Stub):
        """
        fetch_add(ptr, val, ordering='acq_rel')

        Atomically add a value to memory location and return the previous value.
        This is an alias for add() with explicit naming for fetch-and-add semantics.

        Parameters:
        - ptr: pointer to memory location
        - val: value to add
        - ordering: memory ordering ('acq_rel', 'seq_cst', 'relaxed')

        Returns:
        - The previous value before addition

        Supported types: uint8, uint16, uint32, uint64, int8, int16, int32, int64
        """

        pass
