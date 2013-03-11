"""
Compiling @autojit extension classes works as follows:

    * Create an extension Numba/minivect type holding a symtab
    * Capture attribute types in the symtab in the same was as @jit
    * Build attribute struct type

        We probably want to use a perfect hash table for attributes, hashing
        on (attr_name, attr_type), with only one entry having 'attr_name'
        at any given time. This would allow callers to read attributes of
        known or expected types (perhaps inferred from some
        context) dynamically.

    For all methods M with static input types:
        * Compile M
        * Register M in a list of compiled methods

    * Build initial hash-based virtual method table from compiled methods

        * Create pre-hash values for the signatures
            * We use these values to look up methods at runtime

        * Parametrize the virtual method table to build a final hash function:

            slot_index = (((prehash >> table.r) & self.table.m_f) ^
                           self.displacements[prehash & self.table.m_g])

            See also virtual.py and the following SEPs:

                https://github.com/numfocus/sep/blob/master/sep200.rst
                https://github.com/numfocus/sep/blob/master/sep201.rst

            And the following paper to understand the perfect hashing scheme:

                Hash and Displace: Efficient Evaluation of Minimal Perfect
                Hash Functions (1999) by Rasmus Pagn:

                    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.32.6530

    * Create descriptors that wrap the native attributes
    * Create an extension type:

            {
                hash-based virtual method table (PyCustomSlots_Table **)
                PyGC_HEAD
                PyObject_HEAD
                ...
                native attributes
            }

        We precede the object with the table to make this work in a more
        generic scheme, e.g. where a caller is dealing with an unknown
        object, and we quickly want to see whether it support such a
        perfect-hashing virtual method table:

            if (o->ob_type->tp_flags & NATIVELY_CALLABLE_TABLE) {
                PyCustomSlots_Table ***slot_p = ((char *) o) - sizeof(PyGC_HEAD)
                PyCustomSlots_Table *vtab = **slot_p
                look up function
            } else {
                PyObject_Call(...)
            }

        We need to store a PyCustomSlots_Table ** in the object to allow
        the producer of the table to replace the table with a new table
        for all live objects (e.g. by adding a specialization for
        an autojit method).
"""
