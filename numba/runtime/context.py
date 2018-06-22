from __future__ import print_function, absolute_import, division

from llvmlite import ir

from numba import cgutils, types


class NRTContext(object):
    """
    An object providing access to NRT APIs in the lowering pass.
    """

    def __init__(self, context, enabled):
        self._context = context
        self._enabled = enabled

    def _require_nrt(self):
        if not self._enabled:
            raise RuntimeError("NRT required but not enabled")

    def allocate(self, builder, size):
        """
        Low-level allocate a new memory area of `size` bytes.
        """
        self._require_nrt()

        mod = builder.module
        fnty = ir.FunctionType(cgutils.voidptr_t, [cgutils.intp_t])
        fn = mod.get_or_insert_function(fnty, name="NRT_Allocate")
        fn.return_value.add_attribute("noalias")
        return builder.call(fn, [size])

    def free(self, builder, ptr):
        """
        Low-level free a memory area allocated with allocate().
        """
        self._require_nrt()

        mod = builder.module
        fnty = ir.FunctionType(ir.VoidType(), [cgutils.voidptr_t])
        fn = mod.get_or_insert_function(fnty, name="NRT_Free")
        return builder.call(fn, [ptr])

    def meminfo_alloc(self, builder, size):
        """
        Allocate a new MemInfo with a data payload of `size` bytes.

        A pointer to the MemInfo is returned.
        """
        self._require_nrt()

        mod = builder.module
        fnty = ir.FunctionType(cgutils.voidptr_t, [cgutils.intp_t])
        fn = mod.get_or_insert_function(fnty, name="NRT_MemInfo_alloc_safe")
        fn.return_value.add_attribute("noalias")
        return builder.call(fn, [size])

    def meminfo_alloc_dtor(self, builder, size, dtor):
        self._require_nrt()

        mod = builder.module
        fnty = ir.FunctionType(cgutils.voidptr_t,
                               [cgutils.intp_t, cgutils.voidptr_t])
        fn = mod.get_or_insert_function(fnty,
                                        name="NRT_MemInfo_alloc_dtor_safe")
        fn.return_value.add_attribute("noalias")
        return builder.call(fn, [size,
                                 builder.bitcast(dtor, cgutils.voidptr_t)])

    def meminfo_alloc_aligned(self, builder, size, align):
        """
        Allocate a new MemInfo with an aligned data payload of `size` bytes.
        The data pointer is aligned to `align` bytes.  `align` can be either
        a Python int or a LLVM uint32 value.

        A pointer to the MemInfo is returned.
        """
        self._require_nrt()

        mod = builder.module
        u32 = ir.IntType(32)
        fnty = ir.FunctionType(cgutils.voidptr_t, [cgutils.intp_t, u32])
        fn = mod.get_or_insert_function(fnty,
                                        name="NRT_MemInfo_alloc_safe_aligned")
        fn.return_value.add_attribute("noalias")
        if isinstance(align, int):
            align = self._context.get_constant(types.uint32, align)
        else:
            assert align.type == u32, "align must be a uint32"
        return builder.call(fn, [size, align])

    def meminfo_new_varsize(self, builder, size):
        """
        Allocate a MemInfo pointing to a variable-sized data area.  The area
        is separately allocated (i.e. two allocations are made) so that
        re-allocating it doesn't change the MemInfo's address.

        A pointer to the MemInfo is returned.
        """
        self._require_nrt()

        mod = builder.module
        fnty = ir.FunctionType(cgutils.voidptr_t, [cgutils.intp_t])
        fn = mod.get_or_insert_function(fnty, name="NRT_MemInfo_new_varsize")
        fn.return_value.add_attribute("noalias")
        return builder.call(fn, [size])

    def meminfo_new_varsize_dtor(self, builder, size, dtor):
        """
        Like meminfo_new_varsize() but also set the destructor for
        cleaning up references to objects inside the allocation.
        """
        self._require_nrt()

        mod = builder.module
        fnty = ir.FunctionType(cgutils.voidptr_t,
                               [cgutils.intp_t, cgutils.voidptr_t])
        fn = mod.get_or_insert_function(
            fnty, name="NRT_MemInfo_new_varsize_dtor")
        return builder.call(fn, [size, dtor])

    def meminfo_varsize_alloc(self, builder, meminfo, size):
        """
        Allocate a new data area for a MemInfo created by meminfo_new_varsize().
        The new data pointer is returned, for convenience.

        Contrary to realloc(), this always allocates a new area and doesn't
        copy the old data.  This is useful if resizing a container needs
        more than simply copying the data area (e.g. for hash tables).

        The old pointer will have to be freed with meminfo_varsize_free().
        """
        return self._call_varsize_alloc(builder, meminfo, size,
                                        "NRT_MemInfo_varsize_alloc")

    def meminfo_varsize_realloc(self, builder, meminfo, size):
        """
        Reallocate a data area allocated by meminfo_new_varsize().
        The new data pointer is returned, for convenience.
        """
        return self._call_varsize_alloc(builder, meminfo, size,
                                        "NRT_MemInfo_varsize_realloc")

    def meminfo_varsize_free(self, builder, meminfo, ptr):
        """
        Free a memory area allocated for a NRT varsize object.
        Note this does *not* free the NRT object itself!
        """
        self._require_nrt()

        mod = builder.module
        fnty = ir.FunctionType(ir.VoidType(),
                               [cgutils.voidptr_t, cgutils.voidptr_t])
        fn = mod.get_or_insert_function(fnty, name="NRT_MemInfo_varsize_free")
        return builder.call(fn, (meminfo, ptr))

    def _call_varsize_alloc(self, builder, meminfo, size, funcname):
        self._require_nrt()

        mod = builder.module
        fnty = ir.FunctionType(cgutils.voidptr_t,
                               [cgutils.voidptr_t, cgutils.intp_t])
        fn = mod.get_or_insert_function(fnty, name=funcname)
        fn.return_value.add_attribute("noalias")
        return builder.call(fn, [meminfo, size])

    def meminfo_data(self, builder, meminfo):
        """
        Given a MemInfo pointer, return a pointer to the allocated data
        managed by it.  This works for MemInfos allocated with all the
        above methods.
        """
        self._require_nrt()

        from numba.runtime.nrtdynmod import meminfo_data_ty

        mod = builder.module
        fn = mod.get_or_insert_function(meminfo_data_ty,
                                        name="NRT_MemInfo_data_fast")
        return builder.call(fn, [meminfo])

    def _call_incref_decref(self, builder, root_type, typ, value,
                                funcname, getters=()):
        self._require_nrt()

        from numba.runtime.nrtdynmod import incref_decref_ty

        data_model = self._context.data_model_manager[typ]

        members = data_model.traverse(builder)
        for mtyp, getter in members:
            self._call_incref_decref(builder, root_type, mtyp, value,
                                     funcname, getters + (getter,))

        if data_model.has_nrt_meminfo():
            # Call the chain of getters to compute the member value
            for getter in getters:
                value = getter(value)
            try:
                meminfo = data_model.get_nrt_meminfo(builder, value)
            except NotImplementedError as e:
                raise NotImplementedError("%s: %s" % (root_type, str(e)))
            assert meminfo is not None  # since has_nrt_meminfo()
            mod = builder.module
            fn = mod.get_or_insert_function(incref_decref_ty, name=funcname)
            # XXX "nonnull" causes a crash in test_dyn_array: can this
            # function be called with a NULL pointer?
            fn.args[0].add_attribute("noalias")
            fn.args[0].add_attribute("nocapture")
            builder.call(fn, [meminfo])

    def incref(self, builder, typ, value):
        """
        Recursively incref the given *value* and its members.
        """
        self._call_incref_decref(builder, typ, typ, value, "NRT_incref")

    def decref(self, builder, typ, value):
        """
        Recursively decref the given *value* and its members.
        """
        self._call_incref_decref(builder, typ, typ, value, "NRT_decref")
