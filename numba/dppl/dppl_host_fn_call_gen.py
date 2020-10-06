from __future__ import print_function, division, absolute_import

import dpctl
import llvmlite.llvmpy.core as lc
import llvmlite.ir.values as liv
import llvmlite.ir as lir
import llvmlite.binding as lb
from numba.core import types, cgutils

from numba.core.ir_utils import legalize_names

class DPPLHostFunctionCallsGenerator(object):
    def __init__(self, lowerer, cres, num_inputs):
        self.lowerer = lowerer
        self.context = self.lowerer.context
        self.builder = self.lowerer.builder

        self.kernel = cres.kernel
        self.kernel_addr = self.kernel.addressof_ref()

        # Initialize commonly used LLVM types and constant
        self._init_llvm_types_and_constants()
        # Create functions that we need to call
        self._declare_functions()
        # Create a NULL void * pointer for meminfo and parent
        # parts of ndarray type args
        self.null_ptr = self._create_null_ptr()

        self.total_kernel_args = 0
        self.cur_arg           = 0
        self.num_inputs        = num_inputs

        # list of buffer that needs to comeback to host
        self.write_buffs = []

        # list of buffer that does not need to comeback to host
        self.read_only_buffs = []


    def _create_null_ptr(self):
        null_ptr = cgutils.alloca_once(self.builder, self.void_ptr_t,
                size=self.context.get_constant(types.uintp, 1), name="null_ptr")
        self.builder.store(
            self.builder.inttoptr(
                self.context.get_constant(types.uintp, 0), self.void_ptr_t),
                null_ptr)
        return null_ptr


    def _init_llvm_types_and_constants(self):
        self.byte_t          = lc.Type.int(8)
        self.byte_ptr_t      = lc.Type.pointer(self.byte_t)
        self.byte_ptr_ptr_t  = lc.Type.pointer(self.byte_ptr_t)
        self.intp_t          = self.context.get_value_type(types.intp)
        self.long_t          = self.context.get_value_type(types.int64)
        self.int32_t         = self.context.get_value_type(types.int32)
        self.int32_ptr_t     = lc.Type.pointer(self.int32_t)
        self.uintp_t         = self.context.get_value_type(types.uintp)
        self.intp_ptr_t      = lc.Type.pointer(self.intp_t)
        self.uintp_ptr_t     = lc.Type.pointer(self.uintp_t)
        self.zero            = self.context.get_constant(types.uintp, 0)
        self.one             = self.context.get_constant(types.uintp, 1)
        self.one_type        = self.one.type
        self.sizeof_intp     = self.context.get_abi_sizeof(self.intp_t)
        self.void_ptr_t      = self.context.get_value_type(types.voidptr)
        self.void_ptr_ptr_t  = lc.Type.pointer(self.void_ptr_t)
        self.sizeof_void_ptr = self.context.get_abi_sizeof(self.intp_t)
        self.sycl_queue_val = None

    def _declare_functions(self):
        get_queue_fnty = lc.Type.function(self.void_ptr_t, ())
        self.get_queue = self.builder.module.get_or_insert_function(get_queue_fnty,
                                                                name="DPPLQueueMgr_GetCurrentQueue")

        submit_range_fnty = lc.Type.function(self.void_ptr_t,
                [self.void_ptr_t, self.void_ptr_t, self.void_ptr_ptr_t,
                    self.int32_ptr_t, self.intp_t, self.intp_ptr_t,
                    self.intp_t, self.void_ptr_t, self.intp_t])
        self.submit_range = self.builder.module.get_or_insert_function(submit_range_fnty,
                                                                name="DPPLQueue_SubmitRange")


        queue_memcpy_fnty = lc.Type.function(lir.VoidType(), [self.void_ptr_t, self.void_ptr_t, self.void_ptr_t, self.intp_t])
        self.queue_memcpy = self.builder.module.get_or_insert_function(queue_memcpy_fnty,
                                                                name="DPPLQueue_Memcpy")

        queue_wait_fnty =  lc.Type.function(lir.VoidType(), [self.void_ptr_t])
        self.queue_wait = self.builder.module.get_or_insert_function(queue_wait_fnty,
                                                                name="DPPLQueue_Wait")

        usm_shared_fnty = lc.Type.function(self.void_ptr_t, [self.intp_t, self.void_ptr_t])
        self.usm_shared = self.builder.module.get_or_insert_function(usm_shared_fnty,
                                                                name="DPPLmalloc_shared")

        usm_free_fnty = lc.Type.function(lir.VoidType(), [self.void_ptr_t, self.void_ptr_t])
        self.usm_free = self.builder.module.get_or_insert_function(usm_free_fnty,
                                                                   name="DPPLfree_with_queue")

    def allocate_kenrel_arg_array(self, num_kernel_args):
        self.sycl_queue_val = cgutils.alloca_once(self.builder, self.void_ptr_t)
        self.builder.store(self.builder.call(self.get_queue, []), self.sycl_queue_val)

        self.total_kernel_args = num_kernel_args

        # we need a kernel arg array to enqueue
        self.kernel_arg_array = cgutils.alloca_once(
            self.builder, self.void_ptr_t, size=self.context.get_constant(
                types.uintp, num_kernel_args), name="kernel_arg_array")

        self.kernel_arg_ty_array = cgutils.alloca_once(
            self.builder, self.int32_t, size=self.context.get_constant(
                types.uintp, num_kernel_args), name="kernel_arg_ty_array")


    def resolve_and_return_dpctl_type(self, ty):
        val = None
        if ty == types.int32 or isinstance(ty, types.scalars.IntegerLiteral):
            val = self.context.get_constant(types.int32, 4)
        elif ty == types.uint32:
            val = self.context.get_constant(types.int32, 5)
        elif ty == types.boolean:
            val = self.context.get_constant(types.int32, 5)
        elif ty == types.int64:
            val = self.context.get_constant(types.int32, 7)
        elif ty == types.uint64:
            val = self.context.get_constant(types.int32, 8)
        elif ty == types.float32:
            val = self.context.get_constant(types.int32, 12)
        elif ty == types.float64:
            val = self.context.get_constant(types.int32, 13)
        elif ty == types.voidptr:
            val = self.context.get_constant(types.int32, 15)
        else:
            raise NotImplementedError

        assert(val != None)

        return val


    def form_kernel_arg_and_arg_ty(self, val, ty):
        kernel_arg_dst = self.builder.gep(self.kernel_arg_array, [self.context.get_constant(types.int32, self.cur_arg)])
        kernel_arg_ty_dst = self.builder.gep(self.kernel_arg_ty_array, [self.context.get_constant(types.int32, self.cur_arg)])
        self.cur_arg += 1
        self.builder.store(val, kernel_arg_dst)
        self.builder.store(ty, kernel_arg_ty_dst)


    def process_kernel_arg(self, var, llvm_arg, arg_type, gu_sig, val_type, index, modified_arrays):
        if isinstance(arg_type, types.npytypes.Array):
            if llvm_arg is None:
                raise NotImplementedError(arg_type, var)

            storage = cgutils.alloca_once(self.builder, self.long_t)
            self.builder.store(self.context.get_constant(types.int64, 0), storage)
            ty = self.resolve_and_return_dpctl_type(types.int64)
            self.form_kernel_arg_and_arg_ty(self.builder.bitcast(storage, self.void_ptr_t), ty)

            storage = cgutils.alloca_once(self.builder, self.long_t)
            self.builder.store(self.context.get_constant(types.int64, 0), storage)
            ty = self.resolve_and_return_dpctl_type(types.int64)
            self.form_kernel_arg_and_arg_ty(self.builder.bitcast(storage, self.void_ptr_t), ty)


            # Handle array size
            array_size_member = self.builder.gep(llvm_arg,
                    [self.context.get_constant(types.int32, 0), self.context.get_constant(types.int32, 2)])

            ty =  self.resolve_and_return_dpctl_type(types.int64)
            self.form_kernel_arg_and_arg_ty(self.builder.bitcast(array_size_member, self.void_ptr_t), ty)


            # Handle itemsize
            item_size_member = self.builder.gep(llvm_arg,
                    [self.context.get_constant(types.int32, 0), self.context.get_constant(types.int32, 3)])

            ty =  self.resolve_and_return_dpctl_type(types.int64)
            self.form_kernel_arg_and_arg_ty(self.builder.bitcast(item_size_member, self.void_ptr_t), ty)


            # Calculate total buffer size
            total_size = cgutils.alloca_once(self.builder, self.intp_t,
                    size=self.one, name="total_size" + str(self.cur_arg))
            self.builder.store(self.builder.sext(self.builder.mul(self.builder.load(array_size_member),
                               self.builder.load(item_size_member)), self.intp_t), total_size)

            # Handle data
            data_member = self.builder.gep(llvm_arg,
                    [self.context.get_constant(types.int32, 0), self.context.get_constant(types.int32, 4)])

            buffer_name = "buffer_ptr" + str(self.cur_arg)
            buffer_ptr = cgutils.alloca_once(self.builder, self.void_ptr_t,
                                             name=buffer_name)


            args = [self.builder.load(total_size),
                    self.builder.load(self.sycl_queue_val)]
            self.builder.store(self.builder.call(self.usm_shared, args), buffer_ptr)


            # names are replaces usig legalize names, we have to do the same for them to match
            legal_names = legalize_names([var])

            if legal_names[var] in modified_arrays:
                self.write_buffs.append((buffer_ptr, total_size, data_member))
            else:
                self.read_only_buffs.append((buffer_ptr, total_size, data_member))

            # We really need to detect when an array needs to be copied over
            if index < self.num_inputs:
                args = [self.builder.load(self.sycl_queue_val),
                        self.builder.load(buffer_ptr),
                        self.builder.bitcast(self.builder.load(data_member), self.void_ptr_t),
                        self.builder.load(total_size)]
                self.builder.call(self.queue_memcpy, args)


            ty =  self.resolve_and_return_dpctl_type(types.voidptr)
            self.form_kernel_arg_and_arg_ty(self.builder.load(buffer_ptr), ty)

            # Handle shape
            shape_member = self.builder.gep(llvm_arg,
                    [self.context.get_constant(types.int32, 0),
                     self.context.get_constant(types.int32, 5)])

            for this_dim in range(arg_type.ndim):
                shape_entry = self.builder.gep(shape_member,
                                [self.context.get_constant(types.int32, 0),
                                 self.context.get_constant(types.int32, this_dim)])

                ty =  self.resolve_and_return_dpctl_type(types.int64)
                self.form_kernel_arg_and_arg_ty(self.builder.bitcast(shape_entry, self.void_ptr_t), ty)


            # Handle strides
            stride_member = self.builder.gep(llvm_arg,
                    [self.context.get_constant(types.int32, 0),
                     self.context.get_constant(types.int32, 6)])

            for this_stride in range(arg_type.ndim):
                stride_entry = self.builder.gep(stride_member,
                                [self.context.get_constant(types.int32, 0),
                                 self.context.get_constant(types.int32, this_stride)])

                ty =  self.resolve_and_return_dpctl_type(types.int64)
                self.form_kernel_arg_and_arg_ty(self.builder.bitcast(stride_entry, self.void_ptr_t), ty)

        else:
            ty =  self.resolve_and_return_dpctl_type(arg_type)
            self.form_kernel_arg_and_arg_ty(self.builder.bitcast(llvm_arg, self.void_ptr_t), ty)

    def enqueue_kernel_and_read_back(self, loop_ranges):
        # the assumption is loop_ranges will always be less than or equal to 3 dimensions
        num_dim = len(loop_ranges) if len(loop_ranges) < 4 else 3

        # form the global range
        global_range = cgutils.alloca_once(
                        self.builder, self.uintp_t,
                        size=self.context.get_constant(types.uintp, num_dim), name="global_range")

        for i in range(num_dim):
            start, stop, step = loop_ranges[i]
            if stop.type != self.one_type:
                stop = self.builder.sext(stop, self.one_type)

            # we reverse the global range to account for how sycl and opencl range differs
            self.builder.store(stop,
                               self.builder.gep(global_range, [self.context.get_constant(types.uintp, (num_dim-1)-i)]))


        args = [self.builder.inttoptr(self.context.get_constant(types.uintp, self.kernel_addr), self.void_ptr_t),
                self.builder.load(self.sycl_queue_val),
                self.kernel_arg_array,
                self.kernel_arg_ty_array,
                self.context.get_constant(types.uintp, self.total_kernel_args),
                self.builder.bitcast(global_range, self.intp_ptr_t),
                self.context.get_constant(types.uintp, num_dim),
                self.builder.bitcast(self.null_ptr,  self.void_ptr_t),
                self.context.get_constant(types.uintp, 0)
                ]
        self.builder.call(self.submit_range, args)

        self.builder.call(self.queue_wait, [self.builder.load(self.sycl_queue_val)])

        # read buffers back to host
        for write_buff in self.write_buffs:
            buffer_ptr, total_size, data_member = write_buff
            args = [self.builder.load(self.sycl_queue_val),
                    self.builder.bitcast(self.builder.load(data_member), self.void_ptr_t),
                    self.builder.load(buffer_ptr),
                    self.builder.load(total_size)]
            self.builder.call(self.queue_memcpy, args)

            self.builder.call(self.usm_free, [self.builder.load(buffer_ptr), self.builder.load(self.sycl_queue_val)])

        for read_buff in self.read_only_buffs:
            buffer_ptr, total_size, data_member = read_buff
            self.builder.call(self.usm_free, [self.builder.load(buffer_ptr), self.builder.load(self.sycl_queue_val)])
