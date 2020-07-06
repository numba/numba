from __future__ import print_function, division, absolute_import

import dppy.core as driver
import llvmlite.llvmpy.core as lc
import llvmlite.ir.values as liv
import llvmlite.ir as lir
import llvmlite.binding as lb
from .. import types, cgutils

from numba.ir_utils import legalize_names

class DPPyHostFunctionCallsGenerator(object):
    def __init__(self, lowerer, cres, num_inputs):
        self.lowerer = lowerer
        self.context = self.lowerer.context
        self.builder = self.lowerer.builder

        self.current_device = driver.runtime.get_current_device()
        self.current_device_env = self.current_device.get_env_ptr()
        self.current_device_int = int(driver.ffi.cast("uintptr_t",
                                                  self.current_device_env))

        self.kernel_t_obj = cres.kernel._kernel_t_obj[0]
        self.kernel_int = int(driver.ffi.cast("uintptr_t",
                                              self.kernel_t_obj))

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
        self.read_bufs_after_enqueue = []


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
        self.current_device_int_const = self.context.get_constant(
                                            types.uintp,
                                            self.current_device_int)

    def _declare_functions(self):
        create_dppy_kernel_arg_fnty = lc.Type.function(
            self.intp_t,
             [self.void_ptr_ptr_t, self.intp_t, self.void_ptr_ptr_t])

        self.create_dppy_kernel_arg = self.builder.module.get_or_insert_function(create_dppy_kernel_arg_fnty,
                                                              name="create_dp_kernel_arg")

        create_dppy_kernel_arg_from_buffer_fnty = lc.Type.function(
            self.intp_t, [self.void_ptr_ptr_t, self.void_ptr_ptr_t])
        self.create_dppy_kernel_arg_from_buffer = self.builder.module.get_or_insert_function(
                                                   create_dppy_kernel_arg_from_buffer_fnty,
                                                   name="create_dp_kernel_arg_from_buffer")

        create_dppy_rw_mem_buffer_fnty = lc.Type.function(
            self.intp_t, [self.void_ptr_t, self.intp_t, self.void_ptr_ptr_t])
        self.create_dppy_rw_mem_buffer = self.builder.module.get_or_insert_function(
                                          create_dppy_rw_mem_buffer_fnty,
                                          name="create_dp_rw_mem_buffer")

        write_mem_buffer_to_device_fnty = lc.Type.function(
            self.intp_t, [self.void_ptr_t, self.void_ptr_t, self.intp_t, self.intp_t, self.intp_t, self.void_ptr_t])
        self.write_mem_buffer_to_device = self.builder.module.get_or_insert_function(
                                          write_mem_buffer_to_device_fnty,
                                          name="write_dp_mem_buffer_to_device")

        read_mem_buffer_from_device_fnty = lc.Type.function(
            self.intp_t, [self.void_ptr_t, self.void_ptr_t, self.intp_t, self.intp_t, self.intp_t, self.void_ptr_t])
        self.read_mem_buffer_from_device = self.builder.module.get_or_insert_function(
                                        read_mem_buffer_from_device_fnty,
                                        name="read_dp_mem_buffer_from_device")

        enqueue_kernel_fnty = lc.Type.function(
            self.intp_t, [self.void_ptr_t, self.void_ptr_t, self.intp_t, self.void_ptr_ptr_t,
                     self.intp_t, self.intp_ptr_t, self.intp_ptr_t])
        self.enqueue_kernel = self.builder.module.get_or_insert_function(
                                      enqueue_kernel_fnty,
                                      name="set_args_and_enqueue_dp_kernel_auto_blocking")


    def allocate_kenrel_arg_array(self, num_kernel_args):
        self.total_kernel_args = num_kernel_args

        # we need a kernel arg array to enqueue
        self.kernel_arg_array = cgutils.alloca_once(
            self.builder, self.void_ptr_t, size=self.context.get_constant(
                types.uintp, num_kernel_args), name="kernel_arg_array")


    def _call_dppy_kernel_arg_fn(self, args):
        kernel_arg = cgutils.alloca_once(self.builder, self.void_ptr_t,
                                         size=self.one, name="kernel_arg" + str(self.cur_arg))

        args.append(kernel_arg)
        self.builder.call(self.create_dppy_kernel_arg, args)
        dst = self.builder.gep(self.kernel_arg_array, [self.context.get_constant(types.intp, self.cur_arg)])
        self.cur_arg += 1
        self.builder.store(self.builder.load(kernel_arg), dst)


    def process_kernel_arg(self, var, llvm_arg, arg_type, gu_sig, val_type, index, modified_arrays):

        if isinstance(arg_type, types.npytypes.Array):
            if llvm_arg is None:
                raise NotImplementedError(arg_type, var)

            # Handle meminfo.  Not used by kernel so just write a null pointer.
            args = [self.null_ptr, self.context.get_constant(types.uintp, self.sizeof_void_ptr)]
            self._call_dppy_kernel_arg_fn(args)

            # Handle parent.  Not used by kernel so just write a null pointer.
            args = [self.null_ptr, self.context.get_constant(types.uintp, self.sizeof_void_ptr)]
            self._call_dppy_kernel_arg_fn(args)

            # Handle array size
            array_size_member = self.builder.gep(llvm_arg,
                    [self.context.get_constant(types.int32, 0), self.context.get_constant(types.int32, 2)])
            args = [self.builder.bitcast(array_size_member, self.void_ptr_ptr_t),
                    self.context.get_constant(types.uintp, self.sizeof_intp)]
            self._call_dppy_kernel_arg_fn(args)

            # Handle itemsize
            item_size_member = self.builder.gep(llvm_arg,
                    [self.context.get_constant(types.int32, 0), self.context.get_constant(types.int32, 3)])
            args = [self.builder.bitcast(item_size_member, self.void_ptr_ptr_t),
                    self.context.get_constant(types.uintp, self.sizeof_intp)]
            self._call_dppy_kernel_arg_fn(args)

            # Calculate total buffer size
            total_size = cgutils.alloca_once(self.builder, self.intp_t,
                    size=self.one, name="total_size" + str(self.cur_arg))
            self.builder.store(self.builder.sext(self.builder.mul(self.builder.load(array_size_member),
                               self.builder.load(item_size_member)), self.intp_t), total_size)

            # Handle data
            kernel_arg = cgutils.alloca_once(self.builder, self.void_ptr_t,
                    size=self.one, name="kernel_arg" + str(self.cur_arg))
            data_member = self.builder.gep(llvm_arg,
                    [self.context.get_constant(types.int32, 0), self.context.get_constant(types.int32, 4)])

            buffer_name = "buffer_ptr" + str(self.cur_arg)
            buffer_ptr = cgutils.alloca_once(self.builder, self.void_ptr_t,
                                             size=self.one, name=buffer_name)

            # env, buffer_size, buffer_ptr
            args = [self.builder.inttoptr(self.current_device_int_const, self.void_ptr_t),
                    self.builder.load(total_size),
                    buffer_ptr]
            self.builder.call(self.create_dppy_rw_mem_buffer, args)

            # names are replaces usig legalize names, we have to do the same for them to match
            legal_names = legalize_names([var])

            if legal_names[var] in modified_arrays:
                self.read_bufs_after_enqueue.append((buffer_ptr, total_size, data_member))

            # We really need to detect when an array needs to be copied over
            if index < self.num_inputs:
                args = [self.builder.inttoptr(self.current_device_int_const, self.void_ptr_t),
                        self.builder.load(buffer_ptr),
                        self.one,
                        self.zero,
                        self.builder.load(total_size),
                        self.builder.bitcast(self.builder.load(data_member), self.void_ptr_t)]

                self.builder.call(self.write_mem_buffer_to_device, args)

            self.builder.call(self.create_dppy_kernel_arg_from_buffer, [buffer_ptr, kernel_arg])
            dst = self.builder.gep(self.kernel_arg_array, [self.context.get_constant(types.intp, self.cur_arg)])
            self.cur_arg += 1
            self.builder.store(self.builder.load(kernel_arg), dst)

            # Handle shape
            shape_member = self.builder.gep(llvm_arg,
                    [self.context.get_constant(types.int32, 0),
                     self.context.get_constant(types.int32, 5)])

            for this_dim in range(arg_type.ndim):
                shape_entry = self.builder.gep(shape_member,
                                [self.context.get_constant(types.int32, 0),
                                 self.context.get_constant(types.int32, this_dim)])

                args = [self.builder.bitcast(shape_entry, self.void_ptr_ptr_t),
                        self.context.get_constant(types.uintp, self.sizeof_intp)]
                self._call_dppy_kernel_arg_fn(args)

            # Handle strides
            stride_member = self.builder.gep(llvm_arg,
                    [self.context.get_constant(types.int32, 0),
                     self.context.get_constant(types.int32, 6)])

            for this_stride in range(arg_type.ndim):
                stride_entry = self.builder.gep(stride_member,
                                [self.context.get_constant(types.int32, 0),
                                 self.context.get_constant(types.int32, this_stride)])

                args = [self.builder.bitcast(stride_entry, self.void_ptr_ptr_t),
                        self.context.get_constant(types.uintp, self.sizeof_intp)]
                self._call_dppy_kernel_arg_fn(args)

        else:
            args = [self.builder.bitcast(llvm_arg, self.void_ptr_ptr_t),
                    self.context.get_constant(types.uintp, self.context.get_abi_sizeof(val_type))]
            self._call_dppy_kernel_arg_fn(args)

    def enqueue_kernel_and_read_back(self, loop_ranges):
        # the assumption is loop_ranges will always be less than or equal to 3 dimensions
        num_dim = len(loop_ranges) if len(loop_ranges) < 4 else 3

        # Package dim start and stops for auto-blocking enqueue.
        dim_starts = cgutils.alloca_once(
                        self.builder, self.uintp_t,
                        size=self.context.get_constant(types.uintp, num_dim), name="dims")

        dim_stops = cgutils.alloca_once(
                        self.builder, self.uintp_t,
                        size=self.context.get_constant(types.uintp, num_dim), name="dims")

        for i in range(num_dim):
            start, stop, step = loop_ranges[i]
            if start.type != self.one_type:
                start = self.builder.sext(start, self.one_type)
            if stop.type != self.one_type:
                stop = self.builder.sext(stop, self.one_type)
            if step.type != self.one_type:
                step = self.builder.sext(step, self.one_type)

            # substract 1 because do-scheduling takes inclusive ranges
            stop = self.builder.sub(stop, self.one)

            self.builder.store(start,
                               self.builder.gep(dim_starts, [self.context.get_constant(types.uintp, i)]))
            self.builder.store(stop,
                               self.builder.gep(dim_stops, [self.context.get_constant(types.uintp, i)]))

        args = [self.builder.inttoptr(self.current_device_int_const, self.void_ptr_t),
                self.builder.inttoptr(self.context.get_constant(types.uintp, self.kernel_int), self.void_ptr_t),
                self.context.get_constant(types.uintp, self.total_kernel_args),
                self.kernel_arg_array,
                self.context.get_constant(types.uintp, num_dim),
                dim_starts,
                dim_stops]

        self.builder.call(self.enqueue_kernel, args)

        # read buffers back to host
        for read_buf in self.read_bufs_after_enqueue:
            buffer_ptr, array_size_member, data_member = read_buf
            args = [self.builder.inttoptr(self.current_device_int_const, self.void_ptr_t),
                    self.builder.load(buffer_ptr),
                    self.one,
                    self.zero,
                    self.builder.load(array_size_member),
                    self.builder.bitcast(self.builder.load(data_member), self.void_ptr_t)]
            self.builder.call(self.read_mem_buffer_from_device, args)
