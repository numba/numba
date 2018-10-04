from __future__ import print_function, absolute_import

import os
import sys

from llvmlite import ir
from llvmlite.llvmpy.core import Constant, Type

from numba import gdb, types, utils, cgutils, config
from numba.extending import overload, intrinsic


@overload(gdb)
def hook_gdb(*args):
    if not sys.platform.startswith('linux'):
        raise RuntimeError('gdb is only available on linux')
    gdbloc = config.GDB_BINARY
    if not (os.path.exists(gdbloc) and os.path.isfile(gdbloc)):
        msg = ('Is gdb present? Location specified (%s) does not exist. The gdb'
               ' binary location can be set using Numba configuration, see: '
               'http://numba.pydata.org/numba-doc/latest/reference/envvars.html'
               )
        raise RuntimeError(msg % config.GDB_BINARY)

    gdbimpl = gen_impl(args)

    def impl(*args):
        gdbimpl()
    return impl


def gen_impl(const_args):
    int8_t = ir.IntType(8)
    int32_t = ir.IntType(32)
    intp_t = ir.IntType(utils.MACHINE_BITS)
    char_ptr = Type.pointer(Type.int(8))
    zero_i32t = int32_t(0)

    @intrinsic(support_literals=True)
    def gdb_internal(tyctx):
        function_sig = types.void()

        def codegen(cgctx, builder, signature, args):
            pyapi = cgctx.get_python_api(builder)

            mod = builder.module
            pid = cgutils.alloca_once(builder, int32_t, size=1)

            # 32bit pid, 11 char max + terminator
            pidstr = cgutils.alloca_once(builder, int8_t, size=12)

            # str consts
            intfmt = cgctx.insert_const_string(mod, '%d')
            gdb_str = cgctx.insert_const_string(mod, config.GDB_BINARY)
            attach_str = cgctx.insert_const_string(mod, 'attach')
            cmdlang = [cgctx.insert_const_string(
                mod, x.value) for x in const_args]

            # insert getpid, getpid is always successful, call without concern!
            fnty = ir.FunctionType(int32_t, tuple())
            getpid = mod.get_or_insert_function(fnty, "getpid")

            # insert snprintf
            # int snprintf(char *str, size_t size, const char *format, ...);
            fnty = ir.FunctionType(
                int32_t, (char_ptr, intp_t, char_ptr), var_arg=True)
            snprintf = mod.get_or_insert_function(fnty, "snprintf")

            # insert fork
            fnty = ir.FunctionType(int32_t, tuple())
            fork = mod.get_or_insert_function(fnty, "fork")

            # insert execl
            fnty = ir.FunctionType(int32_t, (char_ptr, char_ptr), var_arg=True)
            execl = mod.get_or_insert_function(fnty, "execl")

            # insert sleep
            fnty = ir.FunctionType(int32_t, (int32_t,))
            sleep = mod.get_or_insert_function(fnty, "sleep")

            # do the work
            parent_pid = builder.call(getpid, tuple())
            builder.store(parent_pid, pid)
            pidstr_ptr = builder.gep(pidstr, [zero_i32t], inbounds=True)
            pid_val = builder.load(pid)

            # call snprintf to write the pid into a char *
            stat = builder.call(
                snprintf, (pidstr_ptr, intp_t(12), intfmt, pid_val))
            invalid_write = builder.icmp_signed('>', stat, int32_t(12))
            with builder.if_then(invalid_write, likely=False):
                msg = "Internal error: `snprintf` buffer would have overflowed."
                cgctx.call_conv.return_user_exc(builder, RuntimeError, (msg,))

            # fork, check pids etc
            child_pid = builder.call(fork, tuple())
            fork_failed = builder.icmp_signed('==', child_pid, int32_t(-1))
            with builder.if_then(fork_failed, likely=False):
                msg = "Internal error: `fork` failed."
                cgctx.call_conv.return_user_exc(builder, RuntimeError, (msg,))

            is_child = builder.icmp_signed('==', child_pid, zero_i32t)
            with builder.if_else(is_child) as (then, orelse):
                with then:
                    # is child
                    my_pid = builder.call(getpid, tuple())
                    nullptr = Constant.null(char_ptr)
                    gdb_str_ptr = builder.gep(
                        gdb_str, [zero_i32t], inbounds=True)
                    attach_str_ptr = builder.gep(
                        attach_str, [zero_i32t], inbounds=True)
                    cgutils.printf(
                        builder, "Attaching to PID: %s\n", pidstr_ptr)
                    builder.call(execl, (gdb_str_ptr, gdb_str_ptr,
                                         attach_str_ptr, pidstr_ptr, *cmdlang,
                                         nullptr))
                with orelse:
                    # is parent
                    builder.call(sleep, (int32_t(2),))

            return cgctx.get_constant(types.none, None)
        return function_sig, codegen
    return gdb_internal
