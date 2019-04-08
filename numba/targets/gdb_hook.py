from __future__ import print_function, absolute_import

import os
import sys

from llvmlite import ir

from numba import gdb, gdb_init, gdb_breakpoint, types, utils, cgutils, config
from numba.extending import overload, intrinsic

_path = os.path.dirname(__file__)

_platform = sys.platform
_unix_like = (_platform.startswith('linux') or
              _platform.startswith('darwin') or
              ('bsd' in _platform))


def _confirm_gdb():
    if not _unix_like:
        raise RuntimeError('gdb support is only available on unix-like systems')
    gdbloc = config.GDB_BINARY
    if not (os.path.exists(gdbloc) and os.path.isfile(gdbloc)):
        msg = ('Is gdb present? Location specified (%s) does not exist. The gdb'
               ' binary location can be set using Numba configuration, see: '
               'http://numba.pydata.org/numba-doc/latest/reference/envvars.html'
               )
        raise RuntimeError(msg % config.GDB_BINARY)
    # Is Yama being used as a kernel security module and if so is ptrace_scope
    # limited? In this case ptracing non-child processes requires special
    # permission so raise an exception.
    ptrace_scope_file = os.path.join(os.sep, 'proc', 'sys', 'kernel', 'yama',
                                     'ptrace_scope')
    has_ptrace_scope = os.path.exists(ptrace_scope_file)
    if has_ptrace_scope:
        with open(ptrace_scope_file, 'rt') as f:
            value = f.readline().strip()
        if value != "0":
            msg = ("gdb can launch but cannot attach to the executing program"
                   " because ptrace permissions have been restricted at the "
                   "system level by the Linux security module 'Yama'.\n\n"
                   "Documentation for this module and the security "
                   "implications of making changes to its behaviour can be "
                   "found in the Linux Kernel documentation "
                   "https://www.kernel.org/doc/Documentation/admin-guide/LSM/Yama.rst"
                   "\n\nDocumentation on how to adjust the behaviour of Yama "
                   "on Ubuntu Linux with regards to 'ptrace_scope' can be "
                   "found here "
                   "https://wiki.ubuntu.com/Security/Features#ptrace.")
            raise RuntimeError(msg)


@overload(gdb)
def hook_gdb(*args):
    _confirm_gdb()
    gdbimpl = gen_gdb_impl(args, True)

    def impl(*args):
        gdbimpl()
    return impl


@overload(gdb_init)
def hook_gdb_init(*args):
    _confirm_gdb()
    gdbimpl = gen_gdb_impl(args, False)

    def impl(*args):
        gdbimpl()
    return impl


def init_gdb_codegen(cgctx, builder, signature, args,
                     const_args, do_break=False):

    int8_t = ir.IntType(8)
    void_ptr = ir.PointerType(int8_t)
    int32_t = ir.IntType(32)
    int32_t_ptr = ir.PointerType(ir.IntType(32))
    intp_t = ir.IntType(utils.MACHINE_BITS)
    char_ptr = ir.PointerType(ir.IntType(8))
    zero_i32t = int32_t(0)
    one_i32t = int32_t(1)

    mod = builder.module
    pid = cgutils.alloca_once(builder, int32_t, size=1)

    # 32bit pid, 11 char max + terminator
    pidstr = cgutils.alloca_once(builder, int8_t, size=12)
    # 32bit pid, 11 char max + cmd + terminator = 11 + 7+, make it 20
    callclose_str = cgutils.alloca_once(builder, int8_t, size=20)
    # pointless read buffer, nothing should ever actually be read into it
    read_buffer = cgutils.alloca_once(builder, int8_t, size=1)

    # pipe file descriptors
    pipefds = cgutils.alloca_once(builder, int32_t, size=2)

    # str consts
    intfmt = cgctx.insert_const_string(mod, '%d')
    gdb_str = cgctx.insert_const_string(mod, config.GDB_BINARY)
    attach_str = cgctx.insert_const_string(mod, 'attach')
    callclosefmt = cgctx.insert_const_string(mod, 'call close(%d)')

    new_args = []
    # add break point command to known location
    # this command file thing is due to commands attached to a breakpoint
    # requiring an interactive prompt
    # https://sourceware.org/bugzilla/show_bug.cgi?id=10079
    new_args.extend(['-x', os.path.join(_path, 'cmdlang.gdb')])
    # issue command to continue execution after read() unblocks
    new_args.extend(['-ex', 'c'])
    # then run the user defined args if any
    new_args.extend([x.literal_value for x in const_args])
    cmdlang = [cgctx.insert_const_string(mod, x) for x in new_args]

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

    # insert pipe
    fnty = ir.FunctionType(int32_t, (int32_t_ptr,))
    pipe = mod.get_or_insert_function(fnty, "pipe")

    # insert read
    fnty = ir.FunctionType(int32_t, (int32_t, void_ptr, intp_t))
    read = mod.get_or_insert_function(fnty, "read")

    # insert close
    fnty = ir.FunctionType(int32_t, (int32_t,))
    close = mod.get_or_insert_function(fnty, "close")

    # insert execl
    fnty = ir.FunctionType(int32_t, (char_ptr, char_ptr), var_arg=True)
    execl = mod.get_or_insert_function(fnty, "execl")

    # insert break point
    fnty = ir.FunctionType(ir.VoidType(), tuple())
    breakpoint = mod.get_or_insert_function(fnty, "numba_gdb_breakpoint")

    # do the work
    # call getpid
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

    # call pipe to set up fds
    pipefds_ptr = builder.gep(pipefds, [zero_i32t], inbounds=True)
    stat = builder.call(pipe, (pipefds_ptr,))
    pipe_failed = builder.icmp_signed('==', stat, int32_t(-1))
    with builder.if_then(pipe_failed, likely=False):
        msg = "Internal error: `pipe` failed."
        cgctx.call_conv.return_user_exc(builder, RuntimeError, (msg,))

    # pointers to the read and write end FDs
    read_end = builder.gep(pipefds, [zero_i32t], inbounds=True)
    write_end = builder.gep(pipefds, [one_i32t], inbounds=True)

    # this writes a command like "close(write_end_fd)" into the text buffer that
    # will be one of the const char * that goes into the execl call made to
    # gdb with view of a command like `-ex "close(write_end_fd)"` being executed
    stat = builder.call(snprintf, (callclose_str, intp_t(20), callclosefmt,
                                   builder.load(write_end)))
    invalid_write = builder.icmp_signed('>', stat, int32_t(20))
    with builder.if_then(invalid_write, likely=False):
        msg = "Internal error: `snprintf` buffer would have overflowed."
        cgctx.call_conv.return_user_exc(builder, RuntimeError, (msg,))

    is_child = builder.icmp_signed('==', child_pid, zero_i32t)
    with builder.if_else(is_child) as (then, orelse):
        with then:
            # is child

            # first close the fork-inherited pipe ends, doesn't matter if they
            # return something
            builder.call(close, (builder.load(write_end),))
            builder.call(close, (builder.load(read_end),))

            # stage the gdb call
            nullptr = ir.Constant(char_ptr, None)
            gdb_str_ptr = builder.gep(
                gdb_str, [zero_i32t], inbounds=True)
            attach_str_ptr = builder.gep(
                attach_str, [zero_i32t], inbounds=True)
            callclose_str_ptr = builder.gep(
                callclose_str, [zero_i32t], inbounds=True)
            cgutils.printf(
                builder, "Attaching to PID: %s\n", pidstr)
            buf = (gdb_str_ptr,    # gdb binary w path
                   gdb_str_ptr,    # gdb binary w path
                   attach_str_ptr, # the "attach" command
                   pidstr_ptr,     # PID to attach to (parent, from above)
                   cgctx.insert_const_string(mod, '-ex'), # str "-ex"
                   callclose_str_ptr, # the command "close(write_end_fd)"
                   )
            cgutils.printf(builder,"%s\n", callclose_str)
            buf = buf + tuple(cmdlang) + (nullptr,)
            builder.call(execl, buf)
        with orelse:
            # is parent
            # issue a call to read() from the read-end of the pipe, this will
            # block
            builder.call(read, (builder.load(read_end), read_buffer, intp_t(1)))
            # if breaking is desired, break now
            if do_break is True:
                builder.call(breakpoint, tuple())


def gen_gdb_impl(const_args, do_break):
    @intrinsic
    def gdb_internal(tyctx):
        function_sig = types.void()

        def codegen(cgctx, builder, signature, args):
            init_gdb_codegen(cgctx, builder, signature, args, const_args,
                             do_break=do_break)
            return cgctx.get_constant(types.none, None)
        return function_sig, codegen
    return gdb_internal


@overload(gdb_breakpoint)
def hook_gdb_breakpoint():
    """
    Adds the Numba break point into the source
    """
    if not sys.platform.startswith('linux'):
        raise RuntimeError('gdb is only available on linux')
    bp_impl = gen_bp_impl()

    def impl():
        bp_impl()
    return impl


def gen_bp_impl():
    @intrinsic
    def bp_internal(tyctx):
        function_sig = types.void()

        def codegen(cgctx, builder, signature, args):
            mod = builder.module
            fnty = ir.FunctionType(ir.VoidType(), tuple())
            breakpoint = mod.get_or_insert_function(fnty,
                                                    "numba_gdb_breakpoint")
            builder.call(breakpoint, tuple())
            return cgctx.get_constant(types.none, None)
        return function_sig, codegen
    return bp_internal
