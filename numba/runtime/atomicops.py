from __future__ import print_function, absolute_import, division

from numba.config import MACHINE_BITS
from numba import cgutils
from llvmlite import ir, binding as llvm


_word_type = ir.IntType(MACHINE_BITS)


def _define_atomic_inc_dec(module, op, ordering):
    """Define a llvm function for atomic increment/decrement to the given module
    Argument ``op`` is the operation "add"/"sub".  Argument ``ordering`` is
    the memory ordering.
    """
    ftype = ir.FunctionType(_word_type, [_word_type.as_pointer()])
    fn_atomic = ir.Function(module, ftype, name="nrt_atomic_{0}".format(op))

    [ptr] = fn_atomic.args
    bb = fn_atomic.append_basic_block()
    builder = ir.IRBuilder(bb)
    ONE = ir.Constant(_word_type, 1)
    builder.atomic_rmw(op, ptr, ONE, ordering=ordering)
    res = builder.load(ptr)
    builder.ret(res)


def _define_atomic_cmpxchg(module, ordering):
    ftype = ir.FunctionType(ir.IntType(32), [_word_type.as_pointer(),
                                             _word_type, _word_type,
                                             _word_type.as_pointer()])
    fn_cas = ir.Function(module, ftype, name="nrt_atomic_cas")

    [ptr, cmp, repl, oldptr] = fn_cas.args
    bb = fn_cas.append_basic_block()
    builder = ir.IRBuilder(bb)
    outtup = builder.cmpxchg(ptr, cmp, repl, ordering=ordering)
    old, ok = cgutils.unpack_tuple(builder, outtup, 2)
    builder.store(old, oldptr)
    builder.ret(builder.zext(ok, ftype.return_type))


def define_atomic_ops():
    module = ir.Module()
    _define_atomic_inc_dec(module, "add", ordering='monotonic')
    _define_atomic_inc_dec(module, "sub", ordering='monotonic')
    _define_atomic_cmpxchg(module, ordering='monotonic')
    return module


def compile_atomic_ops():
    # Implement LLVM module with atomic ops
    mod = llvm.parse_assembly(str(define_atomic_ops()))

    # Create a target just for this module
    target = llvm.Target.from_default_triple()
    mod.triple = target.triple
    tm = target.create_target_machine(cpu=llvm.get_host_cpu_name(),
                                      codemodel='jitdefault')
    mcjit = llvm.create_mcjit_compiler(mod, tm)
    mcjit.finalize_object()
    atomic_inc = mcjit.get_pointer_to_function(mod.get_function(
        "nrt_atomic_add"))
    atomic_dec = mcjit.get_pointer_to_function(mod.get_function(
        "nrt_atomic_sub"))
    atomic_cas = mcjit.get_pointer_to_function(mod.get_function(
        "nrt_atomic_cas"))

    return mcjit, atomic_inc, atomic_dec, atomic_cas
