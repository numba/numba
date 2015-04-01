from __future__ import print_function, absolute_import, division

from numba.config import MACHINE_BITS
from llvmlite import ir, binding as llvm


def _define_atomic_func(module, op, ordering):
    word_type = ir.IntType(MACHINE_BITS)
    ftype = ir.FunctionType(word_type, [word_type.as_pointer()])
    fn_atomic = ir.Function(module, ftype, name="nrt_atomic_{0}".format(op))

    [ptr] = fn_atomic.args
    bb = fn_atomic.append_basic_block()
    builder = ir.IRBuilder(bb)
    ONE = ir.Constant(word_type, 1)
    builder.atomic_rmw(op, ptr, ONE, ordering=ordering)
    res = builder.load(ptr)
    builder.ret(res)


def define_atomic_ops():
    module = ir.Module()
    _define_atomic_func(module, "add", ordering='monotonic')
    _define_atomic_func(module, "sub", ordering='monotonic')
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

    return mcjit, atomic_inc, atomic_dec
