from __future__ import print_function, absolute_import, division

import re
from collections import defaultdict, deque

from numba.config import MACHINE_BITS
from numba import cgutils
from llvmlite import ir, binding as llvm

# Flag to enable debug print in NRT_incref and NRT_decref
_debug_print = False

_word_type = ir.IntType(MACHINE_BITS)
_pointer_type = ir.PointerType(ir.IntType(8))

_meminfo_struct_type = ir.LiteralStructType([
    _word_type,     # size_t refct
    _pointer_type,  # dtor_function dtor
    _pointer_type,  # void *dtor_info
    _pointer_type,  # void *data
    _word_type,     # size_t size
    ])


incref_decref_ty = ir.FunctionType(ir.VoidType(), [_pointer_type])
meminfo_data_ty = ir.FunctionType(_pointer_type, [_pointer_type])


def _define_nrt_meminfo_data(module):
    """
    Implement NRT_MemInfo_data_fast in the module.  This allows LLVM
    to inline lookup of the data pointer.
    """
    fn = module.get_or_insert_function(meminfo_data_ty,
                                       name="NRT_MemInfo_data_fast")
    builder = ir.IRBuilder(fn.append_basic_block())
    [ptr] = fn.args
    struct_ptr = builder.bitcast(ptr, _meminfo_struct_type.as_pointer())
    data_ptr = builder.load(cgutils.gep(builder, struct_ptr, 0, 3))
    builder.ret(data_ptr)


def _define_nrt_incref(module, atomic_incr):
    """
    Implement NRT_incref in the module
    """
    fn_incref = module.get_or_insert_function(incref_decref_ty,
                                              name="NRT_incref")
    builder = ir.IRBuilder(fn_incref.append_basic_block())
    [ptr] = fn_incref.args
    is_null = builder.icmp_unsigned("==", ptr, cgutils.get_null_value(ptr.type))
    with cgutils.if_unlikely(builder, is_null):
        builder.ret_void()

    if _debug_print:
        cgutils.printf(builder, "*** NRT_Incref %zu [%p]\n", builder.load(ptr),
                       ptr)
    builder.call(atomic_incr, [builder.bitcast(ptr, atomic_incr.args[0].type)])
    builder.ret_void()


def _define_nrt_decref(module, atomic_decr):
    """
    Implement NRT_decref in the module
    """
    fn_decref = module.get_or_insert_function(incref_decref_ty,
                                              name="NRT_decref")
    calldtor = module.add_function(ir.FunctionType(ir.VoidType(), [_pointer_type]),
                                   name="NRT_MemInfo_call_dtor")

    builder = ir.IRBuilder(fn_decref.append_basic_block())
    [ptr] = fn_decref.args
    is_null = builder.icmp_unsigned("==", ptr, cgutils.get_null_value(ptr.type))
    with cgutils.if_unlikely(builder, is_null):
        builder.ret_void()

    if _debug_print:
        cgutils.printf(builder, "*** NRT_Decref %zu [%p]\n", builder.load(ptr),
                       ptr)
    newrefct = builder.call(atomic_decr,
                            [builder.bitcast(ptr, atomic_decr.args[0].type)])

    refct_eq_0 = builder.icmp_unsigned("==", newrefct,
                                       ir.Constant(newrefct.type, 0))
    with cgutils.if_unlikely(builder, refct_eq_0):
        builder.call(calldtor, [ptr])
    builder.ret_void()


# Set this to True to measure the overhead of atomic refcounts compared
# to non-atomic.
_disable_atomicity = 0


def _define_atomic_inc_dec(module, op, ordering):
    """Define a llvm function for atomic increment/decrement to the given module
    Argument ``op`` is the operation "add"/"sub".  Argument ``ordering`` is
    the memory ordering.  The generated function returns the new value.
    """
    ftype = ir.FunctionType(_word_type, [_word_type.as_pointer()])
    fn_atomic = ir.Function(module, ftype, name="nrt_atomic_{0}".format(op))

    [ptr] = fn_atomic.args
    bb = fn_atomic.append_basic_block()
    builder = ir.IRBuilder(bb)
    ONE = ir.Constant(_word_type, 1)
    if not _disable_atomicity:
        oldval = builder.atomic_rmw(op, ptr, ONE, ordering=ordering)
        # Perform the operation on the old value so that we can pretend returning
        # the "new" value.
        res = getattr(builder, op)(oldval, ONE)
        builder.ret(res)
    else:
        oldval = builder.load(ptr)
        newval = getattr(builder, op)(oldval, ONE)
        builder.store(newval, ptr)
        builder.ret(oldval)

    return fn_atomic


def _define_atomic_cas(module, ordering):
    """Define a llvm function for atomic compare-and-swap.
    The generated function is a direct wrapper of the LLVM cmpxchg with the
    difference that the a int indicate success (1) or failure (0) is returned
    and the last argument is a output pointer for storing the old value.

    Note
    ----
    On failure, the generated function behaves like an atomic load.  The loaded
    value is stored to the last argument.
    """
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

    return fn_cas


def create_nrt_module(ctx):
    """
    Create an IR module defining the LLVM NRT functions.
    A (IR module, library) tuple is returned.
    """
    codegen = ctx.codegen()
    library = codegen.create_library("nrt")

    # Implement LLVM module with atomic ops
    ir_mod = library.create_ir_module("nrt_module")

    atomic_inc = _define_atomic_inc_dec(ir_mod, "add", ordering='monotonic')
    atomic_dec = _define_atomic_inc_dec(ir_mod, "sub", ordering='monotonic')
    _define_atomic_cas(ir_mod, ordering='monotonic')

    _define_nrt_meminfo_data(ir_mod)
    _define_nrt_incref(ir_mod, atomic_inc)
    _define_nrt_decref(ir_mod, atomic_dec)

    return ir_mod, library


def compile_nrt_functions(ctx):
    """
    Compile all LLVM NRT functions and return a library containing them.
    The library is created using the given target context.
    """
    ir_mod, library = create_nrt_module(ctx)

    library.add_ir_module(ir_mod)
    library.finalize()

    return library


_regex_incref = re.compile(r'\s*call void @NRT_incref\((.*)\)')
_regex_decref = re.compile(r'\s*call void @NRT_decref\((.*)\)')
_regex_bb = re.compile(r'([-a-zA-Z$._][-a-zA-Z$._0-9]*:)|^define')


def remove_redundant_nrt_refct(ll_module):
    """
    Remove redundant reference count operations from the
    `llvmlite.binding.ModuleRef`. This parses the ll_module as a string and
    line by line to remove the unnecessary nrt refct pairs within each block.
    Decref calls are moved after the last incref call in the block to avoid
    temporarily decref'ing to zero (which can happen due to hidden decref from
    alias).

    Note: non-threadsafe due to usage of global LLVMcontext
    """
    # Note: As soon as we have better utility in analyzing materialized LLVM
    #       module in llvmlite, we can redo this without so much string
    #       processing.
    def _extract_functions(module):
        cur = []
        for line in str(module).splitlines():
            if line.startswith('define'):
                # start of function
                assert not cur
                cur.append(line)
            elif line.startswith('}'):
                # end of function
                assert cur
                cur.append(line)
                yield True, cur
                cur = []
            elif cur:
                cur.append(line)
            else:
                yield False, [line]

    def _process_function(func_lines):
        out = []
        for is_bb, bb_lines in _extract_basic_blocks(func_lines):
            if is_bb and bb_lines:
                bb_lines = _process_basic_block(bb_lines)
            out += bb_lines
        return out

    def _extract_basic_blocks(func_lines):
        assert func_lines[0].startswith('define')
        assert func_lines[-1].startswith('}')
        yield False, [func_lines[0]]

        cur = []
        for ln in func_lines[1:-1]:
            m = _regex_bb.match(ln)
            if m is not None:
                # line is a basic block separator
                yield True, cur
                cur = []
                yield False, [ln]
            elif ln:
                cur.append(ln)

        yield True, cur
        yield False, [func_lines[-1]]

    def _process_basic_block(bb_lines):
        bb_lines = _move_and_group_decref_after_all_increfs(bb_lines)
        bb_lines = _prune_redundant_refct_ops(bb_lines)
        return bb_lines

    def _examine_refct_op(bb_lines):
        for num, ln in enumerate(bb_lines):
            m = _regex_incref.match(ln)
            if m is not None:
                yield num, m.group(1), None
                continue

            m = _regex_decref.match(ln)
            if m is not None:
                yield num, None, m.group(1)
                continue

            yield ln, None, None

    def _prune_redundant_refct_ops(bb_lines):
        incref_map = defaultdict(deque)
        decref_map = defaultdict(deque)
        for num, incref_var, decref_var in _examine_refct_op(bb_lines):
            assert not (incref_var and decref_var)
            if incref_var:
                incref_map[incref_var].append(num)
            elif decref_var:
                decref_map[decref_var].append(num)

        to_remove = set()
        for var, decops in decref_map.items():
            incops = incref_map[var]
            ct = min(len(incops), len(decops))
            for _ in range(ct):
                to_remove.add(incops.pop())
                to_remove.add(decops.popleft())

        return [ln for num, ln in enumerate(bb_lines)
                if num not in to_remove]

    def _move_and_group_decref_after_all_increfs(bb_lines):
        # find last incref
        last_incref_pos = 0
        for pos, ln in enumerate(bb_lines):
            if _regex_incref.match(ln) is not None:
                last_incref_pos = pos + 1

        # find last decref
        last_decref_pos = 0
        for pos, ln in enumerate(bb_lines):
            if _regex_decref.match(ln) is not None:
                last_decref_pos = pos + 1

        last_pos = max(last_incref_pos, last_decref_pos)

        # find decrefs before last_pos
        decrefs = []
        head = []
        for ln in bb_lines[:last_pos]:
            if _regex_decref.match(ln) is not None:
                decrefs.append(ln)
            else:
                head.append(ln)

        # insert decrefs at last_pos
        return head + decrefs + bb_lines[last_pos:]


    # Early escape if NRT_incref is not used
    try:
        ll_module.get_function('NRT_incref')
    except NameError:
        return ll_module

    processed = []

    for is_func, lines in _extract_functions(ll_module):
        if is_func:
            lines = _process_function(lines)

        processed += lines

    newll = '\n'.join(processed)
    return llvm.parse_assembly(newll)
